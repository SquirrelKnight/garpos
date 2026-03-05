'''
Created:
	07/01/2020 by S. Watanabe
Contains:
	remove_outlier_mad
	calc_atd_vectorized
	calc_forward
	calc_gamma
	jacobian_gamma
	jacobian_pos
Modified:
    2026-03-07 by Hutchinson
        Changed outlier removal from standard deviation to median average deviation.
        Vectorized attitude corrections and calculation of the gamma matrices.
'''

import numpy as np
from scipy.interpolate import BSpline
from scipy.sparse import csr_matrix
from scipy.stats import median_abs_deviation as mad
from scipy.spatial.transform import Rotation as R

# garpos module
from .coordinate_trans import corr_attitude
from .traveltime import calc_traveltime

def remove_outliers_mad(dat_in, rsig):
    """Flag outliers based on the median absolute deviation"""
    # Use nanmedian to safely ignore any existing NaNs in the data
    resi_vals = dat_in['ResiTT'].values
    mad_val = mad(resi_vals, nan_policy='omit')
    med = np.nanmedian(resi_vals)
    
    med_deviation = np.abs(resi_vals - med) / mad_val
    dat_in['flag'] = (med_deviation > rsig) | dat_in['iniflag']
    
    return dat_in

def calc_atd_vectorized(pl, hd, rl, pc):
    """
    Vectorization for attitude corrections.
    Replaces the np.vectorize(corr_attitude) loop.
    """
    # GARPOS original code uses extrinsic ZYX rotations (Heading -> Pitch -> Roll)
    rots = R.from_euler('ZYX', np.column_stack((hd, pc, rl)), degrees=True)
    
    # Apply rotation to the intrinsic lever arm [Forward, Rightward, Downward]
    offsets = rots.apply(pl)
    
    # Map back to GARPOS ENU standards:
    # pole_de (East)  = Y axis (index 1)
    # pole_dn (North) = X axis (index 0)
    # pole_du (Up)    = -Z axis (negative index 2)
    return offsets[:, 1], offsets[:, 0], -offsets[:, 2]
    
def calc_forward(shots, mp, nMT, icfg, svp, T0):
	"""
	Calculate the forward modeling of observation eqs.

	Parameters
	----------
	shots : DataFrame
		GNSS-A shot dataset.
	mp : ndarray
		complete model parameter vector.
	nMT : int
		number of transponders.
	icfg : configparser
		Config file for inversion conditions.
	svp : DataFrame
		Sound speed profile.
	T0 : float
		Typical travel time.

	Returns
	-------
	shots : DataFrame
		GNSS-A shot dataset in which calculated data is added.
	"""

    rsig = float(icfg.get("Inv-parameter","RejectCriteria"))

    # Vectorized ATD offset
    pl = np.array([mp[(nMT+1)*3+0], mp[(nMT+1)*3+1], mp[(nMT+1)*3+2]])
    
    ple0, pln0, plu0 = calc_atd_vectorized(pl, shots.head0.values, shots.roll0.values, shots.pitch0.values)
    ple1, pln1, plu1 = calc_atd_vectorized(pl, shots.head1.values, shots.roll1.values, shots.pitch1.values)
    
    shots['ple0'], shots['pln0'], shots['plu0'] = ple0, pln0, plu0
    shots['ple1'], shots['pln1'], shots['plu1'] = ple1, pln1, plu1

    # calc Residuals
    cTT, cTO = calc_traveltime(shots, mp, nMT, icfg, svp)
    logTTc = np.log( cTT/T0 ) - shots.gamma.values
    ResiTT = shots.logTT.values - logTTc

    shots['calcTT'] = cTT
    shots['TakeOff'] = cTO
    shots['logTTc'] = logTTc
    shots['ResiTT'] = ResiTT
    shots['ResiTTreal'] = ResiTT * shots.TT.values

    if rsig > 0.1:
        shots = remove_outliers_mad(shots, rsig)
        # Note: Dead code calculating unused mean/std was removed here

    return shots


def calc_gamma(mp, shotdat, imp0, spdeg, knots):
	"""
	Calculate correction value "gamma" in the observation eqs.

	Parameters
	----------
	mp : ndarray
		complete model parameter vector.
	shotdat : DataFrame
		GNSS-A shot dataset.
	imp0 : ndarray (len=5)
		Indices where the type of model parameters change.
	p : int
		spline degree (=3).
	knots : list of ndarray (len=5)
		B-spline knots for each component in "gamma".

	Returns
	-------
	gamma : ndarray
		Values of "gamma". Note that scale facter is not applied.
	a : 2-d list of ndarray
		[a0[<alpha>], a1[<alpha>]] :: a[<alpha>] at transmit/received time.
		<alpha> is corresponding to <0>, <1E>, <1N>, <2E>, <2N>.
	"""

    a0, a1 = [], []
    
    st_vals = shotdat.ST.values
    rt_vals = shotdat.RT.values
    
    for k, kn in enumerate(knots):
        if len(kn) == 0:
            a0.append(np.zeros(len(st_vals)))
            a1.append(np.zeros(len(rt_vals)))
            continue
        ct = mp[imp0[k]:imp0[k+1]]
        bs = BSpline(kn, ct, spdeg, extrapolate=False)
        a0.append(bs(st_vals))
        a1.append(bs(rt_vals))

    ls = 1000.  # m/s/m to m/s/km order for gradient

    de0, de1 = shotdat.de0.values, shotdat.de1.values
    dn0, dn1 = shotdat.dn0.values, shotdat.dn1.values
    mte, mtn = shotdat.mtde.values, shotdat.mtdn.values

    gamma0_0 =  a0[0]
    gamma0_1 = (a0[1] * de0 + a0[2] * dn0) / ls
    gamma0_2 = (a0[3] * mte + a0[4] * mtn) / ls

    gamma1_0 =  a1[0]
    gamma1_1 = (a1[1] * de1 + a1[2] * dn1) / ls
    gamma1_2 = (a1[3] * mte + a1[4] * mtn) / ls

    gamma0 = gamma0_0 + gamma0_1 + gamma0_2
    gamma1 = gamma1_0 + gamma1_1 + gamma1_2

    return (gamma0 + gamma1) / 2., [a0, a1]

def jacobian_gamma(shotdat, imp0, spdeg, knots):
    """
    Vectorized calculation of the gamma Jacobian using BSpline.design_matrix.
    Replaces the massive for-loop over control points.
    """
    from scipy.interpolate import BSpline
    
    ndata = len(shotdat)
    n_gamma_params = imp0[-1] - imp0[0]
    
    # Pre-allocate the full Jacobian block for gamma
    jcb_gamma = np.zeros((n_gamma_params, ndata))
    
    ls = 1000.0  # m/s/m to m/s/km order for gradient
    
    de0, de1 = shotdat.de0.values, shotdat.de1.values
    dn0, dn1 = shotdat.dn0.values, shotdat.dn1.values
    mte, mtn = shotdat.mtde.values, shotdat.mtdn.values
    
    # Multipliers for the 5 components: a0, a1E, a1N, a2E, a2N
    mult0 = [np.ones(ndata), de0 / ls, dn0 / ls, mte / ls, mtn / ls]
    mult1 = [np.ones(ndata), de1 / ls, dn1 / ls, mte / ls, mtn / ls]
    
    st_vals = shotdat.ST.values
    rt_vals = shotdat.RT.values
    
    current_row = 0
    for k, kn in enumerate(knots):
        if len(kn) == 0:
            continue
        
        n_params = imp0[k+1] - imp0[k]
        
        # Build strictly sparse design matrices (shape: ndata x n_params)
        B0 = BSpline.design_matrix(st_vals, kn, spdeg)
        B1 = BSpline.design_matrix(rt_vals, kn, spdeg)
        
        # Transpose to (n_params, ndata) and broadcast-multiply the spatial offsets row-by-row
        # In sparse math, .multiply() safely performs element-wise broadcasting!
        term0 = B0.T.multiply(mult0[k])
        term1 = B1.T.multiply(mult1[k])
        
        # Combine ST and RT terms (gamma = (gamma0 + gamma1) / 2)
        jcb_block = (term0 + term1) / 2.0
        
        # Dump the sparse block directly into the dense array
        jcb_gamma[current_row : current_row + n_params, :] = jcb_block.toarray()
        current_row += n_params
        
    return jcb_gamma

def jacobian_pos(icfg, mp, slvidx0, shotdat, mtidx, svp, T0):
	"""
	Calculate Jacobian matrix for positions.

	Parameters
	----------
	icfg : configparser
		Config file for inversion conditions.
	mp : ndarray
		complete model parameter vector.
	slvidx0 : list
		Indices of model parameters to be solved.
	shotdat : DataFrame
		GNSS-A shot dataset.
	mtidx : dictionary
		Indices of mp for each MT.
	svp : DataFrame
		Sound speed profile.
	T0 : float
		Typical travel time.

	Returns
	-------
	jcbpos : lil_matrix
		Jacobian matrix for positions.
	"""

    deltap = float(icfg.get("Inv-parameter","deltap"))
    ndata = shotdat.index.size
    MTs = list(mtidx.keys())
    nMT = len(MTs)
    nmppos = len(slvidx0)

    # Initialize a DENSE array for blazing fast row assignment
    jcbpos = np.zeros((nmppos, ndata))
    imp = 0

    gamma = shotdat.gamma.values
    logTTc = shotdat.logTTc.values
    
    # Pre-extract MT values for fast masking
    mt_vals = shotdat['MT'].values

    # Calc Jacobian for Position
    for j in range(3):
        mpj = mp.copy()
        mpj[nMT*3 + j] += deltap
        cTTj, _ = calc_traveltime(shotdat, mpj, nMT, icfg, svp)
        logTTcj = np.log(cTTj/T0) - gamma
        shotdat[f'jacob{j}'] = (logTTcj - logTTc) / deltap

    # Jacobian for each MT
    for mt in MTs:
        mt_mask = (mt_vals == mt)
        for j in range(3):
            if (mtidx[mt] + j) in slvidx0:
                # Direct numpy array multiplication, no Pandas series math
                jcbpos[imp, :] = shotdat[f'jacob{j}'].values * mt_mask
                imp += 1

    # Jacobian for Center Pos
    for j in range(3):
        if (nMT*3 + j) in slvidx0:
            jcbpos[imp, :] = shotdat[f'jacob{j}'].values
            imp += 1

    # Calc Jacobian for ATD offset
    # Cache original values to prevent massive DataFrame copying
    orig_ple0, orig_pln0, orig_plu0 = shotdat['ple0'].values.copy(), shotdat['pln0'].values.copy(), shotdat['plu0'].values.copy()
    orig_ple1, orig_pln1, orig_plu1 = shotdat['ple1'].values.copy(), shotdat['pln1'].values.copy(), shotdat['plu1'].values.copy()

    for j in range(3): # j = 0:rightward, 1:forward, 2:upward
        idx = nMT*3 + 3 + j
        if not (idx in slvidx0):
            continue
            
        mpj = mp.copy()
        mpj[(nMT+1)*3 + j] += deltap
        pl = np.array([mpj[(nMT+1)*3 + 0], mpj[(nMT+1)*3 + 1], mpj[(nMT+1)*3 + 2]])

        # Vectorized ATD offset applied IN PLACE
        ple0, pln0, plu0 = calc_atd_vectorized(pl, shotdat.head0.values, shotdat.roll0.values, shotdat.pitch0.values)
        ple1, pln1, plu1 = calc_atd_vectorized(pl, shotdat.head1.values, shotdat.roll1.values, shotdat.pitch1.values)
        
        shotdat['ple0'], shotdat['pln0'], shotdat['plu0'] = ple0, pln0, plu0
        shotdat['ple1'], shotdat['pln1'], shotdat['plu1'] = ple1, pln1, plu1

        cTTj, _ = calc_traveltime(shotdat, mpj, nMT, icfg, svp)
        logTTcj = np.log(cTTj/T0) - gamma
        jcbpos[imp, :] = (logTTcj - logTTc) / deltap
        imp += 1

    # Restore original DataFrame state
    shotdat['ple0'], shotdat['pln0'], shotdat['plu0'] = orig_ple0, orig_pln0, orig_plu0
    shotdat['ple1'], shotdat['pln1'], shotdat['plu1'] = orig_ple1, orig_pln1, orig_plu1

    # Return as highly optimized CSR matrix for the linear algebra solver
    return csr_matrix(jcbpos)