'''
Created:
	07/01/2020 by S. Watanabe
Modified:
	02/01/2021 by S. Watanabe
		to fix unknown MT number flag "M00" for mis-response seen in TU's sites.
	01/07/2022 by S. Watanabe
		to set B-spline's knots by time interval (also need to change "Setup.ini" file)
		to use cholesky decomposition for calc. inverse
	03/30/2022 by S. Watanabe and Y. Nakamura
		to adjust the threshold for rank calculation
	07/01/2024 by S. Watanabe 
		to apply a mode for "array take-over"
		(to solve each position and parallel disp. simultaneously)
		"invtyp" is deleted. users can set zero in config files 
		to solve limited parameter(s).
    2026-03-05 by Hutchinson
        Made optimizations to speed up inversions and save RAM.
        Added a tolerance factor to the priori covariance matrix to prevent crashing with
        smaller knotints and smaller SETs.
'''

import os
import sys
import math
import configparser
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix, linalg, identity, diags
from sksparse.cholmod import cholesky
import pandas as pd

# garpos module
from .setup_model import init_position, make_knots, derivative2, data_correlation
from .forward import calc_forward, calc_gamma, jacobian_pos, jacobian_gamma
from .output import outresults


def MPestimate(cfgf, icfgf, odir, suf, lamb0, lgrad, mu_t, mu_m):
    spdeg = 3
    np.set_printoptions(threshold=np.inf)

    if lamb0 <= 0.:
        print("Lambda must be > 0")
        sys.exit(1)

    ################################
    ### Set Inversion Parameters ###
    ################################
    icfg = configparser.ConfigParser()
    icfg.read(icfgf, 'UTF-8')
    knotint0 = float(icfg.get("Inv-parameter","knotint0"))*60.
    knotint1 = float(icfg.get("Inv-parameter","knotint1"))*60.
    knotint2 = float(icfg.get("Inv-parameter","knotint2"))*60.
    rsig = float(icfg.get("Inv-parameter","RejectCriteria"))
    scale = float(icfg.get("Inv-parameter","traveltimescale"))
    maxloop = int(icfg.get("Inv-parameter","maxloop"))
    ConvCriteria = float(icfg.get("Inv-parameter","ConvCriteria"))

    #############################
    ### Set Config Parameters ###
    #############################
    cfg = configparser.ConfigParser()
    cfg.read(cfgf, 'UTF-8')

    ### Read obs file ###
    obsfile  = cfg.get("Data-file", "datacsv")
    shots = pd.read_csv(obsfile, comment='#', index_col=0)

    # check NaN in shotdata and TT > 0
    shots = shots.dropna().reset_index(drop=True)
    shots = shots[shots.TT > 0.].reset_index(drop=True)

    ### Sound speed profile ###
    svpf = cfg.get("Obs-parameter", "SoundSpeed")
    svp = pd.read_csv(svpf, comment='#')
    site = cfg.get("Obs-parameter", "Site_name")

    ### IDs of existing transponder ###
    MTs = cfg.get("Site-parameter", "Stations").split()
    MTs = [ str(mt) for mt in MTs ]
    nMT = len(MTs)
    M00 = shots[ shots.MT == "M00" ].reset_index(drop=True)
    shots = shots[ shots.MT.isin(MTs) ].reset_index(drop=True)

    # for mis-response in MT number verification
    shots["m0flag"] = False
    M00["m0flag"] = True
    M00["flag"] = True
    
    if not M00.empty:
        # Fast way to duplicate M00 for all MTs
        addshots = pd.concat([M00.assign(MT=mt) for mt in MTs], ignore_index=True)
        shots = pd.concat([shots, addshots], ignore_index=True)
        
    chkMT = rsig > 0.1 and len(M00) >= 1

    ############################
    ### Set Model Parameters ###
    ############################
    mppos, Dipos, slvidx0, mtidx = init_position(cfg, MTs)

    if len(slvidx0) == 0:
        Dipos = lil_matrix((0, 0))
        slvidx0 = np.array([])

    nmppos = len(slvidx0)
    cnt = np.array([mppos[imt*3:imt*3+3] for imt in range(nMT)]).mean(axis=0)
    
    shots['mtid'] = shots['MT'].map(mtidx)

    ### Set Model Parameters for gamma ###
    knotintervals = [knotint0, knotint1, knotint1, knotint2, knotint2]
    lambdas = [lamb0] + [lamb0 * lgrad]*4

    knots = make_knots(shots, spdeg, knotintervals)
    ncps = [max([0, len(kn)-spdeg-1]) for kn in knots]

    imp0 = np.cumsum(np.array([len(mppos)] + ncps))

    mp = np.zeros(imp0[-1])
    mp[:imp0[0]] = mppos

    slvidx = np.append(slvidx0, np.arange(imp0[0], imp0[-1], dtype=int))
    H = derivative2(imp0, spdeg, knots, lambdas)

    mp0 = mp[slvidx]
    mp1 = mp0.copy()
    nmp = len(mp0)

    ### Set a priori covariance for model parameters ###
    Di = lil_matrix((nmp, nmp))
    Di[:nmppos, :nmppos] = Dipos
    Di[nmppos:, nmppos:] = H
    Di = Di.tocsc()

    # Define a single, consistent tolerance based on lamb0
    eig_tol = 1.e-8 / lamb0

    # Extract the dense array exactly once to save RAM
    Di_dense = Di.toarray()

    # Calculate rank and eigenvalues using the EXACT same tolerance
    rankDi = np.linalg.matrix_rank(Di_dense, tol=eig_tol)
    
    eigvDi = np.linalg.eigh(Di_dense)[0]
    eigvDi = eigvDi[np.abs(eigvDi.real) > eig_tol].real

    if rankDi != len(eigvDi):
        print(f"Matrix Rank: {rankDi}, Valid Eigenvalues: {len(eigvDi)}")
        print("Error in calculating eigen value of Di !!!")
        sys.exit(1)

    logdetDi = np.log(eigvDi).sum()

    # Initial parameters for gradient gamma
    shots['sta0_e'] = mp[shots['mtid']+0] + mp[len(MTs)*3+0]
    shots['sta0_n'] = mp[shots['mtid']+1] + mp[len(MTs)*3+1]
    shots['sta0_u'] = mp[shots['mtid']+2] + mp[len(MTs)*3+2]
    shots['mtde'] = (shots['sta0_e'].values - cnt[0])
    shots['mtdn'] = (shots['sta0_n'].values - cnt[1])
    shots['de0'] = shots['ant_e0'].values - shots['ant_e0'].values.mean()
    shots['dn0'] = shots['ant_n0'].values - shots['ant_n0'].values.mean()
    shots['de1'] = shots['ant_e1'].values - shots['ant_e1'].values.mean()
    shots['dn1'] = shots['ant_n1'].values - shots['ant_n1'].values.mean()
    shots['iniflag'] = shots['flag'].copy()

    #####################################
    # Set log(TT/T0) and initial values #
    #####################################
    L0 = np.array([(mp[i*3+2] + mp[nMT*3+2]) for i in range(nMT)]).mean()
    L0 = abs(L0 * 2.)

    vl = svp.speed.values
    dl = svp.depth.values
    avevlyr = [(vl[i+1]+vl[i])*(dl[i+1]-dl[i])/2. for i in range(len(svp)-1)]
    V0 = np.sum(avevlyr)/(dl[-1]-dl[0])

    T0 = L0 / V0
    shots["logTT"] = np.log(shots.TT.values/T0)

    ######################
    ## data correlation ##
    ######################
    shots["gamma"] = 0.
    shots = calc_forward(shots, mp, nMT, icfg, svp, T0)

    icorrE = rsig < 0.1 and mu_t > 1.e-3
    if not icorrE:
        mu_t = 0.
        
    tmp = shots[~shots['flag']].reset_index(drop=True)
    ndata = len(tmp)
    scale = scale / T0
    TT0 = tmp.TT.values / T0

    if icorrE:
        E_factor = data_correlation(tmp, TT0, mu_t, mu_m)
        logdetEi = -E_factor.logdet()
    else:
        # OPTIMIZATION: Avoid Dense 20GB matrix crash
        Ei = diags(TT0**2., format='csc') / scale**2.
        logdetEi = (np.log(TT0**2.)).sum()

    #############################
    ### loop for Least Square ###
    #############################
    comment = ""
    iconv = 0
    for iloop in range(maxloop):

        ############################
        ### Calc Jacobian matrix ###
        ############################

        if rsig > 0.1 or iloop == 0:
            # OPTIMIZATION: Dense assignment is faster than lil_matrix
            jcb_dense = np.zeros((nmp, ndata))

        if rsig > 0.1 or iloop == 0:
            # Get the entire gamma jacobian block in one vectorized call!
            jcb_gamma_block = jacobian_gamma(tmp, imp0, spdeg, knots)
            
            # Dump the block directly into the Jacobian and apply the negative scale
            jcb_dense[nmppos:, :] = -jcb_gamma_block * scale

        if len(slvidx0) != 0:
            jcb0 = jacobian_pos(icfg, mp, slvidx0, tmp, mtidx, svp, T0)
            jcb_dense[:nmppos, :] = jcb0[:nmppos, :].toarray()

        jcb = csc_matrix(jcb_dense)
        
        ############################
        ### CALC model parameter ###
        ############################
        alpha = 1.0
        if icorrE:
            LiAk = E_factor.solve_L(jcb.T.tocsc(), use_LDLt_decomposition=False)
            AktEiAk = LiAk.T @ LiAk / scale**2.
            rk = jcb @ E_factor(tmp.ResiTT.values) / scale**2. + Di @ (mp0-mp1)
        else:
            AktEi = jcb @ Ei
            AktEiAk = AktEi @ jcb.T
            rk  = AktEi @ tmp.ResiTT.values + Di @ (mp0-mp1)

        Cki = AktEiAk + Di
        Cki_factor = cholesky(Cki.tocsc(), ordering_method="natural")
        Ckrk = Cki_factor(rk)
        dmp  = alpha * Ckrk
        
        dxmax = max(abs(dmp[:]))
        if len(slvidx0) == 0 and rsig <= 0.1:
            dposmax = 0.
        elif len(slvidx0) == 0 and rsig > 0.1:
            dposmax = ConvCriteria/200.
        else:
            dposmax = max(abs(dmp[:nmppos]))
            if dxmax > 10.:
                alpha = 10./dxmax
                dmp = alpha * dmp
                dxmax = max(abs(dmp[:]))

        mp1 += dmp
        for j in range(len(mp1)):
            mp[slvidx[j]] = mp1[j]

        ####################
        ### CALC Forward ###
        ####################
        gamma, a  = calc_gamma(mp, shots, imp0, spdeg, knots)
        shots["gamma"] = gamma * scale
        av = np.array(a) * scale * V0
        shots['dV'] = shots.gamma * V0

        shots = calc_forward(shots, mp, nMT, icfg, svp, T0)

        if chkMT and iconv >= 1:
            print("Check MT number for shots named 'M00'")
            comment += "Check MT number for shots named 'M00'\n"
            rsigm0 = 1.0
            
            # Optimized boolean masking
            valid_mask = ~shots['flag']
            aveRTT = shots.loc[valid_mask, 'ResiTT'].mean()
            sigRTT = shots.loc[valid_mask, 'ResiTT'].std()
            
            th0 = aveRTT + rsigm0 * sigRTT
            th1 = aveRTT - rsigm0 * sigRTT
            shots.loc[shots.m0flag, 'flag'] = (shots['ResiTT'] > th0) | (shots['ResiTT'] < th1)

        # Update tmp safely for the next iteration
        tmp = shots[~shots['flag']].reset_index(drop=True)
        ndata = len(tmp)
        TT0 = tmp.TT.values / T0

        if rsig > 0.1:
            if icorrE:
                E_factor = data_correlation(tmp, TT0, mu_t, mu_m)
                logdetEi = -E_factor.logdet()
            else:
                Ei = diags(TT0**2., format='csc') / scale**2.
                logdetEi = (np.log(TT0**2.)).sum()

        rttadp = tmp.ResiTT.values

        if icorrE:
            misfit = rttadp @ E_factor(rttadp) / scale**2.
        else:
            rttvec = csr_matrix(np.array([rttadp]))
            misfit = ((rttvec @ Ei) @ rttvec.T)[0,0]

        # Calc Model-parameters' RMSs
        rms = lambda d: np.sqrt((d ** 2.).sum() / d.size)
        datarms = rms(tmp.ResiTTreal.values)

        aved = np.array([(mp[i*3+2] + mp[nMT*3+2]) for i in range(nMT)]).mean()
        reject = shots['flag'].sum()
        ratio  = 100. - float(reject)/float(len(shots))*100.

        ##################
        ### Check Conv ###
        ##################
        loopres  = "Inversion loop %03d, " % (iloop+1)
        loopres += "RMS(TT) = %10.6f ms, " % (datarms*1000.)
        loopres += "used_shot = %5.1f%%, reject = %4d, " % (ratio, reject)
        loopres += "Max(dX) = %10.4f, Hgt = %10.3f" % (dxmax, aved)
        print(loopres)
        comment += "#"+loopres+"\n"

        if (dxmax < ConvCriteria/100. or dposmax < ConvCriteria/1000.) and not chkMT:
            break
        elif dxmax < ConvCriteria:
            iconv += 1
            if iconv == 2:
                break
        else:
            iconv = 0

    #######################
    # calc ABIC and sigma #
    #######################
    dof = float(ndata + rankDi - nmp)
    S = misfit + ((mp0-mp1) @ Di) @ (mp0-mp1)

    logdetCki = Cki_factor.logdet()

    abic = dof * math.log(S) - logdetEi - logdetDi + logdetCki
    sigobs  = (S/dof)**0.5 * scale

    Ck = Cki_factor(identity(nmp).tocsc())
    C = S/dof * Ck.toarray()
    rmsmisfit = (misfit/ndata) ** 0.5 * sigobs

    finalres  = " ABIC = %18.6f " % abic
    finalres += " misfit = % 6.3f " % (rmsmisfit*1000.)
    finalres += suf
    print(finalres)
    comment += "# " + finalres + "\n"

    comment += "# lambda_0^2 = %12.8f\n" % lamb0
    comment += "# lambda_g^2 = %12.8f\n" % (lamb0 * lgrad)
    comment += "# mu_t = %12.8f sec.\n" % mu_t
    comment += "# mu_MT = %5.4f\n" % mu_m

    #####################
    # Write Result data #
    #####################

    resf, dcpos = outresults(odir, suf, cfg, imp0, slvidx0,
                             C, mp, shots, comment, MTs, mtidx, av)

    return [resf, datarms, abic, dcpos]