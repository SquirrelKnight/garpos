"""
Created:
    07/01/2020 by S. Watanabe
Modified: 
    2026-03-05 by Hutchinson
        Replaced legacy Fortran lib_raytrace backend with pure 
        Python Numba JIT compilation. Optimized with numba multithreading (prange) and 
        vectorized array feeding.
"""
import numpy as np
from numba import njit, prange

# =====================================================================
# --- Core Numba raytracing functions ---
# =====================================================================

@njit
def layer_setting(nlyr, depth, l_depth, l_sv, l_sv_trend):
    layer_n = nlyr - 1 # Default to the last layer
    for i in range(1, nlyr):
        if l_depth[i] >= depth:
            layer_n = i
            break
            
    sv = l_sv[layer_n] + (depth - l_depth[layer_n]) * l_sv_trend[layer_n]
    return sv, layer_n

@njit
def ray_path(t_angle, nlyr, layer_d, layer_s, sv_d, sv_s, y_d, y_s, l_dep, l_sv):
    x = 0.0
    ray_length = 0.0
    pp = np.sin(t_angle) / sv_d

    if y_d == y_s:
        return -1.0, ray_length
    
    k1 = layer_s - 1
    lm = layer_d

    sn = np.zeros(nlyr)
    yds = np.zeros(nlyr)
    rn = np.zeros(nlyr)
    scn = np.zeros(nlyr)

    sn[k1:layer_d+1] = pp * l_sv[k1:layer_d+1]
    yds[k1:layer_d+1] = l_dep[k1:layer_d+1]

    yds[k1] = y_s
    sn[k1] = pp * sv_s
    yds[layer_d] = y_d
    sn[layer_d] = pp * sv_d

    # Find max and min to check for critical refraction
    max_sn = -2.0
    min_sn = 2.0
    for i in range(k1, layer_d+1):
        if sn[i] > max_sn: max_sn = sn[i]
        if sn[i] < min_sn: min_sn = sn[i]

    if max_sn > 1.0 or min_sn < -1.0:
        return -1.0, ray_length
    
    scn[k1:layer_d+1] = np.sqrt(1.0 - sn[k1:layer_d+1]**2.0)
    rn[k1:layer_d] = scn[lm] / scn[k1:layer_d]
    rn[layer_d] = 1.0

    for i in range(layer_s, layer_d+1):
        j = i - 1
        dx = (yds[i] - yds[j]) * (sn[i] + sn[j]) / (scn[i] + scn[j])
        x = x + dx
        ray_length = ray_length + rn[i] * dx / scn[j]

    ray_length = ray_length / sn[lm]
    return x, ray_length

@njit
def calc_travel_time_numba(t_angle, nl, layer_d, layer_s, sv_d, sv_s, y_d, y_s, l_dep, l_sv, l_sv_trend, l_th):
    lmax = 55
    epg = 1.0e-2
    eang = np.deg2rad(80)
    epr = 1.0e-12

    epa = np.sin(eang)
    travel_time = 0.0
    pp = np.sin(t_angle) / sv_d
    tinc = 0.0

    if y_d != y_s:
        k1 = layer_s - 1
        
        vn = np.zeros(layer_d+1)
        tn = np.zeros(layer_d+1)

        vn[k1+1:layer_d] = l_sv[k1+1:layer_d]
        vn[k1] = sv_s
        vn[layer_d] = sv_d

        tn[layer_s:layer_d+1] = l_th[layer_s:layer_d+1]
        tn[layer_s] = tn[layer_s] - (y_s - l_dep[k1])
        tn[layer_d] = tn[layer_d] - (l_dep[layer_d] - y_d)

        for i in range(layer_s, layer_d+1):
            j = i - 1
            sn1 = pp * vn[i]
            cn1 = np.sqrt(1.0 - sn1**2.0)
            sn2 = pp * vn[j]
            cn2 = np.sqrt(1.0 - sn2**2.0)

            if abs(l_sv_trend[i]) > epg:
                tinc = (np.log((1.0 + cn2) / (1.0 + cn1)) + np.log(vn[i] / vn[j])) / l_sv_trend[i]
            elif abs(l_sv_trend[i]) <= epg:
                snm = min(sn1, sn2)

                if snm > epa:
                    aatra = 1.0
                    bbtra = 1.0
                    cctra = cn1 * (cn2 + cn1)
                    d2 = cn2**2.0
                    d1 = cn1**2.0
                    tmp = 0.0
                    ls = 1

                    while True:
                        tmpdd = aatra / float(ls)
                        tmp += tmpdd
                        if tmpdd >= epr and ls <= lmax:
                            aatra = aatra * d2 + bbtra * cctra
                            bbtra = bbtra * d1
                            ls += 2
                        else:
                            tinc = tn[i] * tmp * pp * (sn1 + sn2) / (cn1 + cn2)
                            break

                elif snm <= epa:
                    zzc = pp * (sn1 + sn2) / (1.0 + cn1) / (cn1 + cn2)
                    xxc = 1.0 / vn[i]
                    zz = (cn1 - cn2) / (1.0 + cn1)
                    xx = xxc * (vn[i] - vn[j])
                    za = 1.0
                    xa = 1.0
                    tmp = 0.0
                    ls = 1

                    while True:
                        tmpdd = ((za * zzc) + (xa * xxc)) / float(ls)
                        tmp += tmpdd
                        if tmpdd >= epr and ls <= lmax:
                            za = za * zz
                            xa = xa * xx
                            ls += 1
                        else:
                            tinc = tn[i] * tmp
                            break
            
            travel_time = travel_time + tinc
    return travel_time

@njit
def calc_ray_path(distance, y_d, y_s, l_depth, l_sv, nlyr, layer_thickness, layer_sv_trend):
    """Core ray physics for a single point"""
    loop1 = 200
    loop2 = 20
    eps1 = 1e-7
    eps2 = 1e-14
    pi = np.pi

    t_ag, t_tm = 0.0, 0.0

    sv_d, layer_d = layer_setting(nlyr, y_d, l_depth[0:nlyr], l_sv[0:nlyr], layer_sv_trend[0:nlyr])
    sv_s, layer_s = layer_setting(nlyr, y_s, l_depth[0:nlyr], l_sv[0:nlyr], layer_sv_trend[0:nlyr])

    x_hori = np.zeros(6)
    r_nm = 0
    tadeg = np.array([0.0, 20.0, 40.0, 60.0, 70.0, 80.0])
    ta_rough = pi * (180.0 - tadeg) / 180.0

    for i in range(0, 5):
        j = i + 1
        x_hori[j], a0 = ray_path(ta_rough[j], nlyr, layer_d, layer_s, sv_d, sv_s, y_d, y_s, l_depth[0:nlyr], l_sv[0:nlyr])

        if x_hori[j] < 0: continue

        diff1 = distance - x_hori[i]
        diff2 = x_hori[j] - distance

        if diff1 * diff2 < 0: continue

        r_nm = 1
        x1, x2 = x_hori[i], x_hori[j]
        t_angle1, t_angle2 = ta_rough[i], ta_rough[j]

        # First refinement loop
        for k in range(loop1):
            x_diff = x1 - x2
            if abs(x_diff) < eps1: break
            t_angle0 = (t_angle1 + t_angle2) / 2.0
            x0, a0 = ray_path(t_angle0, nlyr, layer_d, layer_s, sv_d, sv_s, y_d, y_s, l_depth[0:nlyr], l_sv[0:nlyr])

            a0 = -a0
            t0 = distance - x0
            
            if t0 * diff1 > 0:
                x1, t_angle1 = x0, t_angle0
            else:
                x2, t_angle2 = x0, t_angle0
                
        diff_true0 = abs((distance - x0) / distance)

        # Second refinement loop
        for k in range(loop2):
            delta_angle = (distance - x0) / a0
            t_angle = t_angle0 + delta_angle

            if abs(delta_angle) < eps2: break

            x0, a0 = ray_path(t_angle, nlyr, layer_d, layer_s, sv_d, sv_s, y_d, y_s, l_depth[0:nlyr], l_sv[0:nlyr])
            a0 = -a0

            diff_true1 = abs((distance - x0) / distance)
            if diff_true0 <= diff_true1:
                t_angle = t_angle0
                break
            if diff_true1 < eps2: break
        
        # Break outer loop once a valid bracket is fully refined
        break
        
    if r_nm == 0:
        return 0.0, 0.0
    else:
        t_ag = t_angle
        t_tm = calc_travel_time_numba(t_angle, nlyr, layer_d, layer_s, sv_d, sv_s, y_d, y_s,
                                      l_depth[0:nlyr], l_sv[0:nlyr], layer_sv_trend[0:nlyr], layer_thickness[0:nlyr])
        return t_ag, t_tm

@njit(parallel=True)
def vectorize_ray_paths(dist0, dist1, yd, ys0, ys1, l_depth, l_sv):
    """
    Loops over all distances.
    parallel=True automatically distributes this across your CPU cores.
    """
    nlyr = len(l_depth)
    n_shots = len(dist0)
    
    # Pre-calculate SVP layers once for the entire array
    layer_thickness = np.zeros(nlyr)
    layer_sv_trend = np.zeros(nlyr)
    layer_thickness[1:nlyr] = l_depth[1:nlyr] - l_depth[0:nlyr-1]
    layer_sv_trend[1:nlyr] = (l_sv[1:nlyr] - l_sv[0:nlyr-1]) / layer_thickness[1:nlyr]

    # Pre-allocate output arrays
    toas1 = np.zeros(n_shots)
    tts1 = np.zeros(n_shots)
    toas2 = np.zeros(n_shots)
    tts2 = np.zeros(n_shots)

    # prange enables multi-core C-speed threading!
    for i in prange(n_shots):
        toas1[i], tts1[i] = calc_ray_path(dist0[i], -yd[i], -ys0[i], l_depth, l_sv, nlyr, layer_thickness, layer_sv_trend)
        toas2[i], tts2[i] = calc_ray_path(dist1[i], -yd[i], -ys1[i], l_depth, l_sv, nlyr, layer_thickness, layer_sv_trend)
        
    return toas1, tts1, toas2, tts2


# =====================================================================
# --- GARPOS traveltime computation function ---
# =====================================================================

def calc_traveltime(shotdat, mp, nMT, icfg, svp):
    """
    Calculate the round-trip travel time.
    Replaces the legacy Fortran wrapper.
    
    Parameters
    ----------
    shotdat : DataFrame
        GNSS-A shot dataset.
    mp : ndarray
        complete model parameter vector.
    nMT : int
        number of transponders.
    icfg : configparser
        Config file for inversion conditions (kept for backward compatibility).
    svp : DataFrame
        Sound speed profile.
    
    Returns
    -------
    calTT : ndarray
        Calculated travel time (sec.).
    calA0 : ndarray
        Calculated take-off angle (degree).
    """
    # Station pos
    sta0_e = mp[shotdat['mtid']+0] + mp[nMT*3+0]
    sta0_n = mp[shotdat['mtid']+1] + mp[nMT*3+1]
    sta0_u = mp[shotdat['mtid']+2] + mp[nMT*3+2]

    # Antenna + offset positions
    e0 = shotdat.ant_e0.values + shotdat.ple0.values
    n0 = shotdat.ant_n0.values + shotdat.pln0.values
    u0 = shotdat.ant_u0.values + shotdat.plu0.values
    e1 = shotdat.ant_e1.values + shotdat.ple1.values
    n1 = shotdat.ant_n1.values + shotdat.pln1.values
    u1 = shotdat.ant_u1.values + shotdat.plu1.values

    # Blazing-fast hypotenuse math
    dist0 = np.hypot(e0 - sta0_e, n0 - sta0_n)
    dist1 = np.hypot(e1 - sta0_e, n1 - sta0_n)

    # Cast arrays to strictly typed Float64 to prevent Numba type errors
    l_depth = svp.depth.values.astype(np.float64)
    l_sv = svp.speed.values.astype(np.float64)

    yd = np.zeros(len(shotdat), dtype=np.float64) + sta0_u
    ys0 = u0.astype(np.float64)
    ys1 = u1.astype(np.float64)

    # Safety checks prior to physics execution
    if np.isnan(yd).any() or np.isnan(ys0).any() or np.isnan(ys1).any():
        print("Fatal Error: NaN detected in vertical positions (yd or ys).")
        import sys
        sys.exit(1)
        
    if np.min(yd) < -l_depth[-1]:
        print("Error: yd is deeper than layer")
        import sys
        sys.exit(1)
        
    if np.max(ys0) > -l_depth[0] or np.max(ys1) > -l_depth[0]:
        # Dynamically append a shallower layer if the array surfaces
        l_depth = np.append(np.array([-40.0], dtype=np.float64), l_depth)
        l_sv = np.append(np.array([l_sv[0]], dtype=np.float64), l_sv)
        
        if np.max(ys0) > -l_depth[0] or np.max(ys1) > -l_depth[0]:
            print("Error: ys is shallower than layer")
            import sys
            sys.exit(1)

    # Execute ray path code
    toas1, tts1, toas2, tts2 = vectorize_ray_paths(dist0, dist1, yd, ys0, ys1, l_depth, l_sv)

    # Calculate total traveltime
    calTT = tts1 + tts2
    calA0 = 180.0 - np.rad2deg((toas1 + toas2) / 2.0)

    return calTT, calA0