'''
This script contains functions used for integrating the equations of motions.
I did want to try leapfrog, but I don't seem to have enough time.
'''
from SPH_functions import *

def Euler_gas(ts, dt, BC, 
              ms, xs0, vs0, hs0, rhos0, fs0, Ps0, alpha, beta, epsilon, gamma = 1.4, 
              selfgravity = False, eps = 1., dphidr = None, G = 6.67e-11, 
              adjustSmoothLen = False):
    '''
    Initial values are given by xs0, vs0, hs0, rhos0, fs0, Ps0.
    Use the Euler's method to integrate the equations of motion (dv/dt and dA/dt) to 
      the times spicified by ts.
    Get a snapshot at each time in ts.
    BC should be a function specifying the boundary conditions. The correct format is
      like xs_new, vs_new = BC(xs0, vs0, xs_new, vs_new).
    If adjustSmoothLen, then use variable smoothing lengths. Otherwise fix them.
    '''
    # ms = np.asarray(ms, dtype=float)
    # N = ms.size
    # xs0 = np.asarray(xs0, dtype=float).reshape(N,-1)
    # vs0 = np.asarray(vs0, dtype=float).reshape(N,-1)
    # hs0 = np.asarray(hs0, dtype=float)
    # rhos0 = np.asarray(rhos0, dtype=float)
    # fs0 = np.asarray(fs0, dtype=float)
    # Ps0 = np.asarray(Ps0, dtype=float)
    As0 = Ps0/rhos0**gamma
    D = xs0.shape[1]
    Cs = rhos0*hs0**D
    
    ts = np.append(0,np.asarray(ts))
    ts = ts[1:] - ts[0:-1]
    index = np.linspace(0,len(ms)-1,len(ms), dtype=int)
    rst_all = []
    for t in ts:
        t_remain = t
        if t_remain < dt:
            dt = t_remain
        while t_remain > 1e-6:
            dvdts0, dAdts0 = EoM_gas(ms, xs0, vs0, hs0, rhos0, fs0, Ps0, 
                                     alpha, beta, epsilon, gamma = gamma, 
                                     selfgravity = selfgravity, eps = eps, 
                                     dphidr = dphidr, G = G)
            # simple Forward Euler's method
            vs_new = vs0 + dvdts0*dt
            xs_new = xs0 + vs0*dt
            As_new = As0 + dAdts0*dt
            
            # implement boundary condition
            xs_new, vs_new = BC(xs0, vs0, xs_new, vs_new)

            # change smoothing lengths?
            if adjustSmoothLen:
                hs_new = adjustSmLen(ms, xs_new, rhos0, hs0, Cs)[1]
                ### there could still be problems with the adjustSmLen fucntion ###
                # ii = np.isinf(hs_new)
                # hs_new[ii] = np.mean(hs_new[~ii])
            else:
                hs_new = hs0
    
            rhos_new, tmp, fs_new = calcRho(ms, xs_new, hs_new, index, 
                                            adjustSmoothLen = adjustSmoothLen)
            Ps_new = As_new*rhos_new**gamma

            xs0 = xs_new+0.
            vs0 = vs_new+0.
            As0 = As_new+0.
            hs0 = hs_new+0.
            rhos0 = rhos_new+0.
            fs0 = fs_new+0.
            Ps0 = Ps_new+0.

            t_remain -= dt
            if (t_remain > 1e-6) & (t_remain < dt):
                dt = t_remain
    
        rst_all.append([xs0, vs0, As0, hs0, rhos0, fs0, Ps0])
    
    return rst_all

def Euler_g(ts, dt, BC, 
            ms, xs0, vs0, h, eps, dphidr, G):
    '''
    Only involve self-gravity here. eps is the softening length.
    The format of dphidr should be g = m * dphidr(r, h, eps).
    We do not adjust smoothing lengths in this case, just to make things easier.
    '''
    ts = np.append(0,np.asarray(ts))
    ts = ts[1:] - ts[0:-1]

    rst_all = []
    for t in ts:
        t_remain = t
        if t_remain < dt:
            dt = t_remain
        while t_remain > 1e-6:
            dvdts0 = dvdt_g(ms, xs0, h, eps, dphidr, G)
            # simple Forward Euler's method
            vs_new = vs0 + dvdts0*dt
            xs_new = xs0 + vs0*dt
            
            # implement boundary condition
            xs_new, vs_new = BC(xs0, vs0, xs_new, vs_new)

            xs0 = xs_new+0.
            vs0 = vs_new+0.

            t_remain -= dt
            if (t_remain > 1e-6) & (t_remain < dt):
                dt = t_remain
    
        rst_all.append([xs0, vs0])
    
    return rst_all















