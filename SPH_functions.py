'''
This file contains functions that will be used in SPH simulations.
The first version of these functions contain a lot of things that are convenient 
  for debugging. Once debugging is finished I do not need them anymore, so they 
  are commented out, although they are useful in the most general cases.
'''
import numpy as np
import scipy
import scipy.integrate
from math import *
import matplotlib.pyplot as plt
import pickle
import time

def calcW(rs, hs, dimension):
    '''
    W is the SPH kernel using cubic spline.
    rs and hs should both be vectors.
    Returns W, \partial W / \partial r, \partial W / \partial h.
    '''
    rs = np.asarray(rs, dtype=float)
    hs = np.asarray(hs, dtype=float)
    
    if dimension == 1:
        alphas = 1/hs
        dadhs = -1/hs**2
    elif dimension == 2:
        alphas = 15/(7*pi*hs**2)
        dadhs = -30/(7*pi*hs**3)
    else:
        alphas = 3/(2*pi*hs**3)
        dadhs = -9/(2*pi*hs**4)
    
    Rs = (rs/hs).reshape(-1)

    if Rs.size == 1:
        if Rs < 1:
            tmp = 2./3 - Rs**2 + 0.5*Rs**3
            dtdRs = -2*Rs + 1.5*Rs**2
            return alphas*tmp, \
                   alphas*dtdRs/hs, \
                   -rs/hs**2*alphas*dtdRs + dadhs*tmp
        elif Rs < 2:
            tmp = 1./6*(2 - Rs)**3
            dtdRs = -1./2*(2 - Rs)**2
            return alphas*tmp, \
                   alphas*dtdRs/hs, \
                   -rs/hs**2*alphas*dtdRs + dadhs*tmp
        else:
            return np.array([0.]), np.array([0.]), np.array([0.])

    ii01 = Rs < 1.
    ii12 = (Rs >= 1.) & (Rs < 2.)
    tmp = np.zeros_like(Rs)
    dtdRs = tmp.copy()
    tmp[ii01] = 2./3 - Rs[ii01]**2 + 0.5*Rs[ii01]**3
    tmp[ii12] = 1./6*(2 - Rs[ii12])**3
    dtdRs[ii01] = -2*Rs[ii01] + 1.5*Rs[ii01]**2
    dtdRs[ii12] = -1./2*(2 - Rs[ii12])**2

    return tmp*alphas, dtdRs*alphas/hs, -rs/hs**2*alphas*dtdRs + dadhs*tmp

def calcRho(ms, xs, hs, index, adjustSmoothLen = False):
    '''
    Calculate the SPH estimate of the density for each particle specified by the 
      array 'index', which gives the indexes of the particles you would like to 
      calculate the densities.
    ms, xs, hs are the mass, position, and smoothing length vectors.
    xs.shape[1] indicates the dimension of the problem.
    Returns rho and f = 1./(1 + hs/3./rhos* \partial rho / \partial h).
    '''
    # ms = np.asarray(ms, dtype=float)
    # xs = np.asarray(xs, dtype=float).reshape(len(ms),-1)
    # hs = np.asarray(hs, dtype=float)
    # index = np.asarray(index, dtype=int).reshape(-1)
   
    N = ms.size # number of particles
    D = xs.shape[1] # dimension of the problem

    rhos = np.zeros(len(index))
    drhodhs = rhos.copy()
    fs = np.ones_like(rhos)
    for count,i in enumerate(index):
        dist = np.sum((xs[i,:] - xs)**2, axis = 1)**0.5
        ind = np.where(dist < 2*hs[i])[0]
        ws, dwdrs, dwdhs = calcW(dist[ind],hs[i]*np.ones(len(ind)),D)
        rhos[count] = np.sum(ms[ind]*ws)
        if not adjustSmoothLen:
            continue
        drhodhs[count] = np.sum(ms[ind]*dwdhs)
        fs[count] = 1./(1 + hs[i]/3./rhos[count]*drhodhs[count])
        
    return rhos, drhodhs, fs

###########################################################
##### there may still be problems with this function! #####
def adjustSmLen(ms, xs, rhos, hs, Cs, tol = 1e-2, fixRho = False):
    '''
    If fixRho, then we find the proper smoothing lengths and C such that rho*h^D = C,
      and the density computed by calcRho for every particle is rho.
    Otherwise adjust the smoothing lengths and densities iteratively according to 
      calcRho and rho*h^D = C.
    '''
    # ms = np.asarray(ms, dtype=float)
    # xs = np.asarray(xs, dtype=float).reshape(len(ms),-1)
    # rhos0 = np.asarray(rhos, dtype=float)
    # hs0 = np.asarray(hs, dtype=float)
    # Cs0 = np.asarray(Cs, dtype=float)
    rhos0 = rhos.copy()
    hs0 = hs.copy()
    Cs0 = Cs.copy()
    
    rhos_new = rhos0.copy()
    hs_new = hs0.copy()
    Cs_new = Cs0.copy()
    index = np.linspace(0,len(rhos0)-1,len(rhos0), dtype=int)
    D = xs.shape[1]

    max_iter = 800
    k = 0
    
    if fixRho:
        while True:
            rst = calcRho(ms, xs, hs0, index, adjustSmoothLen = True)
            hs_new[index] = hs0[index] - (rst[0] - rhos0[index])/rst[1]
            index = np.where(np.abs((hs_new - hs0)/hs0) > tol)[0]
            if len(index) == 0:
                break
            hs0[index] = hs_new[index]+0.

            k += 1
            if k > max_iter:
                break

        Cs_new = rhos0*hs_new**D
            
    else:
        while True:
            rhos_new[index] = calcRho(ms, xs, hs_new, index, adjustSmoothLen = True)[0]
            hs_new[index] = (Cs0[index]/rhos_new[index])**(1./D)
            index = np.where(np.abs((hs_new - hs0)/hs0) > tol)[0]
            if len(index) == 0:
                break
            rhos0[index] = rhos_new[index]+0.
            hs0[index] = hs_new[index]+0.

            k += 1
            if k > max_iter:
                hs_new[index] = hs[index]
                break

    return rhos_new, hs_new, Cs_new
###########################################################

def EoM_gas(ms, xs, vs, hs, rhos, fs, Ps, alpha, beta, epsilon, gamma = 1.4, 
            selfgravity = False, eps = 1., dphidr = None, G = 6.67e-11):
    '''
    Calculate dv/dt and dA/dt for gas. A is entropy.
    '''
    # ms = np.asarray(ms, dtype=float)
    # xs = np.asarray(xs, dtype=float).reshape(len(ms),-1)
    # vs = np.asarray(vs, dtype=float).reshape(len(ms),-1)
    # hs = np.asarray(hs, dtype=float)
    # rhos = np.asarray(rhos, dtype=float)
    # fs = np.asarray(fs, dtype=float)
    # Ps = np.asarray(Ps, dtype=float)
    
    N = ms.size # number of particles
    # if N == 1:
    #     return 0., 0.
    
    D = xs.shape[1] # dimension of the problem
    tmps = -fs*Ps/rhos**2
    
    dvdts = np.zeros_like(xs)
    dAdts = np.zeros_like(ms)
    drhodts = np.zeros_like(ms)
    for i in range(N-1):
        dist_ij = np.sum((xs[i] - xs[i+1:])**2, axis=1)**0.5
        dist_ij[dist_ij < 1e-4] = 1e-4
        nabla_ij = (xs[i] - xs[i+1:]) / dist_ij.reshape(-1,1)
        dwdr_hi = calcW(dist_ij,hs[i]*np.ones(N-1-i),D)[1]
        dwdr_hj = calcW(dist_ij,hs[i+1:],D)[1]

        # self-gravity
        if selfgravity:
            tmp = G*nabla_ij*dphidr(dist_ij, eps).reshape(-1,1)
            dvdts[i] += -np.sum(ms[i+1:].reshape(-1,1)*tmp, axis=0)
            dvdts[i+1:] += ms[i]*tmp

        # inviscid: calculate dv/dt
        ind = np.where(dist_ij < 2*hs[i])[0]
        dvdts[i] += np.sum(((ms[i+1:][ind]*dwdr_hi[ind]).reshape(-1,1)*nabla_ij[ind]), axis=0)*tmps[i]
        dvdts[i+1:][ind] -= ms[i]*tmps[i]*nabla_ij[ind]*dwdr_hi[ind].reshape(-1,1)

        ind = np.where(dist_ij < 2*hs[i+1:])[0]
        dvdts[i] += np.sum((ms[i+1:][ind]*tmps[i+1:][ind]*dwdr_hj[ind]).reshape(-1,1)*nabla_ij[ind], axis=0)
        dvdts[i+1:][ind] -= (ms[i+1:][ind]*tmps[i+1:][ind]*dwdr_hj[ind]).reshape(-1,1)*nabla_ij[ind]

        # artificial viscosity
        if abs(alpha*beta) < 1e-6:
            continue

        # calculate the viscosity factor
        vij = vs[i] - vs[i+1:]
        tmp = np.sum(vij*(xs[i] - xs[i+1:]), axis=1)
        ii = tmp >= 0.

        h_ij = (hs[i] + hs[i+1:])/2.
        rho_ij = (rhos[i] + rhos[i+1:])/2.
        c_ij = ((Ps[i]/rhos[i])**0.5 + (Ps[i+1:]/rhos[i+1:])**0.5)/2.

        mu_ij = h_ij*tmp/(dist_ij**2 + epsilon*h_ij**2)
        Pi_ij = (-alpha*c_ij*mu_ij + beta*mu_ij**2)/rho_ij
        Pi_ij[ii] = 0.

        # calculate dv/dt and dA/dt
        dwdr_av = 0.5*(dwdr_hi + dwdr_hj)
        tmp1 =  (Pi_ij*dwdr_av).reshape(-1,1) * nabla_ij
        dvdts[i] -= np.sum(ms[i+1:].reshape(-1,1)*tmp1, axis=0)
        dvdts[i+1:] += ms[i]*tmp1

        tmp2 = 0.5*(gamma - 1)*np.sum(vij*tmp1, axis=1)
        dAdts[i] += np.sum(ms[i+1:]*tmp2)/rhos[i]**(gamma - 1)
        dAdts[i+1:] += ms[i]*tmp2/rhos[i+1:]**(gamma - 1)

        # drho/dt
        drhodts[i] += 1./fs[i]*np.sum((ms[i+1:]*dwdr_hi).reshape(-1,1)*vij*nabla_ij)
        drhodts[i+1:] += -1./fs[i+1:]*ms[i]*dwdr_hj*np.sum(vij*nabla_ij, axis=1)
        
    return dvdts, dAdts, drhodts

# def dvdt_g(ms, xs, h, eps, dphidr, G):
#     '''
#     Calculate dv/dt caused by self gravity.
#     We assume a universal softening length eps.
#     dphidr specifies the gradient of the gravitational potential with respct to r.
#     The gravitational constant G depends on what units you use.
#     '''
#     # ms = np.asarray(ms, dtype=float)
#     # xs = np.asarray(xs, dtype=float).reshape(len(ms),-1)
    
#     N = ms.size # number of particles
#     # if N == 1:
#     #     return 0., 0.
    
#     # xs = xs.reshape(N,-1)
#     D = xs.shape[1] # dimension of the problem
    
#     dvdts = np.zeros_like(xs)
#     for i in range(N-1):
#         dist_ij = np.sum((xs[i] - xs[i+1:])**2, axis=1)**0.5
#         dist_ij[dist_ij < 1e-4] = 1e-4
#         nabla_ij = (xs[i] - xs[i+1:]) / dist_ij.reshape(-1,1)

#         tmp = G*nabla_ij*dphidr(dist_ij, eps).reshape(-1,1)
#         dvdts[i] += -np.sum(ms[i+1:].reshape(-1,1)*tmp, axis=0)
#         dvdts[i+1:] += ms[i]*tmp
        
#     return dvdts







