'''
Test the standard shocktube problem.
'''
# import sys
# sys.path.append('/n/home10/xiaohanwu/SPH/')
from SPH_functions import *
from integrate import Euler_gas

class Shocktube(object):
    '''
    1D shocktube problem.
    '''
    def __init__(self, bxl = 0.02, bxr = 0.04, adjustSmoothLen = False):
        '''
        The setups are all standard. 
        bxl and bxr give the positions of the outermost particles. I use the 
          particles beyond [-0.5,0.5] as "boundary layers".
        '''
        self.xs0 = np.concatenate((np.arange(-0.5-bxl+0.01/8/2,0,0.01/8), 
                                   np.arange(0.01/2,0.5+bxr,0.01)))
        self.N = len(self.xs0)
        self.xs0 = self.xs0.reshape(self.N,-1)
        self.vs0 = np.zeros_like(self.xs0)
        self.D = self.xs0.shape[1]

        self.index = np.linspace(0,self.N-1,self.N,dtype=int)
        iil = self.xs0 < 0.
        iir = self.xs0 >= 0.

        self.gamma = 1.4
        rhos0 = np.concatenate((np.ones(iil.sum())*1., np.ones(iir.sum())*0.125))
        Ps0 = np.concatenate((np.ones(iil.sum())*1., np.ones(iir.sum())*0.1))
        self.As0 = Ps0/rhos0**self.gamma

        self.ms = np.ones(self.N)*1.*0.01/8
        self.hs0 = 0.01*1.2*np.ones_like(self.ms)
        self.adjustSmoothLen = adjustSmoothLen
        self.rhos0, tmp, self.fs0 = calcRho(self.ms,self.xs0,self.hs0,self.index,
                                            adjustSmoothLen = self.adjustSmoothLen)
        self.Ps0 = self.As0*self.rhos0**1.4

    def BC(self, xs0, vs0, xs_new, vs_new):
        '''
        Boundary condition.
        '''
        xs = xs_new.copy()
        vs = vs_new.copy()
        
        ii_in = (xs0 >= -0.5) & (xs0 <= 0.5)
        vs[~ii_in] = 0.
        
        ii = (xs_new < -0.5) * ii_in
        xs[ii] = -1 - xs_new[ii]
        vs[ii] = -vs_new[ii]
        
        ii = (xs_new > 0.5) * ii_in
        xs[ii] = 1 - xs_new[ii]
        vs[ii] = -vs_new[ii]
        
        return xs, vs

    def run(self, ts, alpha, beta, epsilon = 0.01, dt = 5e-4):
            '''
            Get solutions at the times specified by ts.
            '''
            rst = Euler_gas(ts, dt, self.BC, 
                            self.ms, self.xs0, self.vs0, self.hs0, self.rhos0, 
                            self.fs0, self.Ps0, alpha, beta, epsilon, 
                            gamma = self.gamma, 
                            selfgravity = False,
                            adjustSmoothLen = self.adjustSmoothLen)

            return rst

def main():
    # outpath = '/n/home10/xiaohanwu/SPH/'
    outpath = './'

    # fix smoothing length
    data = Shocktube(adjustSmoothLen = False)

    rst = data.run([0.05,0.10,0.15,0.2], 0, 0)
    xs, vs, As, hs, rhos, fs, Ps = rst[-1]
    
    with open(outpath+'shocktube.dat','wb') as f:
        pickle.dump(rst,f)

    fig, axes = plt.subplots(figsize=(10,10),nrows=2,ncols=2)
    axes[0][0].plot(xs,rhos,'k.')
    axes[0][0].set_xlabel(r'$x$', fontsize=16)
    axes[0][0].set_ylabel(r'$\rho$',fontsize=16)
    axes[0][1].plot(xs,Ps,'k.')
    axes[0][1].set_xlabel(r'$x$', fontsize=16)
    axes[0][1].set_ylabel(r'$P$',fontsize=16)
    axes[1][0].plot(xs,vs,'k.')
    axes[1][0].set_xlabel(r'$x$', fontsize=16)
    axes[1][0].set_ylabel(r'$v$',fontsize=16)
    axes[1][1].plot(xs,Ps/(0.4*rhos),'k.')
    axes[1][1].set_xlabel(r'$x$', fontsize=16)
    axes[1][1].set_ylabel(r'$u$',fontsize=16)
    plt.show()

    rst = data.run([0.05,0.1,0.15,0.2], 1, 2)
    xs, vs, As, hs, rhos, fs, Ps = rst[-1]

    with open(outpath+'shocktube_av.dat','wb') as f:
        pickle.dump(rst,f)

    fig, axes = plt.subplots(figsize=(10,10),nrows=2,ncols=2)
    axes[0][0].plot(xs,rhos,'k.')
    axes[0][0].set_xlabel(r'$x$', fontsize=16)
    axes[0][0].set_ylabel(r'$\rho$',fontsize=16)
    axes[0][1].plot(xs,Ps,'k.')
    axes[0][1].set_xlabel(r'$x$', fontsize=16)
    axes[0][1].set_ylabel(r'$P$',fontsize=16)
    axes[1][0].plot(xs,vs,'k.')
    axes[1][0].set_xlabel(r'$x$', fontsize=16)
    axes[1][0].set_ylabel(r'$v$',fontsize=16)
    axes[1][1].plot(xs,Ps/(0.4*rhos),'k.')
    axes[1][1].set_xlabel(r'$x$', fontsize=16)
    axes[1][1].set_ylabel(r'$u$',fontsize=16)
    plt.show()

    # variable smoothing length
    data = Shocktube(adjustSmoothLen = True)

    rst = data.run([0.05,0.1,0.15,0.2], 1, 2)
    xs, vs, As, hs, rhos, fs, Ps = rst[-1]
    
    with open(outpath+'shocktube_av_vsl.dat','wb') as f:
        pickle.dump(rst,f)

    fig, axes = plt.subplots(figsize=(10,10),nrows=2,ncols=2)
    axes[0][0].plot(xs,rhos,'k.')
    axes[0][0].set_xlabel(r'$x$', fontsize=16)
    axes[0][0].set_ylabel(r'$\rho$',fontsize=16)
    axes[0][1].plot(xs,Ps,'k.')
    axes[0][1].set_xlabel(r'$x$', fontsize=16)
    axes[0][1].set_ylabel(r'$P$',fontsize=16)
    axes[1][0].plot(xs,vs,'k.')
    axes[1][0].set_xlabel(r'$x$', fontsize=16)
    axes[1][0].set_ylabel(r'$v$',fontsize=16)
    axes[1][1].plot(xs,Ps/(0.4*rhos),'k.')
    axes[1][1].set_xlabel(r'$x$', fontsize=16)
    axes[1][1].set_ylabel(r'$u$',fontsize=16)
    plt.show()

if __name__ == '__main__':
    main()





