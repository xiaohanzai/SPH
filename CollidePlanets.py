'''
A stupid planet collision simulation. Totally wrong...
'''
import sys
sys.path.append('/n/home10/xiaohanwu/SPH/')
from SPH_functions import *
from integrate import *

class CollidePlanets(object):
    def __init__(self, xc1, vc1, h1, xc2, vc2, h2, 
                 planet1file = 'Planet300.dat', planet2file = 'Planet300.dat', 
                 gamma = 1.4, adjustSmoothLen = False):
        '''
        Read single planet data from file. Assign each planet with a center position and a 
          velocity. Also assign them with initial smoothing lengths.
        '''
        # path = '/n/home10/xiaohanwu/SPH/'
        path = './'
        data1 = np.loadtxt(path+planet1file)
        data2 = np.loadtxt(path+planet2file)
        
        self.ms = np.concatenate((data1[:,6], data2[:,6]))
        self.xs0 = np.concatenate((data1[:,0:3]+np.asarray(xc1), data2[:,0:3]+np.asarray(xc2)))
        self.vs0 = np.concatenate((data1[:,3:6]+np.asarray(vc1), data2[:,3:6]+np.asarray(vc2)))
        self.N = len(self.ms)
        self.D = self.xs0.shape[1]
        self.index = np.linspace(0,self.N-1,self.N, dtype=int)
        self.adjustSmoothLen = adjustSmoothLen
        
        rhos0 = np.concatenate((data1[:,7], data2[:,7]))
        Ps0 = np.concatenate((data1[:,8], data2[:,8]))
        self.hs0 = np.concatenate((np.ones(len(data1))*h1, np.ones(len(data2))*h2))
        self.gamma = gamma
        C = Ps0/rhos0
        
        self.rhos0, tmp, self.fs0 = calcRho(self.ms, self.xs0, self.hs0, self.index, 
                                            adjustSmoothLen = self.adjustSmoothLen)
        self.Ps0 = C*self.rhos0
        self.As0 = self.Ps0/self.rhos0**(self.gamma-1)
    
    def dphidr(self, r, eps):
        return r/(r**2 + eps**2)**1.5
    
    def BC(self, xs0, vs0, xs_new, vs_new):
        return xs_new, vs_new
    
    def run(self, ts, eps, alpha = 1., beta = 2., epsilon = 0.01, dt = 10., G = 6.67e-11):
            rst = Euler_gas(ts, dt, self.BC, 
                            self.ms, self.xs0, self.vs0, self.hs0, self.rhos0, 
                            self.fs0, self.Ps0, alpha, beta, epsilon, 
                            gamma = self.gamma, 
                            selfgravity = True, eps = eps, dphidr = self.dphidr, G = G,
                            adjustSmoothLen = self.adjustSmoothLen)

            return rst

def main():
    data = CollidePlanets([-1e8,0,0], [25e3,0,0], 1e7, [1e8,0,0], [-25e3,0,0], 1e7)
    rst = data.run(np.arange(10,2000,50), 5e6, dt=5.)

    # outpath = '/n/home10/xiaohanwu/SPH/'
    outpath = './'
    with open(outpath+'planet_test0_0.dat','wb') as f:
        pickle.dump(rst,f)

if __name__ == '__main__':
    main()


