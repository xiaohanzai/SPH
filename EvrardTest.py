'''
Evrard test: adiabatic collapse of a gas sphere.
'''
# import sys
# sys.path.append('/n/home10/xiaohanwu/SPH/')
from SPH_functions import *
from integrate import Euler_gas
import h5py
from mpl_toolkits.mplot3d import Axes3D

class EvrardTest(object):
    '''
    Adiabatic collapse of a gas sphere.
    '''
    def __init__(self, adjustSmoothLen = False, h = 0.12):
        '''
        The initial conditions are taken from the gadget input file. h is the smoothing length, 
          if you want to fix it constant.
        '''
        data = h5py.File('EvrardInput.hdf5')
        # data = h5py.File('/n/hernquistfs2/xiaohanwu/project1/work/gadget_tests/test2/output_hdf5/snapshot_000.hdf5')
        gas = data['PartType0']
        
        self.xs0 = np.array(gas['Coordinates'])
        self.vs0 = np.array(gas['Velocities'])
        self.ms = np.array(gas['Masses'])
        self.D = self.xs0.shape[1]
        self.N = len(self.xs0)
        self.index = np.linspace(0,self.N-1,self.N,dtype=int)

        self.adjustSmoothLen = adjustSmoothLen
        if self.adjustSmoothLen:
            rhos0 = np.array(gas['Density'])
            tmp1, self.hs0, tmp2 = adjustSmLen(self.ms, self.xs0, rhos0, 
                                               np.ones_like(self.ms)*0.01, np.ones_like(self.ms),
                                               fixRho = True)
        else:
            self.hs0 = h*np.ones_like(self.ms)

        self.rhos0, tmp, self.fs0 = calcRho(self.ms, self.xs0, self.hs0, self.index,
                                            adjustSmoothLen = self.adjustSmoothLen)

        us0 = np.array(gas['InternalEnergy'])
        self.gamma = 1.4
        self.Ps0 = (self.gamma-1)*self.rhos0*us0
        self.As0 = self.Ps0/self.rhos0**(self.gamma-1)
        
    def BC(self, xs0, vs0, xs_new, vs_new):
        '''
        No boundary condition in this case.
        '''
        return xs_new, vs_new

    def dphidr(self, r, eps):
        '''
        Used for calculating gravity.
        '''
        return r/(r**2 + eps**2)**1.5
        
    def run(self, ts, alpha = 0.8, beta = 1.6, epsilon = 0.01, dt = 0.5e-2, 
            eps = 0.004, G = 1):
            '''
            Get solutions at the times specified by ts.
            '''
            rst = Euler_gas(ts, dt, self.BC, 
                            self.ms, self.xs0, self.vs0, self.hs0, self.rhos0, 
                            self.fs0, self.Ps0, alpha, beta, epsilon, 
                            gamma = self.gamma, 
                            selfgravity = True, eps = eps, dphidr = self.dphidr, G = G, 
                            adjustSmoothLen = self.adjustSmoothLen)

            return rst

def main():
    # outpath = '/n/home10/xiaohanwu/SPH/'
    outpath = './'

    data = EvrardTest()
    rst = data.run(np.arange(5e-3,2.0,0.01), dt=5e-3)

    with open(outpath+'EvrardTest.dat','wb') as f:
        pickle.dump(rst,f)

if __name__ == '__main__':
    main()






