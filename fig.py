import numpy as np
import matplotlib.pyplot as plt
import pickle
import h5py

def fig1234():
    fname = ['shocktube.dat', 'shocktube_av.dat', 
             'shocktube_av_vsl.dat', 'blastwave_av_vsl.dat', 
             'shocktube_pysph.hdf5', 'blastwave_pysph.hdf5']
    for i in range(4):
        with open(fname[i],'rb') as f:
            rst = pickle.load(f)
        xs, vs, As, hs, rhos, fs, Ps = rst[-1]

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

        if i in [2,3]: # read pysph output
            data = h5py.File(fname[i+2])
            arrs = data['particles']['fluid']['arrays']
            x = np.array(arrs['x'])
            rho = np.array(arrs['rho'])
            P = np.array(arrs['p'])
            v = np.array(arrs['u'])
            axes[0][0].plot(x,rho,'b')
            axes[0][1].plot(x,P,'b')
            axes[1][0].plot(x,v,'b')
            axes[1][1].plot(x,P/0.4/rho,'b')
        
        plt.show()

def fig567():
    ns = [20,50,80,110,140,170]
    fig1,axes1 = plt.subplots(figsize=(16,6),nrows=2,ncols=6)
    fig2,axes2 = plt.subplots(figsize=(12,6),nrows=1,ncols=2)
    fig3,axes3 = plt.subplots(figsize=(12,7),nrows=2,ncols=3)

    with open('EvrardTest.dat','rb') as f: # my simulation results
        rst = pickle.load(f)

    # fig 5 and 7
    for i in range(6):
        n = ns[i]
        t = n*0.01+0.005
        xs, vs, As, hs, rhos, fs, Ps = rst[n]

        # fig 5
        fig1.subplots_adjust(left=0.05,right=0.95)
        axes1[0][i].plot(xs[:,1],xs[:,2],'k.',markersize=2)
        axes1[0][i].set_xlim(-1.5,1.5)
        axes1[0][i].set_ylim(-1.5,1.5)
        axes1[0][i].set_xticks([])
        axes1[0][i].set_yticks([])
        axes1[0][i].set_aspect(1)
        if i == 0:
            axes1[0][i].set_yticks([-1.5,-1,-0.5,0,0.5,1,1.5])
        axes1[0][i].text(1.5,1.5,r'$t=$%.3f' % t,fontsize=16,
                         horizontalalignment='right',verticalalignment='bottom')

        data = h5py.File('gadget_snapshot_'+('0'+str(n))[-3:]+'.hdf5')
        x = np.array(data['PartType0']['Coordinates'])
        rho = np.array(data['PartType0']['Density'])
        axes1[1][i].plot(x[:,1],x[:,2],'k.',markersize=2)
        axes1[1][i].set_xlim(-1.5,1.5)
        axes1[1][i].set_ylim(-1.5,1.5)
        axes1[1][i].set_xticks([-1.5,-1,-0.5,0,0.5,1,1.5])
        axes1[1][i].set_yticks([])
        axes1[1][i].set_aspect(1)
        if i == 0:
            axes1[1][i].set_yticks([-1.5,-1,-0.5,0,0.5,1,1.5])

        # fig 7
        fig3.subplots_adjust(top=0.95,bottom=0.1)
        nr = i//3
        nc = i%3
        rs = np.sum(xs**2,axis=1)**0.5
        ind = np.argsort(rs)
        axes3[nr][nc].plot(rs[ind],rhos[ind],'k')
        r = np.sum(x**2,axis=1)**0.5
        ind = np.argsort(r)
        axes3[nr][nc].plot(r[ind],rho[ind],'k--')
        xlim = axes3[nr][nc].get_xlim()
        ylim = axes3[nr][nc].get_ylim()
        axes3[nr][nc].text(xlim[-1],ylim[-1],r'$t=$%.3f' % t,fontsize=16,
                         horizontalalignment='right',verticalalignment='bottom')
    
    fig1.text(0.5,0.02,r'$x$',fontsize=16)
    fig1.text(0.01,0.5,r'$y$',fontsize=16,rotation=90)

    fig3.text(0.5,0.01,r'$r$',fontsize=16)
    fig3.text(0.05,0.5,r'$\rho$',fontsize=16,rotation=90)

    # fig 6
    fig2.subplots_adjust(left=0.07,right=0.93)

    gadget_energy = np.loadtxt('gadget_energy.txt') # energy output by gadget

    Eint = np.zeros(len(rst)) # these are used to store energies of my simulation
    Epot = Eint.copy()
    Ekin = Eint.copy()
    Etot = Eint.copy()
    phi = lambda r: -1./(r**2 + 0.004**2)**0.5

    ts = 0.005+0.01*np.linspace(0,len(rst)-1,len(rst))
    m = 1/1472.
    for n in range(len(rst)):
        xs, vs, As, hs, rhos, fs, Ps = rst[n]
        Eint[n] = np.sum(m*Ps/0.4/rhos)
        Ekin[n] = 0.5*np.sum(m*vs**2)

        for i in range(len(xs)-1):
            dist = np.sum((xs[i] - xs[i+1:])**2,axis=1)**0.5
            Epot[n] += np.sum(m**2*phi(dist))

        Etot = Eint + Epot + Ekin

    axes2[0].plot(ts,Eint,'k',label='Internal Energy')
    axes2[0].plot(ts,Epot,'b',label='Potential Energy')
    axes2[0].plot(ts,Ekin,'r',label='Kinetic Energy')
    axes2[0].plot(ts,Etot,'m',label='Totoal Energy')
    axes2[0].legend(loc=2)
    axes2[0].set_xlim(0,2)

    axes2[1].plot(gadget_energy[:,0],gadget_energy[:,1],'k')
    axes2[1].plot(gadget_energy[:,0],gadget_energy[:,2],'b')
    axes2[1].plot(gadget_energy[:,0],gadget_energy[:,3],'r')
    axes2[1].plot(gadget_energy[:,0],gadget_energy[:,1:4].sum(axis=1),'m')
    axes2[1].set_xlim(0,2)

    fig2.text(0.5,0.02,r'$t$',fontsize=16)
    fig2.text(0.02,0.5,r'$E$',fontsize=16,rotation=90)

    plt.show()


def fig891011():
    offset = [0,0.2,0.5,0.8,1.2]

    ns = [0,30,60,90,120,150]
    ts = [10,1510,3010,4510,6010,7510]
    N1 = 301

    for n in range(4):
        fig,axes = plt.subplots(figsize=(16,10),nrows=5,ncols=6)
        fig.subplots_adjust(left=0.05,right=0.95,bottom=0.03,top=0.97)

        for j in range(5):
            with open('planet_test'+str(n)+'_'+str(j)+'.dat','rb') as f:
                rst = pickle.load(f)
            for i in range(6):
                xs = rst[ns[i]][0]
                axes[j][i].clear()
                axes[j][i].plot(xs[0:N1,0]/1e8,xs[0:N1,1]/1e8,'k.')
                axes[j][i].plot(xs[N1: ,0]/1e8,xs[N1: ,1]/1e8,'b.')
                axes[j][i].set_xlim(-(1.5+0.2*n),1.5+0.2*n) 
                axes[j][i].set_ylim(-(1.5+0.2*n)+offset[j]/2,1.5+0.2*n+offset[j]/2)
                axes[j][i].set_aspect(1)
                axes[j][i].set_xticks([])
                axes[j][i].set_yticks([])
                if j == 0:
                    axes[j][i].text(1.5+0.2*n,1.5+0.2*n,r'$t=$%d' % ts[i],fontsize=16,
                                    horizontalalignment='right',verticalalignment='bottom')

        fig.text(0.5,0.01,r'$x$',fontsize=16)
        fig.text(0.01,0.5,r'$y$',fontsize=16,rotation=90) 
        plt.show()

def main():
    fig1234()
    fig567()
    fig891011()

if __name__ == '__main__':
    main()
        