
# coding: utf-8

# In[1]:



# In[2]:


import numpy as np
import pandas as pd
import os
import pickle as pl
import tables
import h5py
from scipy.constants import c, pi
import gc
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


# In[3]:


from data_plotters_animators import read_variables
from functions import *
import warnings 
warnings.filterwarnings('ignore')


# In[4]:


import tables


# In[5]:



import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
font = {'size'   : 16}

matplotlib.rc('font', **font)


# In[6]:


from numpy.fft import fftshift
#from fft_module import *
import scipy


# In[7]:


def selmier(l):
    a = 0.6961663*l**2/(l**2 - 0.0684043**2)
    b = 0.4079426*l**2/(l**2 - 0.1162414**2)
    c = 0.8974794*l**2/(l**2 - 9.896161**2)
    return (1 + a + b +c)**0.5


# In[8]:


class Conversion_efficiency(object):

    def __init__(self, freq_band, possition, filename=None, filepath='',filename2 = 'CE',filepath2 = 'output_final/'):
        self.L = 18
        self.n = 1.444
        
        
        self.variables = ('P_p', 'P_s', 'f_p', 'f_s','l_p','l_s,' 'P_out', 'P_bef','CE','rounds')
        
        self.spec, self.fv, self.t, self.P0_p, self.P0_s,self.f_p, self.f_s, self.P_bef,self.ro,U_large,tt,self.u_large =            self.load_spectrum('0',filename, filepath)
        self.P_max = np.max(w2dbm(self.spec))
        
        self.spec, self.fv, self.t, self.P0_p, self.P0_s,self.f_p, self.f_s, self.P_bef,self.ro,U_large,tt,self.u_large =            self.load_spectrum(possition,filename, filepath)
        self.tt = tt
       
        self.freq_band = freq_band

        self.U_large = np.asanyarray(U_large)
        self.nt = np.shape(self.spec)[0]
        self.possition = possition
        if possition == '2' or possition == '1':
            print('finding signal')
            fv_id = self.pos_of_signal()
            self.P_in = self.P0_p + self.P0_s
            
        else:
            print('finding idler')
            fv_id = self.pos_of_idler()
            self.P_in = self.P0_p + self.P0_s

        
        lami = 1e-3*c/self.fv[fv_id]
        self.lam_wanted = lami
        self.n = selmier(1e-3*self.lam_wanted)
        self.time_trip = self.L*self.n/c
        self.lamp = 1e-3*c/self.f_p
        self.l_s = 1e-3*c/self.f_s
        self.U_large_norm =  w2dbm(np.abs(self.U_large)**2) - self.P_max 
        P_out_vec = []
        P_out_vec_casc = []
        self.fv_id = fv_id
        start, end= self.fv[fv_id] - freq_band, self.fv[fv_id] + freq_band
        try:
            fv_id_c = self.pos_of_cascade()
        except ValueError:
            fv_id_c = 0
        start_c, end_c = self.fv[fv_id_c] - freq_band, self.fv[fv_id_c] + freq_band
        for i in U_large:
            self.spec = np.abs(i)**2
            P_out_vec.append(self.calc_P_out(start,end))
            P_out_vec_casc.append(self.calc_P_out(start_c,end_c))
        self.P_out_vec_casc = np.asanyarray(P_out_vec_casc)
        self.P_out_vec = np.asanyarray(P_out_vec)
        self.P_out = np.mean(P_out_vec[::-1][:500])
        self.CE = self.calc_CE()
        

        self.std = { i : None for i in self.variables}
        self.std['P_out'] = np.std(P_out_vec[-500:])
        self.std['CE'] = self.std['P_out']*self.CE/self.P_in
        self.rin = self.time_trip*self.std['P_out']**2 / self.P_out**2
    
        self.std['rin'] = self.rin
        read_write_CE_table(filename2,var = None, rin = self.rin,P_p = self.P0_p, P_s = self.P0_s, f_p = self.f_p,
                                         f_s = self.f_s,P_out = self.P_out,P_bef = self.P_bef, CE = self.CE, var2 = 'CE',std = self.std,file_path=filepath2)
        self.spec = np.mean(np.abs(U_large[0:][:])**2, axis = 0)
        self.spec = np.abs(U_large[-1][:])**2
        self.spec_s = w2dbm(self.spec)-self.P_max 
        return None

    
    def pos_of_idler(self):
        U_sum = np.sum(np.abs(self.U_large)**2, axis = 0)
        fp_id = np.where(U_sum == np.max(U_sum))[0][0]
        plom = fp_id+50
        fv_id = np.where(U_sum[plom:] == np.max(U_sum[plom:]))[0][0]
        fv_id += plom-1
        return fv_id
    
    def pos_of_signal(self):
        U_sum = np.sum(np.abs(self.U_large)**2, axis = 0)
        fp_id = np.where(U_sum == np.max(U_sum))[0][0]
        plom = fp_id - 50
        fv_id = np.where(U_sum[:plom] == np.max(U_sum[:plom]))[0][0]
        return fv_id
    
    
    def pos_of_cascade(self):
        sig_id = self.pos_of_signal() - 50
        U_sum = np.sum(np.abs(self.U_large)**2, axis = 0)
        plom = sig_id
        fv_id = np.where(U_sum[:plom] == np.max(U_sum[:plom]))[0][0]
        return fv_id
    
    
    def load_spectrum(self, possition,filename='data_large', filepath=''):
        with h5py.File(filepath+filename+'.hdf5','r') as f: 
            l = f.get(possition)
            U_large = ()
            u_large = ()
            integers_list = [int(i) for i in l.keys()]
            integers_list.sort()
            integers_generator = (str(n) for n in integers_list)
            for i in integers_generator:
                steady_state = i
                layers = possition + '/' + steady_state
                D = read_variables(filename,layers, filepath)
                U = D['U']
                u = D['u']
                u_large += (u,)
                U_large += (U,)

            fv = D['fv']
            ro = D['ro']

            Uabs = w2dbm(np.abs(U)**2)
            P0_s = D['P0_s']
            P0_p = D['P0_p']
            t = D['t']
            f_p = D['f_p']
            f_s = D['f_s']
            layers = '1/0'
            
            D = read_variables(filename,layers, filepath)
            Uabss =np.abs(D['U']*(t[1] - t[0]))**2
            fvs = D['fv']
            tt = D['t']

            P_bef = simps(Uabss,fvs)
            P_bef /= (2*np.max(tt))
            #print(P_bef)
        return dbm2w(Uabs), fv,t, P0_p, P0_s, f_p, f_s,P_bef, ro, U_large,t, u_large



    def calc_P_out(self,start,end):
        i = np.where(
            np.abs(self.fv - start) == np.min(np.abs(self.fv - start)))[0][0]
        j = np.where(
            np.abs(self.fv - end) == np.min(np.abs(self.fv - end)))[0][0]
        E_out = simps(self.spec[i:j]*(self.tt[1] - self.tt[0])**2, self.fv[i:j])
        P_out = E_out/(2*np.max(self.tt))
        return P_out   


    def calc_CE(self):
        CE = 100*self.P_out/ (self.P0_p + self.P0_s)
        return CE


    
    def P_out_round(self,filepath,filesave):
        """Plots the output average power with respect to round trip number"""
        self.l_p = 1e-3*c/self.f_p
        fig = plt.figure(figsize=(20,10))
        plt.plot(range(len(self.P_out_vec)), self.P_out_vec)
        plt.xlabel('Rounds')
        plt.ylabel('Output Power')
        plt.title(f"$P_p=$ {float(CE.P0_p):.{2}} W, $P_s=$ {float(CE.P0_s*1e3):.{2}} mW, $\\lambda_p=$ {float(CE.lamp):.{6}} nm,  $\\lambda_s=$ {float(CE.l_s):.{6}} nm, maximum output at: {float(CE.lam_wanted):.{6}} nm ({float(1e-3*c/CE.lam_wanted):.6} Thz)")
        plt.savefig(filepath+'power_per_round'+filesave+'.png')

        data = (range(len(self.P_out_vec)), self.P_out_vec)
        _data ={'pump_power':self.P0_p, 'pump_wavelength': self.l_p, 'out_wave': self.lam_wanted}
        with open(filepath+'power_per_round'+filesave+'.pickle','wb') as f:
            pl.dump(fig,f)
        plt.clf()
        plt.close('all')
        
        #diff = [self.P_out_vec[i+1] - self.P_out_vec[i] for i in range(len(self.P_out_vec) - 1)]
        #fig = plt.figure(figsize=(20,10))
        #plt.plot(diff)
        #plt.title('mean and std of the last 100: '+str(np.std(diff))+' '+str(np.mean(diff)))
        #plt.savefig(filepath+'finite_dif'+filesave+'.png')
    def P_out_round_casc(self,filepath,filesave):
        """Plots the output average power with respect to round trip number"""
        self.l_p = 1e-3*c/self.f_p
        fig = plt.figure(figsize=(20,10))
        plt.plot(range(len(self.P_out_vec)), self.P_out_vec_casc)
        plt.xlabel('Rounds')
        plt.ylabel('Output Power')
        plt.title(f"$P_p=$ {float(CE.P0_p):.{2}} W, $P_s=$ {float(CE.P0_s*1e3):.{2}} mW, $\\lambda_p=$ {float(CE.lamp):.{6}} nm,  $\\lambda_s=$ {float(CE.l_s):.{6}} nm, maximum output at: {float(CE.lam_wanted):.{6}} nm ({float(1e-3*c/CE.lam_wanted):.6} Thz)")
        plt.savefig(filepath+'power_per_round_casc'+filesave+'.png')
        data = (range(len(self.P_out_vec)), self.P_out_vec)
        _data ={'pump_power':self.P0_p, 'pump_wavelength': self.l_p, 'out_wave': self.lam_wanted}
        with open(filepath+'power_per_round_casc'+filesave+'.pickle','wb') as f:
            pl.dump(fig,f)
        plt.clf()
        plt.close('all')


# In[9]:


def read_write_CE_table(filename,var = None,rin = None, P_p = None, P_s = None, f_p = None, f_s = None,P_out = None, P_bef = None,CE = None, var2 = 'CE',std = None,file_path=''):
        
        """ Given values of the parameters this function uses pandas to open an
            hdf5 file and append to the dataframe there. It also returns the full data
            for post-processing. 
            
            It returns a tuple of 2 numpy arrays the first with the variable var and the second with
            the conversion efficiencty (as default). If no input is given( default then it just reads the )
        """
        try:
            l_s = 1e-3*c/f_s
        except TypeError:
            l_s = None
            pass
        try:
            l_p = 1e-3*c/f_p
        except TypeError:
            l_p = None
            pass
        print(l_s)
        A = np.array([P_p, P_s, f_p, f_s,l_s,l_p, P_out, P_bef, CE,rin]).T
        a = pd.DataFrame(A, index = ['P_p', 'P_s','f_p', 'f_s','l_s','l_p', 'P_out','P_bef', 'CE','rin']).T
        try:
            ab = pd.read_hdf(file_path+filename+'.hdf5')
            if not(A.any() == None):
                ab = ab.append(a, ignore_index=True)
        except IOError:
            if not(A.any() == None):
                ab = a
            else: 
                sys.exit("There is no data in file or given")
            pass
        store = ab.to_hdf(file_path+filename+'.hdf5',key='a')
        b = pd.DataFrame.from_dict([std])
        try:
            ba = pd.read_hdf(file_path+filename+'_std.hdf5', key = 'b')
            if not(A.any() == None):
                ba = ba.append(b, ignore_index=True)
        except IOError:
            if not(A.any() == None):
                ba = b
            else: 
                sys.exit("There is no data in file or given")
            pass
        store2 = ba.to_hdf(file_path+filename+'_std.hdf5', key = 'b')

        if var is None:
            return None
        else:
            return ab[var].as_matrix(),ab[var2].as_matrix(),ba


def plot_rin(var,var2 = 'rin',filename = 'CE', filepath='output_final/', filesave= None):
    var_val, CE,std = read_write_CE_table(filename,var,var2 = var2,file_path=filepath)
    std = std[var2].as_matrix()
    if var is 'arb':
        var_val = [i for i in range(len(CE))] 
    fig = plt.figure(figsize=(20.0, 10.0))
    plt.plot(var_val, 10*np.log10(CE),'o-')
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
    plt.xlabel(var)
    plt.ylabel('RIN (dBc/hz)')
    plt.savefig(filesave+'.png',bbox_inches = 'tight')
    data = (var_val, CE,std)
    with open(str(filesave)+'.pickle','wb') as f:
        pl.dump((fig,data),f)
    plt.clf()
    plt.close('all')

    return None
        
        

def plot_CE(var,var2 = 'CE',filename = 'CE', filepath='output_final/', filesave= None):
    var_val, CE,std = read_write_CE_table(filename,var,var2 = var2,file_path=filepath)
    std = std[var2].as_matrix()
    
    if var is 'arb':
        var_val = [i for i in range(len(CE))] 
    fig = plt.figure(figsize=(20.0, 10.0))
    plt.errorbar(var_val, CE, yerr=std, capsize= 10)
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
    plt.xlabel(var)
    plt.ylabel(var2)
    plt.savefig(filesave+'.png',bbox_inches = 'tight')
    data = (var_val, CE,std)
    with open(str(filesave)+'.pickle','wb') as f:
        pl.dump((fig,data),f)
    plt.clf()
    plt.close('all')

    return None


# In[10]:


def contor_plot(CE,fmin = None,fmax = None,  rounds = None,filename = None):
    if not(fmin):
        fmin = CE.fv[CE.fv_id] - CE.freq_band
    if not(fmax):
        fmax = CE.fv[CE.fv_id] + CE.freq_band
    print(fmin,fmax)
    i = np.where(np.abs(CE.fv - fmin) == np.min(np.abs(CE.fv - fmin)))[0][0]
    j = np.where(np.abs(CE.fv - fmax) == np.min(np.abs(CE.fv - fmax)))[0][0]
    


    if rounds is None:
        rounds = np.shape(CE.U_large_norm)[0]
   
    CE.ro = range(rounds)
    x,y = np.meshgrid(CE.ro[:rounds], CE.fv[i:j])
    z = CE.U_large_norm[:rounds,i:j].T
    #print(np.shape(x), np.shape(z))
    low_values_indices = z < -60  # Where values are low
    z[low_values_indices] = -60  # All low values set to 0
    fig = plt.figure(figsize=(20,10))
    plt.contourf(x,y, z, np.arange(-60,2,2),extend = 'min',cmap=plt.cm.jet)
    plt.xlabel(r'$rounds$')
    plt.ylim(fmin,fmax)
    #plt.xlim(0,200)
    plt.ylabel(r'$f(THz)$')
    plt.colorbar()
    l_p = 1e-3*c/CE.f_p
    plt.title(f"$P_p=$ {float(CE.P0_p):.{2}} W, $P_s=$ {float(CE.P0_s*1e3):.{2}} mW, $\\lambda_p=$ {float(CE.lamp):.{6}} nm,  $\\lambda_s=$ {float(CE.l_s):.{6}} nm, maximum output at: {float(CE.lam_wanted):.{6}} nm")
    data = (CE.ro, CE.fv, z )
    _data ={'pump_power':CE.P0_p, 'pump_wavelength': l_p, 'out_wave': CE.lam_wanted}
    if filename is not None:
        plt.savefig(str(filename), bbox_inches = 'tight')
        plt.clf()
        plt.close('all')
        #with open(str(filename)+'.pickle','wb') as f:
        #    pl.dump((data,_data),f)


    else:
        plt.show()
    return None


# In[11]:


def contor_plot_time(CE, rounds = None,filename = None):

    if rounds is None:
        rounds = np.shape(CE.U_large_norm)[0]
   
    CE.ro = range(rounds)
    x,y = np.meshgrid(CE.ro[:rounds], CE.t)
    z = (np.abs(CE.u_large)**2)[:rounds,:].T / (2*np.max(CE.t))
    #print(np.shape(x), np.shape(z))
    #low_values_indices = z < -60  # Where values are low
    #z[low_values_indices] = -60  # All low values set to 0
    fig = plt.figure(figsize=(20,10))
    plt.contourf(x,y, z,cmap=plt.cm.jet)
    plt.xlabel(r'$rounds$')
    #plt.ylim(fmin,fmax)
    #plt.xlim(0,200)
    plt.ylabel(r'$f(THz)$')
    plt.colorbar()
    l_p = 1e-3*c/CE.f_p
    plt.title(f"$P_p=$ {float(CE.P0_p):.{2}} W, $P_s=$ {float(CE.P0_s*1e3):.{2}} mW, $\\lambda_p=$ {float(CE.lamp):.{6}} nm,  $\\lambda_s=$ {float(CE.l_s):.{6}} nm, maximum output at: {float(CE.lam_wanted):.{6}} nm")
    data = (CE.ro, CE.fv, z )
    _data ={'pump_power':CE.P0_p, 'pump_wavelength': l_p, 'out_wave': CE.lam_wanted}
    if filename is not None:
        plt.savefig(str(filename), bbox_inches = 'tight')
        plt.clf()
        plt.close('all')
        #with open(str(filename)+'.pickle','wb') as f:
        #    pl.dump((data,_data),f)


    else:
        plt.show()
    return None


# In[12]:


def contor_plot_anim(CE,iii,fmin = None,fmax = None,  rounds = None,filename = None):
    if not(fmin):
        fmin = CE.fv[CE.fv_id] - CE.freq_band
    if not(fmax):
        fmax = CE.fv[CE.fv_id] + CE.freq_band

    i = np.where(np.abs(CE.fv - fmin) == np.min(np.abs(CE.fv - fmin)))[0][0]
    j = np.where(np.abs(CE.fv - fmax) == np.min(np.abs(CE.fv - fmax)))[0][0]
    


    if rounds is None:
        rounds = np.shape(CE.U_large_norm)[0]
   
    CE.ro = range(rounds)
    x,y = np.meshgrid(CE.ro[:rounds], CE.fv[i:j])
    z = np.copy(CE.U_large_norm[:rounds,i:j].T)

    #print(np.shape(x), np.shape(z))
    low_values_indices = z < -60  # Where values are low
    z[low_values_indices] = -60  # All low values set to 0
    z[:,iii:] = -60

    f, (ax, ax2) = plt.subplots(2, 1,figsize = (7,1.5), sharex=True)
    
    # plot the same data on both axes
    al = ax.contourf(x,y, z, np.arange(-60,2,2),extend = 'min',cmap=plt.cm.plasma)
    al2 = ax2.contourf(x,y, z, np.arange(-60,2,2),extend = 'min',cmap=plt.cm.plasma)

    # zoom-in / limit the view to different portions of the data
    #plt.ylim(285.5,286.5)
    ax.set_ylim(249.4, 251.4)  # outliers only
    ax2.set_ylim(214, 215)  # most of the data

    # hide the spines between ax and ax2
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop='off')  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    d = .008  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    plt.ylabel(r'$f(THz)$',position=(0.5,1.1))
    #plt.xlabel(r'Oscillations')

    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off
    ax2_divider = make_axes_locatable(ax)
    cax2 = ax2_divider.append_axes("top", size="7%", pad="2%")
    cbar = f.colorbar(al2,cax=cax2,orientation='horizontal')
    cbar.ax.xaxis.set_ticks_position('top')
    #if filename is not None:
    ax.set_yticks([250.4])
    ax2.set_yticks([214.5])
    plt.savefig(str(filename), bbox_inches = 'tight')
    plt.clf()
    plt.close('all')
        #with open(str(filename)+'.pickle','wb') as f:
        #    pl.dump((data,_data),f)
    return None


def contor_plot_anim_single(CE,iii,fmin = None,fmax = None,  rounds = None,filename = None):
    if not(fmin):
        fmin = CE.fv[CE.fv_id] - CE.freq_band
    if not(fmax):
        fmax = CE.fv[CE.fv_id] + CE.freq_band

    i = np.where(np.abs(CE.fv - fmin) == np.min(np.abs(CE.fv - fmin)))[0][0]
    j = np.where(np.abs(CE.fv - fmax) == np.min(np.abs(CE.fv - fmax)))[0][0]
    


    if rounds is None:
        rounds = np.shape(CE.U_large_norm)[0]
   
    CE.ro = range(rounds)
    x,y = np.meshgrid(CE.ro[:rounds], CE.fv[i:j])
    z = np.copy(CE.U_large_norm[:rounds,i:j].T)

    #print(np.shape(x), np.shape(z))
    low_values_indices = z < -60  # Where values are low
    z[low_values_indices] = -60  # All low values set to 0
    z[:,iii:] = -60
    
    fig = plt.figure(figsize=(7,1.5))
    ax = fig.add_subplot(111)
    al2 = ax.contourf(x,y, z, np.arange(-60,2,2),extend = 'min',cmap=plt.cm.plasma)
    plt.ylim(285.5,286.5)
    #plt.xlim(0,200)
    plt.ylabel(r'$f(THz)$')

   
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off
    ax2_divider = make_axes_locatable(ax)
    cax2 = ax2_divider.append_axes("top", size="7%", pad="2%")
    cbar = fig.colorbar(al2,cax=cax2,orientation='horizontal')
    cbar.ax.xaxis.set_ticks_position('top')

    plt.savefig(str(filename), bbox_inches = 'tight')
    plt.clf()
    plt.close('all')
    return None


def P_out_round_anim(CE,iii,filesave):
    """Plots the output average power with respect to round trip number"""
    tempy = CE.P_out_vec[:iii]
    
    fig = plt.figure(figsize=(7,1.5))
    plt.plot(range(len(tempy)), tempy)
    plt.xlabel('Oscillations')
    plt.ylabel('Power')
    plt.ylim(0,np.max(CE.P_out_vec)+0.1*np.max(CE.P_out_vec))
    plt.xlim(0,len(CE.P_out_vec))
    plt.savefig(filesave+'.png',bbox_inches = 'tight')
    plt.close('all')
    plt.clf()
    return None


# In[13]:


def final_1D_spec(ii,specs,filename):

    fig = plt.figure(figsize=(10,8))
    #fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()

    ax1.plot(specs.fv, specs.spec_s, label = r'$\lambda_p$='+str(specs.l_p)+r', $\lambda_s $='+str(specs.l_s))

    ax1.set_xlabel(r'$f (THz)$')
    ax1.set_ylabel(r'spec (dB)')
    #ax1.set_xticks(np.arange(min(specs.fv), max(specs.fv)+1, 10))
    #ax1.set_ylim(260,320)
    #print(round(min(specs.fv)),round(max(specs.fv)))
    #sys.exit()
    #ax1.set_xticks(np.arange(round(min(specs.fv)),round(max(specs.fv)),10))
    new_tick_locations = ax1.get_xticks()

    def tick_function(X):
        l = 1e-3*c/X
        return ["%.2f" % z for z in l]
    
    #ax1.set_ylim(-100,1)
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(new_tick_locations)
    ax2.set_xticklabels(tick_function(new_tick_locations))
    ax2.set_xlabel(r"$\lambda (nm)$")
    #plt.ylim(260,320)
    ax1.legend()
    plt.savefig(filename+'spectrum_fopo_final'+str(ii)+'.png', bbox_inches = 'tight')
    #plt.show()
    with open(filename+str(ii)+'.pickle','wb') as f:
        pl.dump(fig,f)
    plt.clf()
    plt.close('all')

    return None


# In[14]:


def giff_it_up(i,spots,fps):
    delay = 100/fps
    com = 'convert -delay ' +str(delay)+' -loop 0 '
    for iii in spots:
        com += 'animators'+str(i) + '/contor/'+str(iii)+'.png '
    com += 'animators'+str(i) + '/contor/animation_cont.gif'
    
    os.system(com)
    com = 'convert -delay ' +str(delay)+' -loop 0 '
    for iii in spots:
        com += 'animators'+str(i) + '/power/'+str(iii)+'.png '
    com += 'animators'+str(i) + '/power/animation_power.gif'
    
    os.system(com)
    
    com = 'convert -delay ' +str(delay)+' -loop 0 '
    for iii in spots:
        com += 'animators'+str(i) + '/contor_single/'+str(iii)+'.png '
    com += 'animators'+str(i) + '/contor_single/animation_cont_single.gif'
    
    os.system(com)
    #os.system('mv animators'+str(i) + '/contor_single/animation_cont_single.gif ~/storage/Dropbox/nusod/Presentation/figs/animation_cont_single'+str(i)+'.gif' )
    #os.system('mv animators'+str(i) + '/contor/animation_cont.gif ~/storage/Dropbox/nusod/Presentation/figs/animation_cont'+str(i)+'.gif' )
    #os.system('mv animators'+str(i) + '/power/animation_power.gif ~/storage/Dropbox/nusod/Presentation/figs/animation_power'+str(i)+'.gif' )
    
    return None


# In[20]:


from os import listdir
#from os.path import , join
data_dump =  'output_dump'
outside_dirs = [f for f in listdir(data_dump)]
inside_dirs = [[f for f in listdir(data_dump+ '/'+out_dir)] for out_dir in outside_dirs ]


# In[22]:


which = 'output_dump_pump_wavelengths/7w'
which = 'output_dump_pump_wavelengths/wrong'
which = 'output_dump_pump_wavelengths'
#which = 'output_dump_pump_wavelengths/2_rounds'
#which ='output_dump_pump_powers/ram0ss0'
#which = 'output_dump/'#_pump_powers'
which_l = 'output_dump/output'



outside_vec = range(len(outside_dirs))
#outside_vec = range(2,3)
inside_vec = [range(len(inside) - 1) for inside in inside_dirs]
#inside_vec = [13]
animators = False
spots = range(0,8100,100)


# In[ ]:


os.system('rm -r output_final ; mkdir output_final')
for pos in ('4','2'):

    for ii in outside_vec:
        ii = str(ii)
        which = which_l+ ii
        os.system('rm output_final/CE.hdf5 output_final/CE_std.hdf5')
        os.system('mkdir output_final/'+str(ii))
        os.system('mkdir output_final/'+str(ii)+'/pos'+pos+'/ ;'+'mkdir output_final/'+str(ii)+'/pos'+pos+'/many ;'+'mkdir output_final/'+str(ii)+'/pos'+pos+'/spectra;'
                 +'mkdir output_final/'+str(ii)+'/pos'+pos+'/powers;'+'mkdir output_final/'+str(ii)+'/pos'+pos+'/casc_powers;'
                 +'mkdir output_final/'+str(ii)+'/pos'+pos+'/final_specs;')


        for i in inside_vec[int(ii)]:
            print(ii,i)
            CE = Conversion_efficiency(2,possition = pos, filename = 'data_large',filepath = which+'/output'+str(i)+'/data/')

            fmin,fmax,rounds  = 310,330,2000#np.min(CE.fv),np.max(CE.fv),None
            fmin,fmax,rounds = None, None, None
            fmin,fmax,rounds = np.min(CE.fv),np.max(CE.fv), None
            if animators:
                os.system('rm -rf animators'+str(i)+'; mkdir animators'+str(i))
                os.system('mkdir animators'+str(i)+'/contor animators'+str(i)+'/power animators'+str(i)+'/contor_single')
                #sys.exit()
                for iii in spots:
                    contor_plot_anim(CE,iii,fmin,fmax,rounds,filename= 'animators'+str(i)+'/contor/'+str(iii))
                    contor_plot_anim_single(CE,iii,fmin,fmax,rounds,filename= 'animators'+str(i)+'/contor_single/'+str(iii))
                    P_out_round_anim(CE,iii,filesave = 'animators'+str(i)+'/power/'+str(iii))
                    gc.collect()
                giff_it_up(i,spots,30)

            contor_plot(CE,fmin,fmax,rounds,filename= 'output_final/'+str(ii)+'/pos'+pos+'/spectra/'+str(ii)+'_'+str(i))
            #contor_plot_time(CE, rounds = None,filename = 'output_final/'+str(ii)+'/pos'+pos+'/'+'time_'+str(ii)+'_'+str(i))
            CE.P_out_round(filepath =  'output_final/'+str(ii)+'/pos'+pos+'/powers/', filesave =str(ii)+'_'+str(i))
            CE.P_out_round_casc(filepath =  'output_final/'+str(ii)+'/pos'+pos+'/casc_powers/',filesave = str(ii)+'_'+str(i))
            final_1D_spec(i,CE,filename = 'output_final/'+str(ii)+'/pos'+pos+'/final_specs/')
            #print(1e-3*c/CE.f_p, CE.lam_wanted)
            del CE
            gc.collect()

        #var1, var2 = 'P_bef', 'P_out'
        for var1,var2 in (('P_p', 'P_out'), ('P_p', 'CE')):
            plot_CE(var1, var2,filesave = 'output_final/'+str(ii)+'/pos'+pos+'/many/'+var2+str(ii))  
        plot_rin('P_p', 'rin',filesave = 'output_final/'+str(ii)+'/pos'+pos+'/many''/rin_'+str(ii))  
        
        
    os.system('rm -r prev_anim/*; mv animators* prev_anim')


# In[ ]:




