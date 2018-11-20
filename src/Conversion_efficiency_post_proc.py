import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
import pickle as pl
import tables
import h5py
from scipy.constants import c, pi
import gc
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from data_plotters_animators import read_variables
from functions import *
import warnings 
warnings.filterwarnings('ignore')
import tables
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from numpy.fft import fftshift
import scipy
from os import listdir

font = {'size'   : 16}
matplotlib.rc('font'
              , **font)


def selmier(l):
    a = 0.6961663*l**2/(l**2 - 0.0684043**2)
    b = 0.4079426*l**2/(l**2 - 0.1162414**2)
    c = 0.8974794*l**2/(l**2 - 9.896161**2)
    return (1 + a + b +c)**0.5


class Conversion_efficiency(object):
    def __init__(self, freq_band, last, safety, possition, filename=None, filepath='',filename2 = 'CE',filepath2 = 'output_final/'):
        self.mode_names = ('LP01x', 'LP01y')
        self.n = 1.444

        
        self.variables = ('P_p', 'P_s', 'f_p', 'f_s','l_p','l_s,' 'P_out', 'P_bef','CE', 'CE_std', 'P_out_std','rin')
        self.spec,init_phase, self.fv, self.t, self.P0_p,\
        self.P0_s, self.f_p, self.f_s, self.ro, self.L = \
                    self.load_spectrum_initial(filename, filepath)
        self.pos_of_pump()

        self.P_max = np.array([np.max(i) for i in w2dbm(self.spec)])

        self.spec, self.P_bef, U_large, Uphase = \
                   self.load_spectrum_oscillating(possition,self.fv, self.t,filename, filepath)

        self.phase_diff = Uphase - init_phase#np.asarray([Uphase[i,:] - init_phase for i in range(Uphase.shape[0])])

        if type(last) is float:
            self.last = int(last* U_large.shape[0])
        else:
            self.last = last
        
        self.safety = int(len(self.fv)/(7*2))
        
        self.tt = self.t
        self.freq_band = freq_band
        self.U_large = np.asanyarray(U_large)
        
        
        self.nt = np.shape(self.spec)
        self.possition = possition

        
        self.U_av = np.average(self.U_large[self.last:,:,:],axis = 0)
        self.P_in = self.P0_p + self.P0_s
        self.pos_of_signal()
        self.pos_of_idler()
        self.pos_of_cascades()

        if possition == '2':
            fv_id = self.fs_id
            fv_id_c = self.fv_id_c[0]
        elif possition == '1':
            fv_id = self.fs_id
            fv_id_c = self.fv_id_c[0]
        else:
            fv_id = self.fi_id
            fv_id_c = self.fv_id_c[1]
        self.fv_id_c = fv_id_c        
        self.lam_wanted = 1e-3*c/self.fv[fv_id]
        self.lamp = 1e-3*c/self.f_p
        self.l_s = 1e-3*c/self.f_s

        #sys.exit()
        self.U_large_norm = np.empty_like(U_large)

        self.n = selmier(1e-3*self.lam_wanted)
        self.time_trip = self.L*self.n/c
        
        #print(self.P_max)
        for i,P_max in enumerate(self.P_max):
            self.U_large_norm[:,i,:] =\
                    w2dbm(self.U_large[:,i,:]) - P_max

        P_out_vec = []
        P_out_vec_casc = []
        start, end = self.fv[fv_id] - freq_band, self.fv[fv_id] + freq_band

        start_end_c = [[np.argmin(np.abs(self.fv - self.fv[fv_id_c[i]] + freq_band)),
                         np.argmin(np.abs(self.fv - self.fv[fv_id_c[i]] - freq_band))] for i in range(2)]
        
        start_i = [np.argmin(np.abs(self.fv - i)) for i in start]
        end_i = [np.argmin(np.abs(self.fv - i)) for i in end]

        

        Uabs_large = U_large
        for i in Uabs_large:
            self.spec = i
            P_out_vec.append(self.calc_P_out(start_i,end_i))
            P_out_vec_casc.append([self.calc_P_out([start_end[0]],[start_end[1]])
                                for start_end in start_end_c])
        print([self.fv[start_i], self.fv[end_i]])

        self.P_out_vec = np.asanyarray(P_out_vec)
        self.P_out_vec_casc = np.reshape(np.asanyarray(P_out_vec_casc), (2, self.P_out_vec.shape[0],1))

        #for l, la in enumerate(last):
        D_now = {}

        D_now['P_out'] = np.mean(self.P_out_vec[-self.last::,:], axis = 0)
        D_now['CE'] = 100*D_now['P_out']/ (self.P0_p + self.P0_s)
        D_now['P_out_std'] = np.std(self.P_out_vec[-self.last::,:], axis = 0)

        D_now['CE_std'] = np.std(self.P_out_vec[-self.last::,:] / (self.P0_p + self.P0_s), axis = 0)

        D_now['rin'] = 10*np.log10(self.time_trip*D_now['P_out_std']**2 / D_now['P_out']**2)
        D_now['P_p'], D_now['P_s'], D_now['f_p'], D_now['f_s'],\
            D_now['l_p'], D_now['l_s'], D_now['P_bef'] =\
            self.P0_p, self.P0_s, self.f_p, self.f_s, self.lamp, self.l_s, self.P_bef

        for i,j in zip(D_now.keys(), D_now.values()):
            D_now[i] = [j]
        
        if os.path.isfile(filepath2+filename2+'.pickle'):
            with open(filepath2+filename2+'.pickle','rb') as f:
                D = pl.load(f)
            for i,j in zip(D.keys(), D.values()):
                D[i] = j + D_now[i]
        else:
            D = D_now
        with open(filepath2+filename2+'.pickle','wb') as f:
            pl.dump(D,f)
        self.spec = U_large[-1,:,:]
        
        self.spec_s = np.empty_like(self.spec)
        for i in range(len(self.P_max)):
           self.spec_s[i,:] = w2dbm(self.spec[i,:]) - self.P_max[i]

        return None

    def pos_of_pump(self):
        self.fp_id = np.array([np.argmax(self.spec[i,:]) -1 \
                         for i in range(self.spec.shape[0])])
        #print('pump',self.fv[self.fp_id],1e-3*c/self.fv[self.fp_id])
        return None
    
    def pos_of_idler(self):
        plom =  self.fp_id + self.safety
        #print(self.fp_id )
        self.fi_id = np.array([np.argmax(self.U_av[i,plom[i]:])\
                         for i in range(self.U_av.shape[0])])
        self.fi_id += plom -1
        return None   
    
    def pos_of_signal(self):
        plom =  self.fp_id - self.safety
        self.fs_id = np.array([np.argmax(self.U_av[i,:plom[i]])\
                         for i in range(self.U_av.shape[0])])
        return None   
    

    def pos_of_cascades(self):
        plom = self.fs_id -self.safety
        self.fv_id_c = []
        self.fv_id_c.append([np.argmax(self.U_av[i,:plom[i]])\
                         for i in range(self.U_av.shape[0])])
        plom = self.fi_id + self.safety
        self.fv_id_c = [[self.fs_id - (self.fp_id - self.fs_id),
                        self.fs_id - 2 * (self.fp_id - self.fs_id) ],
                        [self.fi_id + (self.fi_id - self.fp_id),
                        self.fi_id + 2 * (self.fi_id - self.fp_id) ]]
        no_cacs = False
        for j in range(2):
            for i in range(2):
                if self.fv_id_c[j][i] >= self.fv.shape[-1]:
                    print(self.fv_id_c[j][i], self.fv.shape[-1])
                    self.fv_id_c = [[self.fs_id,self.fs_id],
                                    [self.fi_id,self.fi_id]]
                    break
            if no_cacs:
                break
        return None
    
    
    def load_spectrum_initial(self,filename='data_large', filepath=''):
        D = read_variables(filepath+filename, 'input')
        L, nm, P0_p, P0_s, f_p, f_s, ro= D['extra_data']
        fv, t = D['fv'], D['t']
        Uabs = np.abs(D['U'])**2
        Uphase = np.angle(D['U'])
        Uabs = Uabs[np.newaxis,:,:]
        return Uabs, Uphase,fv, t, P0_p, P0_s, f_p, f_s, ro, L


    def load_spectrum_oscillating(self, possition,fv, t, filename='data_large', filepath=''):
        pos = int(possition) - 1

        U_total = read_variables(filepath+filename, 'results')['U']

        U_large = np.abs(U_total[pos,:,:])**2
        U_phase = np.angle(U_total[pos,:,:])
        U_large = U_large.reshape([U_large.shape[0], 1, U_large.shape[1]])
        Uabs = np.abs(U_total[pos,-1,:])**2
        Uabss =np.abs(U_total[0,0,:]*(t[1] - t[0]))**2
        P_bef = simps(Uabss,fv)/(np.max(t))
        return Uabs, P_bef, U_large, U_phase

    
    def calc_P_out(self,start,end):
        P_out = []
        for i,j,sp in zip(start,end,self.spec):
            P_out.append(simps(sp[i:j]*(self.tt[1] - self.tt[0])**2,\
                         self.fv[i:j])/(2*np.abs(np.min(self.tt))))
        return P_out   

    
    def P_out_round(self,P,filepath,filesave):
        """Plots the output average power with respect to round trip number"""
        string_title = "$P_p=$ {:.2f}W, $P_s=$ {:.2f}mW, $\\lambda_p=$ {:.2f}nm"
        string_title += " $\\lambda_s=$ {:.2f}nm, maximum output at: {:.2f}nm"
        string_title += "({:.2f}Thz),\n"
        string_title += "cascades at: {:.2f}Thz, {:.2f}Thz"
 
        
        x = range(len(P))
        y = np.asanyarray(P)
        fig = plt.figure(figsize=(20.0, 10.0))
        plt.subplots_adjust(hspace=0.1)
        for i, v in enumerate(range(y.shape[1])):
            v = v+1
            ax1 = plt.subplot(y.shape[1], 1, v)
            plt.plot(x, y[:,i], '-', label = self.mode_names[i])
            #plt.xlim(0,1000)
            ax1.legend(loc=2)
            if i != y.shape[1] - 1:
                ax1.get_xaxis().set_visible(False)
        ax = fig.add_subplot(111, frameon=False)
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        

        ax.set_title(string_title.format(CE.P0_p, CE.P0_s*1e3, CE.lamp, CE.l_s,\
                             CE.lam_wanted[i], 1e-3*c/CE.lam_wanted[i], \
                             CE.fv[CE.fv_id_c[0]][0], CE.fv[CE.fv_id_c[1]][0]))
        plt.grid()
        ax.yaxis.set_label_coords(-0.05, 0.5)
        ax.xaxis.set_label_coords(0.5, -0.05)
        ax.set_xlabel('Rounds')
        ax.set_ylabel('Output Power (W)')
        plt.savefig(filepath+'power_per_round'+filesave+'.png')
        data = (range(len(P)), P)
        _data ={'pump_power':self.P0_p, 'pump_wavelength': self.lamp, 'out_wave': self.lam_wanted}
        with open(filepath+'power_per_round'+filesave+'.pickle','wb') as f:
            pl.dump((data,_data),f)
        plt.clf()
        plt.close('all')
        return None


    def final_1D_spec(self,filename,wavelengths = None):
        x,y = self.fv, self.spec_s
        fig = plt.figure(figsize=(20.0, 10.0))
        for i, v in enumerate(range(y.shape[0])):
            v = v+1
            ax1 = plt.subplot(y.shape[0], 1, v)
            plt.plot(x,y[i,:], '-', label = self.mode_names[i])
            ax1.legend(loc=2)
            if i is 0:
                axl = ax1.twiny()
                axl.set_xlim(ax1.get_xlim())
                if wavelengths is None:
                    new_tick_locations = ax1.get_xticks()
                    axl.set_xticks(new_tick_locations)
                    axl.set_xticklabels(tick_function(new_tick_locations))
                else:
                    new_tick_locations = [1e-3*c/i for i in wavelengths]
                    axl.set_xticks(new_tick_locations)
                    axl.set_xticklabels(wavelengths)
                axl.set_xlabel(r"$\lambda (nm)$")
            if i != y.shape[0] - 1:
                ax1.get_xaxis().set_visible(False)
        ax = fig.add_subplot(111, frameon=False)
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        #ax.set_title(f"$P_p=$ {float(CE.P0_p):.{6}} W, $P_s=$ {float(CE.P0_s*1e3):.{2}} mW, $\\lambda_p=$ {float(CE.lamp):.{6}} nm,  $\\lambda_s=$ {float(CE.l_s):.{6}} nm, maximum output at: {float(CE.lam_wanted[i]):.{6}} nm ({float(1e-3*c/CE.lam_wanted[i]):.6} Thz)")
        plt.grid()
        ax.yaxis.set_label_coords(-0.05, 0.5)
        ax.xaxis.set_label_coords(0.5, -0.05)
        ax.set_xlabel(r'$f (THz)$')
        ax.set_ylabel(r'Spec (dB)')
        plt.savefig(filename+'.png', bbox_inches = 'tight')
        

        data = (x, y)
        _data ={'pump_power':self.P0_p, 'pump_wavelength': self.lamp, 'out_wave': self.lam_wanted}
        with open(filename+str(ii)+'.pickle','wb') as f:
            pl.dump((data,_data),f)
        plt.clf()
        plt.close('all')
        return None


    def plot_phase_diff(self,folder, filename):

        x =  range(np.shape(self.U_large_norm)[0])
        X, Y = np.meshgrid(x, self.fv)
        fig = plt.figure(figsize=(20,10))
        plt.contourf(X,Y, self.phase_diff.T)
        plt.xlabel('Rounds')
        plt.ylabel('Frequency (Thz)')
        plt.colorbar()
        plt.savefig(folder + filename +'.png')
        plt.close()
        return None

    def plot_phase_diff_specific(self,folder, filename):

        x =  range(np.shape(self.U_large_norm)[0])
        y = [self.phase_diff[:, i] for i in (self.fs_id, self.fp_id)] 
        label_vec = ['signal', 'pump', 'idler']
        fig,ax = plt.subplots(2,1, figsize=(20,10), sharex = True)
        for i, yy in enumerate(y):
            ax[i].plot(x,yy, label = label_vec[i])
        ax[1].set_xlabel('Rounds')
        ax[0].set_ylabel('Phase (rad)')
        
        ax[1].set_ylabel('Phase (rad)')
        

        ax[0].legend()
        ax[1].legend()
        plt.savefig(folder + filename +'.png')
        plt.close()
        return None


def plot_rin(var,var2 = 'rin',filename = 'CE', filepath='output_final/', filesave= None):
    var_val, CE,std = read_CE_table(filename,var,var2 = var2,file_path=filepath)
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

def read_CE_table(x_key,y_key ,filename, std = False):
    with open(filename+'.pickle','rb') as f:
        D = pl.load(f)
    x = D[x_key]
    y = D[y_key]

    if std:
        try:

            err_bars = D[y_key+'_std']
        except KeyError:
            sys.exit('There is not error bar for the variable you are asking for.')
    else:
        err_bars = 0
    x, y, err_bars = np.asanyarray(x),np.asanyarray(y), np.asanyarray(err_bars)
    
    return x,y,err_bars


def plot_CE(x_key,y_key,std,filename = 'CE', filepath='output_final/', filesave= None):
    x, y, err_bars = read_CE_table(x_key,y_key,filepath+filename,std = std)
    fig = plt.figure(figsize=(20.0, 10.0))
    plt.subplots_adjust(hspace=0.1)
    mode_labels = ('LP01x','LP01y')


    for i, v in enumerate(range(y.shape[1])):
        v = v+1
        ax1 = plt.subplot(y.shape[1], 1, v)
        if std:
            ax1.errorbar(x,y[:,i], yerr=err_bars[:,i], capsize= 10, label = mode_labels[i])
        else:
            ax1.plot(x,y[:,i], label = mode_labels[i])
        ax1.legend(loc=2)
        if i != y.shape[1] - 1:
            ax1.get_xaxis().set_visible(False)
    ax = fig.add_subplot(111, frameon=False)
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    plt.grid()
    ax.yaxis.set_label_coords(-0.05, 0.5)
    ax.xaxis.set_label_coords(0.5, -0.05)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    plt.savefig(filesave+'.png',bbox_inches = 'tight')
    data = (x, y,err_bars)
    with open(str(filesave)+'.pickle','wb') as f:
        pl.dump((fig,data),f)
    plt.clf()
    plt.close('all')
    return None



def contor_plot(CE,fmin = None,fmax = None,  rounds = None,folder = None,filename = None):
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
    z = CE.U_large_norm[:rounds,:,i:j]

    #low_values_indices = z < -60  # Where values are low
    #z[low_values_indices] = -60  # All low values set to 0
    for nm in range(z.shape[1]):
        fig = plt.figure(figsize=(20,10))
        plt.contourf(x,y, z[:,nm,:].T, np.arange(-60,2,2),extend = 'min',cmap=plt.cm.jet)
        plt.xlabel(r'$rounds$')
        plt.ylim(fmin,fmax)
        plt.ylabel(r'$f(THz)$')
        plt.colorbar()
        plt.title(f"$P_p=$ {float(CE.P0_p):.{2}} W, $P_s=$ {float(CE.P0_s*1e3):.{2}} mW, $\\lambda_p=$ {float(CE.lamp):.{6}} nm,  $\\lambda_s=$ {float(CE.l_s):.{6}} nm, maximum output at: {float(CE.lam_wanted[nm]):.{6}} nm")
        data = (CE.ro, CE.fv, z)
        _data ={'pump_power':CE.P0_p, 'pump_wavelength': CE.lamp, 'out_wave': CE.lam_wanted}
        if filename is not None:
            plt.savefig(folder+str(nm)+'_'+filename, bbox_inches = 'tight')
            plt.clf()
            plt.close('all')
        else:
            plt.show()

    if filename is not None:
        with open(str(folder+filename)+'.pickle','wb') as f:
            pl.dump((data,_data),f)
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


def tick_function(X):
    l = 1e-3*c/X
    return ["%.2f" % z for z in l]


#from os.path import , join
data_dump =  'output_dump'
outside_dirs = [f for f in listdir(data_dump)]
inside_dirs = [[f for f in listdir(data_dump+ '/'+out_dir)] for out_dir in outside_dirs ]


which = 'output_dump_pump_wavelengths/7w'
which = 'output_dump_pump_wavelengths/wrong'
which = 'output_dump_pump_wavelengths'
#which = 'output_dump_pump_wavelengths/2_rounds'
#which ='output_dump_pump_powers/ram0ss0'
#which = 'output_dump/'#_pump_powers'
which_l = 'output_dump/output'



outside_vec = range(len(outside_dirs))
outside_dirs.sort()
inside_vec = [range(len(inside) - 1) for inside in inside_dirs]
#inside_vec = [13]
animators = False
spots = range(0,8100,100)
wavelengths = [1200,1400,1050,930,800]
#wavelengths = None


os.system('rm -r output_final ; mkdir output_final')
for pos in ('4','2', '1'):

    for ii in outside_vec:
        ii = str(ii)
        which = which_l+ ii
        
        os.system('mkdir output_final/'+str(ii))
        os.system('mkdir output_final/'+str(ii)+'/pos'+pos+'/ ;'
                 + 'mkdir output_final/'+str(ii)+'/pos'+pos+'/many ;'
                 + 'mkdir output_final/'+str(ii)+'/pos'+pos+'/spectra;'
                 + 'mkdir output_final/'+str(ii)+'/pos'+pos+'/powers;'
                 + 'mkdir output_final/'+str(ii)+'/pos'+pos+'/casc_powers;'
                 + 'mkdir output_final/'+str(ii)+'/pos'+pos+'/casc_powers/0;'
                 + 'mkdir output_final/'+str(ii)+'/pos'+pos+'/casc_powers/1;'
                 + 'mkdir output_final/'+str(ii)+'/pos'+pos+'/final_specs;'
                 + 'mkdir output_final/'+str(ii)+'/pos'+pos+'/contor_phases;'
                 + 'mkdir output_final/'+str(ii)+'/pos'+pos+'/wave_phases;')


        for i in inside_vec[int(ii)]:
            print(ii,i)
            CE = Conversion_efficiency(freq_band = 10,possition = pos,last = 0.5,\
                safety = 1, filename = 'data_large',\
                filepath = which+'/output'+str(i)+'/data/',filepath2 = 'output_final/'+str(ii)+'/pos'+str(pos)+'/')

            fmin,fmax,rounds  = np.min(CE.fv),np.max(CE.fv),None
            #fmin,fmax,rounds = 180,400, None
            #fmin,fmax,rounds = np.min(CE.fv),np.max(CE.fv), None
            #if animators:
            #    os.system('rm -rf animators'+str(i)+'; mkdir animators'+str(i))
            #    os.system('mkdir animators'+str(i)+'/contor animators'+str(i)+'/power animators'+str(i)+'/contor_single')
  
            #    for iii in spots:
            #        contor_plot_anim(CE,iii,fmin,fmax,rounds,filename= 'animators'+str(i)+'/contor/'+str(iii))
            #        contor_plot_anim_single(CE,iii,fmin,fmax,rounds,filename= 'animators'+str(i)+'/contor_single/'+str(iii))
            #        P_out_round_anim(CE,iii,filesave = 'animators'+str(i)+'/power/'+str(iii))
            #        gc.collect()
            #    giff_it_up(i,spots,30)

            #contor_plot(CE,fmin,fmax,rounds,folder = 'output_final/'+str(ii)+'/pos'+pos+'/spectra/',filename= str(ii)+'_'+str(i))
            #CE.plot_phase_diff('output_final/'+str(ii)+'/pos'+pos+'/contor_phases/', filename= str(ii)+'_'+str(i))
            #CE.plot_phase_diff_specific('output_final/'+str(ii)+'/pos'+pos+'/wave_phases/', filename= str(ii)+'_'+str(i))
            #contor_plot_time(CE, rounds = None,filename = 'output_final/'+str(ii)+'/pos'+pos+'/'+'time_'+str(ii)+'_'+str(i))
            CE.P_out_round(CE.P_out_vec,filepath =  'output_final/'+str(ii)+'/pos'+pos+'/powers/', filesave =str(ii)+'_'+str(i))
            CE.P_out_round(CE.P_out_vec_casc[0],filepath =  'output_final/'+str(ii)+'/pos'+pos+'/casc_powers/0/',filesave = str(ii)+'_'+str(i))
            CE.P_out_round(CE.P_out_vec_casc[1],filepath =  'output_final/'+str(ii)+'/pos'+pos+'/casc_powers/1/',filesave = str(ii)+'_'+str(i))
            
            CE.final_1D_spec(filename = 'output_final/'+str(ii)+'/pos'+pos+'/final_specs/'+'spectrum_fopo_final'+str(ii)+'_'+str(i),wavelengths = wavelengths)
            del CE
            gc.collect()
        for x_key,y_key,std in (('P_p', 'P_out',True), ('P_p', 'CE',True), ('P_p', 'rin',False)):
            plot_CE(x_key,y_key,std,filename = 'CE',\
                filepath='output_final/'+str(ii)+'/pos'+pos+'/', filesave = 'output_final/'+str(ii)+'/pos'+pos+'/many/'+y_key+str(ii))
        
    #os.system('rm -r prev_anim/*; mv animators* prev_anim')




"""
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
    self.lamp = 1e-3*c/CE.f_p
    plt.title(f"$P_p=$ {float(CE.P0_p):.{2}} W, $P_s=$ {float(CE.P0_s*1e3):.{2}} mW, $\\lambda_p=$ {float(CE.lamp):.{6}} nm,  $\\lambda_s=$ {float(CE.l_s):.{6}} nm, maximum output at: {float(CE.lam_wanted):.{6}} nm")
    data = (CE.ro, CE.fv, z )
    _data ={'pump_power':CE.P0_p, 'pump_wavelength': self.lamp, 'out_wave': CE.lam_wanted}
    if filename is not None:
        plt.savefig(str(filename), bbox_inches = 'tight')
        plt.clf()
        plt.close('all')
        #with open(str(filename)+'.pickle','wb') as f:
        #    pl.dump((data,_data),f)


    else:
        plt.show()
    return None


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

"""
