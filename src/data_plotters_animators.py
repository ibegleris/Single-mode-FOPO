import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.constants import c
import h5py
import sys
import warnings
warnings.simplefilter("ignore", UserWarning)
font = {'size': 18}

mpl.rc('font', **font)


def w2dbm(W, floor=-100):
    """This function converts a power given in W to a power given in dBm.
       Inputs::
               W(float): power in units of W
       Returns::
               Power in units of dBm(float)
    """
    if type(W) != np.ndarray:
        if W > 0:
            return 10. * np.log10(W) + 30
        elif W == 0:
            return floor
        else:
            print(W)
            raise(ZeroDivisionError)
    a = 10. * (np.ma.log10(W)).filled(floor/10-3) + 30
    return a


class Plotter_saver(object):

    def __init__(self, plots, filesaves, fv, t):
        if plots and filesaves:
            self.exporter = self.plotter_saver_both
        elif plots and not(filesaves):
            self.exporter = self.plotter_only
        elif not(plots) and filesaves:
            self.exporter = self.saver_only
        else:
            sys.exit("You are not exporting anything,\
    				  wasted calculation")
        #t = t[np.newaxis,3,:]
        self.fv, self.t,self.lv  = [self.reshape_x_axis(x) for x in (fv,t, 1e-3*c/fv)]

        return None
    def reshape_x_axis(self, x):
        return np.reshape(x, int(x.shape[0]*x.shape[1]))


    def initiate_reshape(self, u, U,nm):
        u, U = (np.reshape(i, [nm,int(u.shape[0]*u.shape[1])]) for i in (u, U))
        return u, U
    
    def plotter_saver_both(self, index, int_fwm, sim_wind, u, U, P0_p, P0_s,
                           f_p, f_s, ro, mode_names, pump_wave='',
                           filename=None, title=None, im=0, plots=True):
        u,U = self.initiate_reshape(u,U,int_fwm.nm)
        self.plotter(index, int_fwm, sim_wind, u, U, P0_p, P0_s,
                     f_p, f_s, ro, mode_names, pump_wave,
                     filename, title, im, plots)
        self.saver(index, int_fwm, sim_wind, u, U, P0_p, P0_s, f_p, f_s,
                    ro, mode_names, pump_wave, filename, title,
                   im, plots)
        return None

    def plotter_only(self, index, int_fwm, sim_wind, u, U, P0_p, P0_s,
                     f_p, f_s, ro, mode_names, pump_wave='',
                     filename=None, title=None, im=0, plots=True):
        u,U = self.initiate_reshape(u,U,int_fwm.nm)
        self.plotter(index, int_fwm, sim_wind, u, U, P0_p, P0_s,
                     f_p, f_s, ro, mode_names, pump_wave,
                     filename, title, im, plots)
        return None

    def saver_only(self, index, int_fwm, sim_wind, u, U, P0_p, P0_s,
                   f_p, f_s, ro, mode_names, pump_wave='',
                   filename=None, title=None, im=0, plots=True):
        u,U = self.initiate_reshape(u,U,int_fwm.nm)
        
        self.saver(index, int_fwm, sim_wind, u, U, P0_p, P0_s, f_p, f_s,
                    ro, mode_names, pump_wave, filename, title,
                   im, plots)
        return None

    def plotter(self, index, int_fwm, sim_wind, u, U, P0_p, P0_s,
                f_p, f_s, ro, mode_names, pump_wave='',
                filename=None, title=None, im=0, plots=True):
        """Plots many modes"""

        
        x, y = 1e-3*c/self.fv, w2dbm(np.abs(U)**2)
        xlim, ylim = [800, 1400], [-80, 100]
        xlabel, ylabel = r'$\lambda (nm)$', r'$Spectrum (a.u.)$'
        filesave = 'output'+pump_wave+'/output' + \
            str(index) + '/figures/wavelength/'+filename
        plot_multiple_modes(int_fwm.nm, x, y, mode_names,
                            ylim, xlim, xlabel, ylabel, title, filesave, im)

        # Frequency
        x, y = self.fv, w2dbm(sim_wind.dt[0]**2*np.abs(U)**2)# - np.max(w2dbm(sim_wind.dt[0]**2*np.abs(U)**2))
        xlim, ylim = [np.min(x), np.max(x)], [np.min(y) + 0.1*np.min(y), 1]
        xlim, ylim = [np.min(x), np.max(x)], [-50,100]
        
        xlabel, ylabel = r'$f (THz)$', r'$Spectrum (a.u.)$'
        filesave = 'output'+pump_wave+'/output' + \
            str(index) + '/figures/frequency/'+filename
        plot_multiple_modes(int_fwm.nm, x, y, mode_names,
                            ylim, xlim, xlabel, ylabel, title, filesave, im)

        # Time
        x, y = self.t, np.abs(u)**2
        xlim, ylim = [np.min(x), np.max(x)], [6.8, 7.8]
        xlabel, ylabel = r'$time (ps)$', r'$Spectrum (W)$'
        filesave = 'output'+pump_wave+'/output' + \
            str(index) + '/figures/time/'+filename
        plot_multiple_modes(int_fwm.nm, x, y, mode_names,
                            ylim, xlim, xlabel, ylabel, title, filesave, im)
        return None

    def saver(self, index, int_fwm, sim_wind, u, U, P0_p, P0_s, f_p, f_s
                , ro, mode_names, pump_wave='', filename=None, title=None,
              im=0, plots=True):
        """Dump to HDF5 for postproc"""

        if filename[:4] != 'port':
            layer = filename[-1]+'/'+filename[:-1]
        else:
            layer = filename
        if layer[0] is '0':
            extra_data = np.array([int_fwm.z, int_fwm.nm,P0_p, P0_s, f_p, f_s, ro])
            save_variables('data_large', layer, filepath='output'+pump_wave+'/output'+str(index)+'/data/', U=U, t=self.t,
                               fv=self.fv,  extra_data = extra_data)
        else:
            save_variables('data_large', layer, filepath='output'+pump_wave+'/output'+str(index)+'/data/', U=U)
            
        return None



def plot_multiple_modes(nm, x, y, mode_names, ylim, xlim, xlabel, ylabel, title, filesave=None, im=None):
    """
    Dynamically plots what is asked of it for multiple modes given at set point.
    """
    fig = plt.figure(figsize=(20.0, 10.0))
    plt.subplots_adjust(hspace=0.1)
    for i, v in enumerate(range(nm)):
        v = v+1
        ax1 = plt.subplot(nm, 1, v)
        plt.plot(x, y[i, :], '-', label=mode_names[i])
        ax1.legend(loc=2)
        ax1.set_ylim(ylim)
        ax1.set_xlim(xlim)
        if i != nm - 1:
            ax1.get_xaxis().set_visible(False)
    ax = fig.add_subplot(111, frameon=False)
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.set_title(title)
    plt.grid(True)
    ax.yaxis.set_label_coords(-0.05, 0.5)
    ax.xaxis.set_label_coords(0.5, -0.05)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if type(im) != int:
        newax = fig.add_axes([0.8, 0.8, 0.2, 0.2], anchor='NE')
        newax.imshow(im)
        newax.axis('off')
    if filesave == None:
        plt.show()
    else:
        plt.savefig(filesave, bbox_inched='tight')
    plt.close(fig)
    return None


def animator_pdf_maker(rounds, pump_index):
    """
    Creates the animation and pdf of the FOPO at different parts of the FOPO 
    using convert from imagemagic. Also removes the pngs so be carefull

    """
    print("making pdf's and animations.")
    space = ('wavelength', 'freequency', 'time')
    for sp in space:
        file_loc = 'output/output'+str(pump_index)+'/figures/'+sp+'/'
        strings_large = ['convert '+file_loc+'00.png ']
        for i in range(4):
            strings_large.append('convert ')
        for ro in range(rounds):
            for i in range(4):
                strings_large[i+1] += file_loc+str(ro)+str(i+1)+'.png '
            for w in range(1, 4):
                if i == 5:
                    break
                strings_large[0] += file_loc+str(ro)+str(w)+'.png '
        for i in range(4):
            os.system(strings_large[i]+file_loc+str(i)+'.pdf')

        file_loca = file_loc+'portA/'
        file_locb = file_loc+'portB/'
        string_porta = 'convert '
        string_portb = 'convert '
        for i in range(rounds):
            string_porta += file_loca + str(i) + '.png '
            string_portb += file_locb + str(i) + '.png '

        string_porta += file_loca+'porta.pdf '
        string_portb += file_locb+'portb.pdf '
        os.system(string_porta)
        os.system(string_portb)

        for i in range(4):
            os.system(
                'convert -delay 30 '+file_loc+str(i)+'.pdf '+file_loc+str(i)+'.mp4')
        os.system('convert -delay 30 ' + file_loca +
                  'porta.pdf ' + file_loca+'porta.mp4 ')
        os.system('convert -delay 30 ' + file_locb +
                  'portb.pdf ' + file_locb+'portb.mp4 ')

        for i in (file_loc, file_loca, file_locb):
            print('rm ' + i + '*.png')
            os.system('rm ' + i + '*.png')
        os.system('sleep 5')
    return None


def read_variables(filename, layer, filepath=''):
    with h5py.File(filepath+str(filename)+'.hdf5', 'r') as f:
        D = {}
        for i in f.get(layer).keys():
            try:
                D[str(i)] = f.get(layer + '/' + str(i)).value
            except AttributeError:
                pass
    return D


def save_variables(filename, layers, filepath='', **variables):
    with h5py.File(filepath + filename + '.hdf5', 'a') as f:
        for i in (variables):
            f.create_dataset(layers+'/'+str(i), data=variables[i])
    return None


def consolidate(max_rounds, int_fwm,master_index, index,  filename = 'data_large'):
    """
    Loads the HDF5 data and consolidates them for storage size
    reduction after the oscillations are done.
    """


    layer_0 = '0/0'
    filepath = 'output{}/output{}/data/'.format(master_index, index)
    file_read = filepath + filename
    file_save = filepath + filename+'_conc'
    
    # Input data, small, no need to cons
    D = read_variables(file_read, '0/0')
    save_variables(file_save, 'input', **D)

    if max_rounds ==0:
        max_rounds +=1
    U_cons = np.zeros([4,max_rounds, 7*int_fwm.nt], dtype = np.complex128)
    # Reading of all the oscillating spectra and sending them to a 3D array
    unfortmated_string = '{}/{}/U'
    with h5py.File(file_read+'.hdf5', 'r') as f:
        for pop in range(1,5):
            for r in range(max_rounds):
                U_cons[pop - 1,r,:] = f.get(unfortmated_string.format(pop,r)).value
    save_variables(file_save, 'results', U = U_cons)            
    os.system('mv '+file_save+'.hdf5 '+file_read+'.hdf5')
    return None