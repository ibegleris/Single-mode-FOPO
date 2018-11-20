import numpy as np
import sys
import os
import sys
sys.path.append('src')
from scipy.constants import c, pi
from joblib import Parallel, delayed
from mpi4py.futures import MPIPoolExecutor
from mpi4py import MPI
from scipy.fftpack import fftshift, fft
import os
import time as timeit
os.system('export FONTCONFIG_PATH=/etc/fonts')

from functions import *

from time import time, sleep

import pickle
@profile
def oscilate(sim_wind, int_fwm, noise_obj, TFWHM_p, TFWHM_s, index, master_index, P0_p1, P0_s, f_p, f_s, p_pos, s_pos, splicers_vec,
             WDM_vec, Dop, dAdzmm, D_pic, pulse_pos_dict_or, plots, ex, pm_fopa, pm_WDM1, fopa):
    mode_names = ['LP01a']
    u = np.zeros(sim_wind.t.shape, dtype='complex128')
    U = np.zeros(sim_wind.fv.shape, dtype='complex128')    #

    T0_p = TFWHM_p / 2 / (np.log(2))**0.5
    T0_s = TFWHM_s / 2 / (np.log(2))**0.5
    noise_new = noise_obj.noise_func(int_fwm)
    u = noise_new

    woff1 = (p_pos[1] + (int_fwm.nt) // 2) * 2 * pi * sim_wind.df[p_pos[0]]
    u[p_pos[0], :] += (P0_p1)**0.5 * np.exp(1j *
                                            (woff1) * sim_wind.t[p_pos[0]])

    woff2 = -(s_pos[1] - (int_fwm.nt - 1) // 2) * \
        2 * pi * sim_wind.df[s_pos[0]]

    u[s_pos[0], :] += (P0_s)**0.5 * np.exp(-1j *
                                           (woff2) * sim_wind.t[s_pos[0]])

    U = fftshift(fft(u), axes=-1)

    master_index = str(master_index)
    max_rounds = arguments_determine(-1)
    if fopa:
        print('Fibre amplifier!')
        max_rounds = 0

    ex.exporter(index, int_fwm, sim_wind, u, U, P0_p1,
                P0_s, f_p, f_s, max_rounds,  mode_names, master_index, '00', 'original pump', D_pic[0], plots)

    U_original_pump = np.copy(U)

    # Pass the original pump through the WDM1, port1 is in to the loop, port2
    noise_new = noise_obj.noise_func_freq(int_fwm, sim_wind)
    u, U = WDM_vec[0].pass_through((U, noise_new))[0]



    ro = -1
    t_total = 0
    factors_xpm, factors_fwm,gama,tsh, w_tiled = \
                dAdzmm.factors_xpm, dAdzmm.factors_fwm, dAdzmm.gama, dAdzmm.tsh, dAdzmm.w_tiled
    dz,dzstep,maxerr = int_fwm.dz,int_fwm.dzstep,int_fwm.maxerr
    Dop = np.ascontiguousarray(Dop/2)
    factors_xpm = np.ascontiguousarray(factors_xpm)
    factors_fwm = np.ascontiguousarray(factors_fwm)
    gama = np.ascontiguousarray(gama)
    tsh = np.ascontiguousarray(tsh)
    w_tiled = np.ascontiguousarray(w_tiled)
    while ro < max_rounds:

        ro += 1
        print('round', ro)
        pulse_pos_dict = [
            'round ' + str(ro) + ', ' + i for i in pulse_pos_dict_or]

        ex.exporter(index, int_fwm, sim_wind, u, U, P0_p1,
                    P0_s, f_p, f_s, ro,  mode_names, master_index, str(ro) + '1', pulse_pos_dict[3], D_pic[5], plots)
        # Phase modulate before the Fibre
        U = pm_fopa.modulate(U)
        u = ifft(ifftshift(U, axes=-1))

        #Pulse propagation
        U, dz = pulse_propagation(u,dz,dzstep,maxerr, Dop,factors_xpm, factors_fwm, gama,tsh,w_tiled)

        ex.exporter(index, int_fwm, sim_wind, u, U, P0_p1,
                    P0_s, f_p, f_s, ro, mode_names, master_index, str(ro) + '2', pulse_pos_dict[0], D_pic[2], plots)

        max_noise = 10*noise_new.max()
        #checks if the fft's are causing boundary condtion problems 
        if (U[:, 0] > max_noise).any() or (U[:, -1] > max_noise).any():
            with open("error_log", "a") as myfile:
                myfile.write("Pump: %5f, Seed: %5f, lamp: %5f, lams: %5f \n" % (
                    P0_p1, P0_s, 1e-3*c/f_p, 1e-3*c/f_s))
            break
            
        # pass through WDM2 port 2 continues and port 1 is out of the loop
        noise_new = noise_obj.noise_func_freq(int_fwm, sim_wind)
        (out1, out2), (u, U) = WDM_vec[1].pass_through(
            (U, noise_new))

        ex.exporter(index, int_fwm, sim_wind, u, U, P0_p1,
                    P0_s, f_p, f_s, ro,  mode_names, master_index, str(ro) + '3', pulse_pos_dict[3], D_pic[3], plots)

        # Splice7 after WDM2 for the signal
        noise_new = noise_obj.noise_func_freq(int_fwm, sim_wind)
        (u, U) = splicers_vec[2].pass_through(
            (U, noise_new))[0]
       

        #Phase modulate the oscillating signal so that to be in phase with the one coming in
        U = pm_WDM1.modulate(U_original_pump, U)
        # Pass again through WDM1 with the signal now
        (u, U) = WDM_vec[0].pass_through(
            (U_original_pump, U))[0]

        ################################The outbound stuff#####################
        ex.exporter(index, int_fwm, sim_wind, out1, out2, P0_p1,
                    P0_s, f_p, f_s, ro,  mode_names, master_index, str(ro) + '4', pulse_pos_dict[4], D_pic[6], plots)
    consolidate(ro, int_fwm,master_index, index)
    return ro


def calc_P_out(U, U_original_pump, fv, t):
    U = np.abs(U)**2
    U_original_pump = np.abs(U_original_pump)**2
    freq_band = 2
    fp_id = np.where(U_original_pump == np.max(U_original_pump))[0][0]
    plom = fp_id + 10
    fv_id = np.where(U[plom:] == np.max(U[plom:]))[0][0]
    fv_id += plom - 1
    start, end = fv[fv_id] - freq_band, fv[fv_id] + freq_band
    i = np.where(
        np.abs(fv - start) == np.min(np.abs(fv - start)))[0][0]
    j = np.where(
        np.abs(fv - end) == np.min(np.abs(fv - end)))[0][0]
    E_out = simps(U[i:j] * (t[1] - t[0])**2, fv[i:j])
    P_out = E_out / (2 * np.abs(np.min(t)))
    return P_out


@unpack_args
def formulate(index, n2, gama, alphadB, z, P_p, P_s, TFWHM_p, TFWHM_s, spl_losses, betas,
              lamda_c, WDMS_pars, lamp, lams, num_cores, maxerr, ss, plots,
              N, nplot, master_index, filesaves, Df_band, fr, fopa):
    "------------------propagation paramaters------------------"
    dzstep = z / nplot                        # distance per step
    dz_less = 1e2
    int_fwm = sim_parameters(n2, 1, alphadB)
    int_fwm.general_options(maxerr, ss)
    int_fwm.propagation_parameters(N, z, nplot, dz_less)
    lamda = lamp * 1e-9  # central wavelength of the grid[m]

    "-----------------------------f-----------------------------"

    "---------------------Aeff-Qmatrixes-----------------------"
    M = Q_matrixes(int_fwm.nm, int_fwm.n2, lamda_c, gama)
    "----------------------------------------------------------"

    "---------------------Grid&window-----------------------"

    P_p_bef,P_s_bef = pre_fibre_init_power(WDMS_pars[0][0], WDMS_pars[0][1], lamp, P_p, P_s)

    fv, where, f_centrals = fv_creator(
        lamp, lams, lamda_c, int_fwm, betas, M, P_p_bef,P_s_bef, Df_band)
    print(fv[0][1] - fv[0][0])
    #print(1e-3 * c / np.array(f_centrals))
    p_pos, s_pos, i_pos = where
    sim_wind = sim_window(fv, lamda, f_centrals, lamda_c, int_fwm)
    "----------------------------------------------------------"

    "---------------------Loss-in-fibres-----------------------"
    slice_from_edge = (sim_wind.fv[-1] - sim_wind.fv[0]) / 100
    loss = Loss(int_fwm, sim_wind, amax=0)

    int_fwm.alpha = loss.atten_func_full(fv)
    int_fwm.gama = np.array(
        [-1j * n2 * 2 * M * pi * (1e12 * f_c) / (c) for f_c in f_centrals])
    #if ss == 0:
    #    int_fwm.gama[:] = -1j * n2 * 2 * M * pi * (1e12 * f_centrals[3]) / (c)
    int_fwm.gama[0:2] = 0 
    int_fwm.gama[5:] = 0 
    #for i in range(len(int_fwm.gama)):
    #    print(i, int_fwm.gama[i])

    #exit()
    "----------------------------------------------------------"
    "--------------------Dispersion----------------------------"
    Dop = dispersion_operator(betas, lamda_c, int_fwm, sim_wind)
    "----------------------------------------------------------"
    "---------------------Raman Factors------------------------"
    ram = Raman_factors(fr)
    ram.set_raman_band(sim_wind)
    "----------------------------------------------------------"

    "--------------------Noise---------------------------------"
    noise_obj = Noise(int_fwm, sim_wind)
    "----------------------------------------------------------"

    pulse_pos_dict_or = ('after propagation', "pass WDM2",
                         "pass WDM1 on port2 (remove pump)",
                         'add more pump', 'out')

    keys = ['loading_data/green_dot_fopo/pngs/' +
            str(i) + str('.png') for i in range(7)]
    D_pic = [plt.imread(i) for i in keys]

    "----------------Construct the integrator----------------"
    non_integrand = Integrand(int_fwm.gama, sim_wind.tsh,
                              sim_wind.w_tiled, ss,ram, cython_tick=True,
                              timer=False)
    "--------------------------------------------------------"

    "----------------------Formulate WDMS--------------------"
    if WDMS_pars == 'signal_locked':

        Omega = 2 * pi * c / (lamp * 1e-9) - 2 * pi * c / (lams * 1e-9)
        omegai = 2 * pi * c / (lamp * 1e-9) + Omega
        lami = 1e9 * 2 * pi * c / (omegai)
        WDMS_pars = ([lamp, lams],     # WDM up downs in wavelengths [m]
                     [lami, lams],
                     [lami, lamp],
                     [lami, lams])

    WDM_vec = [WDM(i[0], i[1], sim_wind.fv, c,fopa)
               for i in WDMS_pars]  # WDM up downs in wavelengths [m]
    # Phase modulators contructors
    pm_fopa = Phase_modulation_FOPA(sim_wind.fv, where)
    pm_WDM1 = Phase_modulation_infase_WDM(P_s, where, WDM_vec[0])
    "--------------------------------------------------------"
    # for ei,i in enumerate(WDM_vec):
    #    i.plot(filename = str(ei))
    "----------------------Formulate splicers--------------------"
    splicers_vec = [Splicer(loss=i) for i in spl_losses]
    "------------------------------------------------------------"

    f_p, f_s = sim_wind.fv[where[0][0], where[0][1]], sim_wind.fv[where[1][0], where[1][1]] 

    ex = Plotter_saver(plots, filesaves, sim_wind.fv,
                       sim_wind.t)  # construct exporter
    ro = oscilate(sim_wind, int_fwm, noise_obj, TFWHM_p, TFWHM_s, index, master_index, P_p, P_s, f_p, f_s, p_pos, s_pos, splicers_vec,
                  WDM_vec, Dop, non_integrand, D_pic, pulse_pos_dict_or, plots, ex, pm_fopa, pm_WDM1,fopa)

    return None


def main():
    "-----------------------------Stable parameters----------------------------"
    # Number of computing cores for sweep
    num_cores = arguments_determine(1)
    # maximum tolerable error per step in integration
    maxerr = 1e-13
    ss = 1                                      # includes self steepening term
    Df_band_vec =  [5, 5, 10,  20]
    fr = 0.18
    plots = False                             # Do you want plots, (slow!)
    filesaves = True                          # Do you want data dump?


    complete = False
    nplot = 1                                 # number of plots within fibre min is 2
    if arguments_determine(-1) == 0:
        fopa = True                         # If no oscillations then the WDMs are deleted to 
                                            # make the system in to a FOPA
    else:
        fopa = False


    if 'mpi' in sys.argv:
        method = 'mpi'
    elif 'joblib' in sys.argv:
        method = 'joblib'
    else:
        method = 'single'
    "--------------------------------------------------------------------------"
    stable_dic = {'num_cores': num_cores, 'maxerr': maxerr, 'ss': ss, 'plots': plots,
                   'nplot': nplot, 'filesaves': filesaves,
                    'fr':fr, 'fopa':fopa}
    "------------------------Can be variable parameters------------------------"
    n2 = 2.5e-20                            # Nonlinear index [m/W]
    gama = 10e-3                             # Overwirtes n2 and Aeff w/m
    alphadB = 0  # 0.0011667#666666666668        # loss within fibre[dB/m]
    z = 18                                    # Length of the fibre
    wave_idx = 0
    power_area_idx = 0
    N = np.array([i for i in range(2,13)])                                    # 2**N grid points
    # Power list. [wavelength, power_area]
    
    P_p_vec = [[my_arange(3.5, 3.9, 0.1), my_arange(4, 4.5, 0.05),
                my_arange(4.6, 8.1 ,0.1), my_arange(8.2,12 ,0.1 ) ],
    
               [my_arange(3.5, 3.9, 0.1), my_arange(4, 4.5, 0.05),
                my_arange(4.6, 8.1 ,0.1), my_arange(8.2,12 ,0.1 ) ],
               
               [my_arange(3.5, 3.9, 0.1), my_arange(4, 4.5, 0.05), 
                my_arange(4.6, 8.1 ,0.1), my_arange(8.2,12 ,0.1 ) ],

               [my_arange(3.5, 3.9, 0.1), my_arange(4, 4.5, 0.05), 
                my_arange(4.6, 8.1 ,0.1), my_arange(8.2,12 ,0.1 ) ],
               
               [my_arange(3.5, 4.4, 0.1), my_arange(4.5, 5, 0.05), 
                my_arange(5.1, 8.1 ,0.1), my_arange(8.2,12 ,0.1 ) ]] 
    
    Df_band = Df_band_vec[power_area_idx]
    P_p = P_p_vec[wave_idx][power_area_idx]
    P_p = [6]#[4.9,4.95,5]
    P_s = 0#100e-3
    TFWHM_p = 0                                # full with half max of pump
    TFWHM_s = 0                                # full with half max of signal
    # loss of each type of splices [dB]
    spl_losses = [0, 0, 1.4]

    betas = np.array([0, 0, 0, 6.756e-2,    # propagation constants [ps^n/m]
                      -1.002e-4, 3.671e-7]) * 1e-3
    lamda_c = 1051.85e-9
    # Zero dispersion wavelength [nm]
    # max at ls,li = 1095, 1010
    WDMS_pars = ([1048., 1204.16],
                 [927.7,  1204.16])  # WDM up downs in wavelengths [m]



    lamp_vec = [1046,1047, 1048, 1049, 1050]
    
    lamp = [lamp_vec[wave_idx]]
    lams = ['lock' for i in range(len(lamp))]
    lamp = lamp_vec[wave_idx]
    lams = 'lock'
    var_dic = {'n2': n2, 'gama': gama, 'alphadB': alphadB, 'z': z, 'P_p': P_p,
               'P_s': P_s, 'TFWHM_p': TFWHM_p, 'TFWHM_s': TFWHM_s,
               'spl_losses': spl_losses, 'betas': betas,
               'lamda_c': lamda_c, 'WDMS_pars': WDMS_pars,
               'lamp': lamp, 'lams': lams, 'N':N, 'Df_band': Df_band}

    "--------------------------------------------------------------------------"
    outside_var_key = 'P_p'
    inside_var_key = 'N'
    inside_var = var_dic[inside_var_key]
    outside_var = var_dic[outside_var_key]
    del var_dic[outside_var_key]
    del var_dic[inside_var_key]
    "----------------------------Simulation------------------------------------"
    D_ins = [{'index': i, inside_var_key: insvar}
             for i, insvar in enumerate(inside_var)]

    large_dic = {**stable_dic, **var_dic}

    if len(inside_var) < num_cores:
        num_cores = len(inside_var)

    profiler_bool = arguments_determine(0)
    for kk, variable in enumerate(outside_var):
        create_file_structure(kk)

        _temps = create_destroy(inside_var, str(kk))
        _temps.prepare_folder()
        large_dic['lams'] = lams[kk]
        large_dic['master_index'] = kk
        large_dic[outside_var_key] = variable
        if profiler_bool:
            for i in range(len(D_ins)):
                formulate(**{**D_ins[i], ** large_dic})
        elif method == 'mpi':
            iterables = ({**D_ins[i], ** large_dic} for i in range(len(D_ins)))
            with MPIPoolExecutor() as executor:
                A = executor.map(formulate, iterables)
        else:
            A = Parallel(n_jobs=num_cores)(delayed(formulate)(**{**D_ins[i], ** large_dic}) for i in range(len(D_ins)))
        _temps.cleanup_folder()
    print('\a')
    return None


class Band_predict(object):
    def __init__(self, Df_band, nt):
        self.bands = []
        self.df = Df_band / nt
        self.ro = []

    def calculate(self, A, Df_band, over_band):
        self.bands.append(Df_band)
        self.ro.append(A)
        if len(bands) == 1:
            return Df_band + 1
        a = (self.bands[-1] - self.bands[-2]) / (self.ro[-1] - self.ro[-2])
        b = self.bands[-1] - a * self.ro[-1]
        for i in over_band:
            try:
                Df_band[i] = a * arguments_determine(-1) + b
            except TypeError:
                Df_band[i] = None
        return Df_band


if __name__ == '__main__':
    start = time()
    main()
    dt = time() - start
    print(dt, 'sec', dt / 60, 'min', dt / 60 / 60, 'hours')
