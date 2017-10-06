from __future__ import division, print_function
import numpy as np
#import matplotlib as mpl
# mpl.use('Agg')
import sys
from scipy.constants import c, pi
from joblib import Parallel, delayed
from mpi4py.futures import MPIPoolExecutor
from mpi4py import MPI
from scipy.fftpack import fftshift, fft
import os
import time as timeit
os.system('export FONTCONFIG_PATH=/etc/fonts')
from functions import *
from fft_module import *

from time import time, sleep


def oscilate(sim_wind, int_fwm, noise_obj, TFWHM_p, TFWHM_s, index,
             master_index, P0_p1, P0_s, f_p, f_s, p_pos, s_pos, splicers_vec,
             WDM_vec, M, hf, Dop, dAdzmm, D_pic, pulse_pos_dict_or, plots):

    u = np.zeros(
        [len(sim_wind.t), len(sim_wind.zv)], dtype='complex128')
    U = np.zeros([len(sim_wind.t),
                  len(sim_wind.zv)], dtype='complex128')

    T0_p = TFWHM_p/2/(np.log(2))**0.5
    T0_s = TFWHM_s/2/(np.log(2))**0.5
    noise_new = noise_obj.noise_func(int_fwm)
    u[:, 0] = noise_new

    woff1 = (p_pos+(int_fwm.nt)//2)*2*pi*sim_wind.df
    u[:, 0] += (P0_p1)**0.5 * np.exp(1j*(woff1)*sim_wind.t)

    woff2 = -(s_pos - (int_fwm.nt-1)//2)*2*pi*sim_wind.df
    # *np.exp(-sim_wind.t**2/T0_s)
    u[:, 0] += (P0_s)**0.5 * np.exp(-1j*(woff2)*sim_wind.t)

    U[:, 0] = fftshift(fft(u[:, 0]))

    sim_wind.w_tiled = sim_wind.w + sim_wind.woffset
    master_index = str(master_index)

    plotter_dbm(index, int_fwm.nm, sim_wind, u, U, P0_p1,
                P0_s, f_p, f_s, 0, 0, master_index, '00',
                'original pump', D_pic[0], plots)

    U_original_pump = np.copy(U[:, 0])

    noise_new = noise_obj.noise_func_freq(int_fwm, sim_wind)
    u[:, 0], U[:, 0] = WDM_vec[0].pass_through(
        (U[:, 0], noise_new), sim_wind)[0]

    max_rounds = arguments_determine(-1)

    ro = -1

    t_total = 0

    #P_out = []
    #w= Window(40)
    converged = False
    while ro < max_rounds and not(converged):

        ro += 1

        print('round', ro)
        pulse_pos_dict = [
            'round ' + str(ro)+', ' + i for i in pulse_pos_dict_or]

        plotter_dbm(index, int_fwm.nm, sim_wind, u, U, P0_p1,
                    P0_s, f_p, f_s, 0, ro, master_index,
                    str(ro) + '1', pulse_pos_dict[3], D_pic[5], plots)

        u, U = pulse_propagation(u, U, int_fwm, M, sim_wind, hf, Dop, dAdzmm)

        plotter_dbm(index, int_fwm.nm, sim_wind, u, U, P0_p1,
                    P0_s, f_p, f_s, -1, ro, master_index,
                    str(ro)+'2', pulse_pos_dict[0], D_pic[2], plots)

        # pass through WDM2 port 2 continues and port 1 is out of the loop
        noise_new = noise_obj.noise_func_freq(int_fwm, sim_wind)
        (out1, out2), (u[:, -1], U[:, -1]) = WDM_vec[1].pass_through(
            (U[:, -1], noise_new), sim_wind)

        plotter_dbm(index, int_fwm.nm, sim_wind, u, U, P0_p1,
                    P0_s, f_p, f_s, -1, ro, master_index,
                    str(ro)+'3', pulse_pos_dict[1], D_pic[3], plots)

        # Splice7 after WDM2 for the signal
        noise_new = noise_obj.noise_func_freq(int_fwm, sim_wind)

        (u[:, -1], U[:, -1]) = splicers_vec[2].pass_through(
            (U[:, -1], noise_new), sim_wind)[0]

        # Pass again through WDM1 with the signal now
        (u[:, 0], U[:, 0]) = WDM_vec[0].pass_through(
            (U_original_pump, U[:, -1]), sim_wind)[0]

        ################################The outbound stuff#####################
        U_out = np.reshape(out2, (len(sim_wind.t), 1))
        u_out = np.reshape(out1, (len(sim_wind.t), 1))
        plotter_dbm(index, int_fwm.nm, sim_wind, u_out, U_out, P0_p1,
                    P0_s, f_p, f_s, -1, ro, master_index,
                    str(ro)+'4', pulse_pos_dict[4], D_pic[6], plots)
        # if ro == np.sum(w.size_vec):
        #	converged = converge_checker(w,P_out, 20)
        #	print('next check on', np.sum(w.size_vec))

    return None


def calc_P_out(U, U_original_pump, fv, t):
    U = np.abs(U)**2
    U_original_pump = np.abs(U_original_pump)**2
    freq_band = 2
    fp_id = np.where(U_original_pump == np.max(U_original_pump))[0][0]
    plom = fp_id+10
    fv_id = np.where(U[plom:] == np.max(U[plom:]))[0][0]
    fv_id += plom-1
    #fv_id = fp_id
    start, end = fv[fv_id] - freq_band, fv[fv_id] + freq_band
    i = np.where(
        np.abs(fv - start) == np.min(np.abs(fv - start)))[0][0]
    j = np.where(
        np.abs(fv - end) == np.min(np.abs(fv - end)))[0][0]
    E_out = simps(U[i:j]*(t[1] - t[0])**2, fv[i:j])
    P_out = E_out/(2*np.max(t))
    return P_out


@unpack_args
def formulate(index, n2, gama, alphadB, z, P_p, P_s, TFWHM_p,
              TFWHM_s, spl_losses, betas, lamda_c, WDMS_pars,
              lamp, lams, num_cores, maxerr, ss, ram, plots,
              N, nt, nplot, master_index):
    "------------------propagation paramaters------------------"
    dzstep = z/nplot						# distance per step
    dz_less = 1e2
    # dz = dzstep/dz_less		 # starting guess value of the step
    int_fwm = sim_parameters(n2, 1, alphadB)
    int_fwm.general_options(maxerr, raman_object, ss, ram)
    int_fwm.propagation_parameters(N, z, nplot, dz_less)
    lamda = lamp*1e-9  # central wavelength of the grid[m]
    fv_idler_int = 10  # safety for the idler to be spotted
    "-----------------------------f-----------------------------"

    "---------------------Aeff-Qmatrixes-----------------------"
    M = Q_matrixes(int_fwm.nm, int_fwm.n2, lamda, gama)
    "----------------------------------------------------------"

    "---------------------Grid&window-----------------------"
    fv, where = fv_creator(lamp, lams, int_fwm, prot_casc=0)
    p_pos, s_pos = where
    sim_wind = sim_window(fv, lamda, lamda_c, int_fwm, fv_idler_int)
    "----------------------------------------------------------"

    "---------------------Loss-in-fibres-----------------------"
    slice_from_edge = (sim_wind.fv[-1] - sim_wind.fv[0])/100
    loss = Loss(int_fwm, sim_wind, amax=0)

    int_fwm.alpha = loss.atten_func_full(fv)

    "----------------------------------------------------------"

    "--------------------Dispersion----------------------------"
    Dop = dispersion_operator(betas, lamda_c, int_fwm, sim_wind)
    "----------------------------------------------------------"

    "--------------------Noise---------------------------------"
    noise_obj = Noise(int_fwm, sim_wind)
    "----------------------------------------------------------"

    "---------------Formulate the functions to use-------------"
    string = "dAdzmm_r"+str(int_fwm.ram)+"_s"+str(int_fwm.ss)
    func_dict = {'dAdzmm_ron_s1': dAdzmm_ron_s1,
                 'dAdzmm_ron_s0': dAdzmm_ron_s0,
                 'dAdzmm_roff_s0': dAdzmm_roff_s0,
                 'dAdzmm_roff_s1': dAdzmm_roff_s1}
    pulse_pos_dict_or = ('after propagation', "pass WDM2",
                         "pass WDM1 on port2 (remove pump)",
                         'add more pump', 'out')

    keys = ['loading_data/green_dot_fopo/pngs/' +
            str(i)+str('.png') for i in range(7)]
    D_pic = [plt.imread(i) for i in keys]

    dAdzmm = func_dict[string]
    raman = raman_object(int_fwm.ram, int_fwm.how)
    raman.raman_load(sim_wind.t, sim_wind.dt)
    hf = raman.hf
    "--------------------------------------------------------"

    "----------------------Formulate WDMS--------------------"
    if WDMS_pars == 'signal_locked':

        Omega = 2*pi*c/(lamp*1e-9) - 2*pi*c/(lams*1e-9)
        omegai = 2*pi*c/(lamp*1e-9) + Omega
        lami = 1e9*2*pi*c/(omegai)
        WDMS_pars = ([lamp, lams], 	# WDM up downs in wavelengths [m]
                     [lami, lams],
                     [lami, lamp],
                     [lami, lams])
    # print(WDMS_pars)
    # sys.exit()
    # print(lamp)
    # WDMS_pars = ([lamp, 1200.39], 	# WDM up downs in wavelengths [m]
    #			[930,  1200.39],
    #			[930,lamp],
    #			[930, 1200.39])
    WDM_vec = [WDM(i[0], i[1], sim_wind.fv, c)
               for i in WDMS_pars]  # WDM up downs in wavelengths [m]

    # print(WDMS_pars)
    # sys.exit()
    "--------------------------------------------------------"
    # for ei,i in enumerate(WDM_vec):
    #	i.plot(filename = str(ei))
    "----------------------Formulate splicers--------------------"
    splicers_vec = [Splicer(loss=i) for i in spl_losses]
    "------------------------------------------------------------"

    f_p, f_s = 1e-3*c/lamp, 1e-3*c/lams
    oscilate(sim_wind, int_fwm, noise_obj, TFWHM_p, TFWHM_s, index,
             master_index, P_p, P_s, f_p, f_s, p_pos, s_pos, splicers_vec,
             WDM_vec, M, hf, Dop, dAdzmm, D_pic, pulse_pos_dict_or, plots)

    # while cop !=0:
    #	sleep(0.5)
    return None


def main():
    "---------Stable parameters--------"
    num_cores = arguments_determine(1)	 	# Number of computing cores
    maxerr = 1e-13							# maximum tolerable error per step in integration
    ss = 1				  					# includes self steepening term
    ram = 'on'				  				# Raman contribution 'on' if yes and 'off' if no
    plots = False 							# Do you want plots?
    N = 12									# 2**N grid points
    nt = 2**N 								# number of grid points
    nplot = 2								# number of plots within fibre min is 2
    if 'mpi' in sys.argv:
        method = 'mpi'
    elif 'joblib' in sys.argv:
        method = 'joblib'
    else:
        method = 'single'
    "----------------------------------------"
    stable_dic = {'num_cores': num_cores, 'maxerr': maxerr,
                  'ss': ss, 'ram': ram, 'plots': plots,
                  'N': N, 'nt': nt, 'nplot': nplot}
    "--------------Can be variable parameters-----------"
    n2 = 2.5e-20							# Nonlinear index [m/W]
    gama = 10e-3 							# Overwirtes n2 and Aeff w/m
    alphadB = 0  # 0.0011667#666666666668		# loss within fibre[dB/m]
    z = 18									# Length of the fibre
    P_p = [1, 2, 4, 5, 6]
    P_s = 0*100e-3  # [10e-3,100e-3,1]							# Signal power [W]
    TFWHM_p = 0								# full with half max of pump
    TFWHM_s = 0								# full with half max of signal
    spl_losses = [[0, 0, 1.], [0, 0, 1.2], [0, 0, 1.3], [
        0, 0, 1.4]]					# loss of each type of splices [dB]
    spl_losses = [0, 0, 1.4]

    betas = np.array([0, 0, 0, 6.756e-2,  # propagation constants [ps^n/m]
                      -1.002e-4, 3.671e-7])*1e-3
    lamda_c = 1051.85e-9
    # Zero dispersion wavelength [nm]
    # max at ls,li = 1095, 1010
    variation = np.arange(-28, 42, 2)
    variation = [0]
    WDMS_pars = ([1048., 1199.32],
                 [930.996,  1199.32])  # WDM up downs in wavelengths [m]

    #WDMS_pars = []
    # for i in variation:
    #	WDMS_pars.append(([1051.5, 1095+i], 	# WDM up downs in wavelengths [m]
    #					[1011.4,  1095],
    #					[1011.4,1051.5],
    #					[1011.4, 1095]))

    # print(WDMS_pars)
    # sys.exit()
    # WDMS_pars = 'signal_locked' # lockes the WDMS to keep the max amount of
    # signal in the cavity (seeded)

    lamp = [1046, 1048, 1050]
    lams = [1241.09, 1199.32, 1149.35]
    lamp = [lamp[0]]

    lams = lams[0]
    var_dic = {'n2': n2, 'gama': gama, 'alphadB': alphadB, 'z': z, 'P_p': P_p,
               'P_s': P_s, 'TFWHM_p': TFWHM_p, 'TFWHM_s': TFWHM_s,
               'spl_losses': spl_losses, 'betas': betas,
               'lamda_c': lamda_c, 'WDMS_pars': WDMS_pars,
               'lamp': lamp, 'lams': lams}

    "----------------------------"
    outside_var_key = 'lamp'
    inside_var_key = 'P_p'
    inside_var = var_dic[inside_var_key]
    outside_var = var_dic[outside_var_key]
    del var_dic[outside_var_key]
    del var_dic[inside_var_key]
    "-----------Simulation----------"
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
            A = Parallel(n_jobs=num_cores)(delayed(formulate)
                                           (**{**D_ins[i], ** large_dic})
                                           for i in range(len(D_ins)))
        _temps.cleanup_folder()

    print('\a')
    return None

if __name__ == '__main__':
    start = time()
    main()
    dt = time() - start
    print(dt, 'sec', dt/60, 'min', dt/60/60, 'hours')
