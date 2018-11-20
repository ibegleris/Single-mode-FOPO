# -*- coding: utf-8 -*-
from __future__ import division, print_function
import sys
import os
import numpy as np
from scipy.constants import pi, c, hbar, Planck
from scipy.io import loadmat
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import simps
from scipy.fftpack import ifftshift
from math import factorial
from integrand_and_rk import *
from data_plotters_animators import *
import cmath
from time import time
from scipy.fftpack import fft, ifft
phasor = np.vectorize(cmath.polar)
import warnings
from functools import wraps
# Pass through the @profile decorator if line profiler (kernprof) is not in use
# Thanks Paul!!
try:
    builtins.profile
except AttributeError:
    def profile(func):
        return func


def arguments_determine(j):
    """
    Makes sence of the arguments that are passed through from sys.agrv. 
    Is used to fix the mpi4py extra that is given. Takes in the possition 
    FROM THE END of the sys.argv inputs that you require (-1 would be the rounds
    for the oscillator).
    """
    A = []
    a = np.copy(sys.argv)
    # a.reverse()
    for i in a[::-1]:
        try:
            A.append(int(i))
        except ValueError:
            continue
    return A[j]


def unpack_args(func):
    if 'mpi' in sys.argv:
        @wraps(func)
        def wrapper(args):
            return func(**args)

        return wrapper
    else:
        return func


def my_arange(a, b, dr, decimals=6):
    res = [a]
    k = 1
    while res[-1] < b:
        tmp = round(a + k*dr, decimals)
        if tmp > b:
            break
        res.append(tmp)
        k += 1

    return np.asarray(res)


def dbm2w(dBm):
    """This function converts a power given in dBm to a power given in W.
       Inputs::
               dBm(float): power in units of dBm
       Returns::
               Power in units of W (float)
    """
    return 1e-3*10**((dBm)/10.)


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


def dispersion_operator(betas, lamda_c, int_fwm, sim_wind):
    """
    Calculates the dispersion operator in rad/m units
    INputed are the dispersion operators at the omega0
    Local include the taylor expansion to get these opeators at omegac 
    Returns Dispersion operator
    """
    c_norm = c*1e-12  # Speed of light [m/ps] #Central wavelength [nm]
    wc = 2*pi * c_norm / sim_wind.lamda
    w0 = 2*pi * c_norm / lamda_c

    betap = np.zeros_like(betas)

    for j in range(len(betas.T)):
        if j == 0:
            betap[j] = betas[j]
        fac = 0
        for k in range(j, len(betas.T)):
            betap[j] += (1/factorial(fac)) * \
                betas[k] * (wc - w0)**(fac)
            fac += 1

    w = sim_wind.w + sim_wind.woffset

    Dop = np.zeros(sim_wind.fv.shape, dtype=np.complex)

    alpha = np.reshape(int_fwm.alpha, np.shape(Dop))
    Dop -= fftshift(alpha/2)
    betap[0] -= betap[0]

    for j, bb in enumerate(betap[2:]):
        Dop -= 1j*(w**(j+2) * bb / factorial(j+2))

    return Dop


def Q_matrixes(nm, n2, lamda_g, gama=None):
    """ Calculates the 1/Aeff (M) from the gamma given.
        The gamma is supposed to be measured at lamda_g
        (in many cases we assume that is the same as where
        the dispersion is measured at).
    """
    if nm == 1:
        # loads M1 and M2 matrices
        mat = loadmat('loading_data/M1_M2_1m_new.mat')
        M1 = np.real(mat['M1'])
        M2 = mat['M2']
        M2[:, :] -= 1
        M1[0:4] -= 1
        M1[-1] -= 1
        #gamma_or = 3*n2*(2*pi/lamda)*M1[4]
        if gama is not None:
            M1[4] = gama / (n2*(2*pi/lamda_g))
            M1[5] = gama / (n2*(2*pi/lamda_g))
        M = M1[4, 0]

    if nm == 2:
        mat = loadmat("loading_data/M1_M2_new_2m.mat")
        M1 = np.real(mat['M1'])
        M2 = mat['M2']
        M2[:] -= 1
        M1[:4, :] -= 1
        M1[6, :] -= 1
    return M


class sim_parameters(object):

    def __init__(self, n2, nm, alphadB):
        self.n2 = n2
        self.nm = nm
        self.alphadB = alphadB

    def general_options(self, maxerr,
                        ss='1'):
        self.maxerr = maxerr
        self.ss = ss
        return None

    def propagation_parameters(self, N, z, nplot, dz_less):
        self.N = N
        self.nt = 2**self.N
        self.z = z
        self.nplot = nplot
        self.dzstep = self.z
        self.dz = self.dzstep/dz_less
        return None


class sim_window(object):

    def __init__(self, fv, lamda, f_centrals, lamda_c, int_fwm):
        self.fv = fv
        self.lamda = lamda
        self.F = f_centrals[1] - f_centrals[0]
        self.deltaf = np.array([np.max(f) - np.min(f) for f in fv])  # [THz]
        self.df = self.deltaf/int_fwm.nt  # [THz]
        self.T = 1 / self.df  # Time window (period)[ps]
        self.fmed = np.array([0.5*(f[-1] + f[0])*1e12 for f in fv])  # [Hz]
        self.fp = 1e-12*c/self.lamda

        self.woffset = 2*pi*(self.fmed[3] - c/lamda)*1e-12  # [rad/ps]

        # central angular frequency [rad/ps]
        self.w0 = 2*pi*np.asarray(f_centrals)

        self.tsh = 1/self.w0  # shock time [ps]
        self.dt = self.T/int_fwm.nt  # timestep (dt)     [ps]
        # time vector      [ps]
        self.t = np.array(
            [(range(int_fwm.nt)-np.ones(int_fwm.nt)*int_fwm.nt/2) * dt for dt in self.dt])
        self.w = np.array(
            [fftshift(2 * pi * (fv - 1e-12*self.fmed[3]), axes=-1) for fv in self.fv])

        self.w_bands = np.array([fftshift(2 * pi * (fv - fc),
                                          axes=-1) for fv, fc in zip(self.fv, f_centrals)])

        self.lv = 1e-3*c/self.fv

        self.zv = int_fwm.dzstep*np.asarray(range(0, int_fwm.nplot+1))

        self.w_tiled = np.copy(self.w_bands)
        self.Omega = 2 * pi * np.abs(f_centrals[1] - f_centrals[0])


def idler_limits(sim_wind, U_original_pump, U, noise_obj):

    size = len(U[:, 0])
    pump_pos = np.argsort(U_original_pump)[-1]
    out_int = np.argsort(U[(pump_pos + 1):, 0])[-1]

    out_int += pump_pos


    lhs_int = np.max(
        np.where(U[pump_pos+1:out_int-1, 0] <= noise_obj.pquant_f)[0])

    rhs_int = np.min(np.where(U[out_int+1:, 0] <= noise_obj.pquant_f)[0])

    lhs_int += pump_pos
    rhs_int += out_int
    lhs_int = out_int - 20
    rhs_int = out_int + 20

    fv_id = (lhs_int, rhs_int)
    return fv_id


class Loss(object):

    def __init__(self, int_fwm, sim_wind, amax=None, apart_div=16):
        """
        Initialise the calss Loss, takes in the general parameters and 
        the freequenbcy window. From that it determines where the loss will become
        freequency dependent. With the default value being an 16th of the difference
        of max and min. 

        """
        self.alpha = int_fwm.alphadB/4.343
        if amax == None:
            self.amax = self.alpha
        else:
            self.amax = amax/4.343

        self.flims_large = [(np.min(f), np.max(f)) for f in sim_wind.fv]

        self.apart = [np.abs(films[1] - films[0])
                      for films in self.flims_large]
        self.apart = [i/apart_div for i in self.apart]
        self.begin = [films[0] + ap for films,
                      ap in zip(self.flims_large, self.apart)]
        self.end = [films[1] - ap for films,
                    ap in zip(self.flims_large, self.apart)]

    def atten_func_full(self, fv):
        a_s, b_s = [], []
        for films, begin, end in zip(self.flims_large, self.begin, self.end):
            a_s.append(((self.amax - self.alpha) / (films[0] - begin),

                        (self.amax - self.alpha) / (films[1] - end)))
            b_s.append((-a_s[-1][0] * begin, -a_s[-1][1] * end))
        aten_large = np.zeros(fv.shape)
        for ii, f in enumerate(fv):
            aten = []

            for ff in f:

                if ff <= self.begin[ii]:
                    aten.append(a_s[ii][0] * ff + b_s[ii][0])
                elif ff >= self.end[ii]:
                    aten.append(a_s[ii][1] * ff + b_s[ii][1])
                else:
                    aten.append(0)
            aten = np.asanyarray(aten)

            # print(aten)
            aten_large[ii, :] = aten
        return aten_large + self.alpha

    def plot(self, fv):

        y = self.atten_func_full(fv)

        fig, ax = plt.subplots(1, 7, sharey=True, figsize=(20, 10))

        for f, yy, axn in zip(fv, y, ax):
            axn.plot(f, yy)

        plt.savefig(
            "loss_function_fibre.png", bbox_inches='tight')
        plt.close(fig)


class WDM(object):

    def __init__(self, x1, x2, fv, modes=1, fopa = False):
        """
                This class represents a 2x2 WDM coupler. The minimum and maximums are
                given and then the object represents the class with WDM_pass the calculation
                done.
        """
        self.l1 = x1   # High part of port 1
        self.l2 = x2  # Low wavelength of port 1
        self.f1 = 1e-3 * c / self.l1   # High part of port 1
        self.f2 = 1e-3 * c / self.l2  # Low wavelength of port 1
        self.omega = 0.5*pi/np.abs(self.f1 - self.f2)
        self.phi = 2*pi - self.omega*self.f2
        self.fv = fv
        self.fv_wdm = self.omega*fv+self.phi

        eps = np.sin(self.fv_wdm)
        eps2 = 1j*np.cos(self.fv_wdm)
        self.A = np.array([[eps, eps2],
                           [eps2, eps]])
        if fopa:
            self.U_calc = self.U_calc_fopa
        return None

    def U_calc_fopa(self, U_in):
        """
        Uses the array defined in __init__ to calculate 
        the outputed amplitude in arbitary units

        """

        return U_in
    def U_calc(self, U_in):
        """
        Uses the array defined in __init__ to calculate 
        the outputed amplitude in arbitary units

        """
        Uout = (self.A[0, 0] * U_in[0] + self.A[0, 1] * U_in[1],)
        Uout += (self.A[1, 0] * U_in[0] + self.A[1, 1] * U_in[1],)

        return Uout

    def pass_through(self, U_in):
        """
        Passes the amplitudes through the object. returns the u, U and Uabs
        in a form of a tuple of (port1,port2)
        """
        # print(np.shape(U_in[0]))
        #U_in[0],U_in[1] = U_in[0][:,np.newaxis],U_in[1][:,np.newaxis]

        U_out = self.U_calc(U_in)
        u_out = ()
        for i, UU in enumerate(U_out):
            u_out += (ifft(ifftshift(UU, axes=-1)),)
            #u_out += (UU,)
        return ((u_out[0], U_out[0]), (u_out[1], U_out[1]))

    def il_port1(self, fv_sp=None):
        """
        For visualisation of the wdm loss of port 1. If no input is given then it is plotted
        in the freequency vector that the function is defined by. You can however 
        give an input in wavelength.
        """
        if fv_sp is None:
            return (np.sin(self.omega*self.fv+self.phi))**2
        else:
            return (np.sin(self.omega*(1e-3*c/fv_sp)+self.phi))**2

    def il_port2(self, fv_sp=None):
        """
        Like il_port1 but with cosine (oposite)
        """
        if fv_sp is None:
            return (np.cos(self.omega*self.fv+self.phi))**2
        else:
            return (np.cos(self.omega*(1e-3*c/fv_sp) + self.phi))**2

    def plot(self, filename=False, xlim=False):
        fig = plt.figure()
        p1, p2 = self.il_port1(), self.il_port2()
        fv, p1, p2 = [np.reshape(i, int(i.shape[0]*i.shape[1]))
                      for i in (self.fv, p1, p2)]

        plt.plot(1e-3*c/fv, p1, 'o-', label="%0.2f" %
                 (self.l1) + ' nm port')
        plt.plot(1e-3*c/fv, p2, 'o-', label="%0.1f" %
                 (self.l2) + ' nm port')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=2)
        plt.xlabel(r'$\lambda (n m)$')
        # plt.xlim()
        plt.ylabel('Power Ratio')
        if xlim:
            plt.xlim(xlim)
        if filename:
            #os.system('mkdir output/WDMs_loss')
            plt.savefig(filename+'.png')
        else:
            plt.show()
        plt.close(fig)
        return None

    def plot_dB(self, lamda, filename=False):
        fig = plt.figure()
        plt.plot(lamda, 10*np.log10(self.il_port1(lamda)),
                 label="%0.2f" % (self.l1*1e9) + ' nm port')
        plt.plot(lamda, 10*np.log10(self.il_port2(lamda)),
                 label="%0.2f" % (self.l2*1e9) + ' nm port')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=2)
        plt.xlabel(r'$\lambda (\mu m)$')
        plt.ylabel(r'$Insertion loss (dB)$')
        plt.ylim(-60, 0)
        #plt.xlim((900, 1250))
        if filename:

            plt.savefig('output/WDMs&loss/WDM_dB_high_' +
                        str(self.l1)+'_low_'+str(self.l2)+'.png')
        else:
            plt.show()
        plt.close(fig)
        return None


def create_file_structure(kk=''):
    """
    Is set to create and destroy the filestructure needed 
    to run the program so that the files are not needed in the repo
    """
    folders_large = ('output_dump',
                     'output_final', 'output'+str(kk))
    folders_large += (folders_large[-1] + '/output',)
    folders_large += (folders_large[-1] + '/data',)
    folders_large += (folders_large[-2] + '/figures',)

    outs = folders_large[-1]
    folders_figures = ('/frequency', '/time', '/wavelength')
    for i in folders_figures:
        folders_figures += (i+'/portA', i+'/portB')
    for i in folders_figures:
        folders_large += (outs + i,)
    folders_large += (outs+'/WDMs',)
    for i in folders_large:
        if not os.path.isdir(i):
            os.system('mkdir ' + i)
    return None


class Splicer(WDM):

    def __init__(self, loss=1):
        self.loss = loss
        self.c1 = 10**(-0.1*self.loss/2.)
        self.c2 = (1 - 10**(-0.1*self.loss))**0.5

    def U_calc(self, U_in):
        """
        Operates like a beam splitter that reduces the optical power by the loss given (in dB).
        """
        U_out1 = U_in[0] * self.c1 + 1j * U_in[1] * self.c2
        U_out2 = 1j * U_in[0] * self.c2 + U_in[1] * self.c1
        return U_out1, U_out2


def norm_const(u, sim_wind):
    t = sim_wind.t
    fv = sim_wind.fv
    U_temp = fftshift(fft(u))
    first_int = simps(np.abs(U_temp)**2, fv)
    second_int = simps(np.abs(u)**2, t)
    return (first_int/second_int)**0.5


class Noise(object):

    def __init__(self, int_fwm, sim_wind):
        self.pquant = np.array([np.sum(
            Planck*(fv*1e12)/(T*1e-12))
            for fv, T in zip(sim_wind.fv, sim_wind.T)])
        self.pquant = (self.pquant/2)**0.5
        return None

    def noise_func(self, int_fwm):
        seed = np.random.seed(int(time()*np.random.rand()))
        noise = np.array([pquant * (np.random.randn(int_fwm.nt)
                                    + 1j*np.random.randn(int_fwm.nt)) for pquant in self.pquant])

        return noise

    def noise_func_freq(self, int_fwm, sim_wind):
        noise = self.noise_func(int_fwm)
        noise_freq = fftshift(fft(noise), axes=-1)
        return noise_freq





def dF_sidebands(beta, lamp, lam_z, n2, M, P_p,P_s):
    omegap, omega_z = (1e-12*2*pi*c/i for i in (lamp, lam_z))

    omega = omegap - omega_z
    gama = 1e12 * n2*omegap/(c * (1/M))
    a = beta[4]/12 + omega * beta[5]/12
    b = beta[2] + omega * beta[3] + omega**2 * \
        beta[4] / 2 + omega**3 * beta[5]/6
    g = 2 * gama * P_p
    det = b**2 - 4 * a * g

    if det < 0:
        print('No sidebands predicted by simple model!')
        sys.exit(1)
    Omega = np.array([(-b + det**0.5) / (2*a), (-b - det**0.5) / (2*a)])
    Omega = Omega[Omega > 0]
    Omega = [i**0.5 for i in Omega]
    Omega = (Omega * np.logical_not(np.iscomplex(Omega))).real
    F = np.max(Omega) / (2*pi)
    f_p = omegap/(2*pi)
    print('frequency band', F)
    return F, f_p


def pre_fibre_init_power(l1, l2, lamp, P_p, P_s):
    f1 = 1e-3 * c / l1
    f2 = 1e-3 * c / l2
    fp = 1e-3 * c / lamp
    omega = 0.5*pi/np.abs(f1 - f2)
    phi = 2*pi - omega*f2
    res = [i * np.sin(omega * fp + phi)**2 for i in (P_p, P_s)]
    return res


def fv_creator(lamp, lams, lamda_c, int_fwm, betas, M, P_p,P_s, Df_band=1):
    """
    Cretes 7 split frequency grid set up around the waves from degenerate
    FWM. The central freuency of the bands is determined by the non-depleted
    pump approximation and is power dependent. The wideness of these bands
    is determined inputed. This returns an array of shape [7, nt] with 
    each collumn holding the data of the 7 frequency bands. 
    Inputs::
        lamp: wavelength of the pump (float)
        lamda_c: wavelength of the zero dispersion wavelength(ZDW) (float)
        int_fwm: class that holds nt (number of points in each band)
        betas: Taylor coeffiencts of beta around the ZDW (Array)
        M : The M coefficient (or 1/A_eff) (float)
        P_p: pump power
        Df_band: band frequency bandwidth in Thz, (float)
    Output::
        fv: Frequency vector of bands (Array of shape [7, nt]) 
    """

    fv = np.zeros([7, int_fwm.nt], dtype=np.float)
    lamp *= 1e-9

    F, f_p = dF_sidebands(betas, lamp, lamda_c, int_fwm.n2, M, P_p, P_s)

    f_centrals = [f_p + i * F for i in range(-3, 4)]
    # print(1e-3*c/f_centrals[2])
    # sys.exit()
    a = np.linspace(-Df_band/2, 0, int_fwm.nt//2)
    da = a[-1] - a[-2]
    b = [a[-1]+da]
    for i in range(1, int_fwm.nt//2):
        b.append(b[i-1] + da)
    defa = np.concatenate((a, b))
    for i, fc in enumerate(f_centrals):
        fv[i, :] = fc + defa
    if lams == 'lock' or P_s == 0:
        fp = f_p
        fs = f_p - F
        fi = f_p + F
    else:
        fp = f_p
        fs =  1e-3 * c/lams
        fi = fp - (fs - fp)
    p_pos = np.where(np.abs(fv - fp) == np.min(np.abs(fv - fp)))
    s_pos = np.where(np.abs(fv - fs) == np.min(np.abs(fv - fs)))
    i_pos = np.where(np.abs(fv - fi) == np.min(np.abs(fv - fi)))
    p_pos = [p_pos[0][0], p_pos[1][0]]

    s_pos = [s_pos[0][0], s_pos[1][0]]
    i_pos = [i_pos[0][0], i_pos[1][0]]
    where = [p_pos, s_pos, i_pos]
    check_ft_grid(fv, da)
    return fv, where, f_centrals


def energy_conservation(entot):
    if not(np.allclose(entot, entot[0])):
        fig = plt.figure()
        plt.plot(entot)
        plt.grid()
        plt.xlabel("nplots(snapshots)", fontsize=18)
        plt.ylabel("Total energy", fontsize=18)
        # plt.show()
        plt.close()
        sys.exit("energy is not conserved")
    return 0


def check_ft_grid(Fv, diff):
    """Grid check for fft optimisation"""
    for fv in Fv:
        if fv.any() < 0:
            sys.exit("some of your grid is negative")

        if np.log2(np.shape(fv)[0]) == int(np.log2(np.shape(fv)[0])):
            nt = np.shape(fv)[0]
        else:
            print("fix the grid for optimization\
                 of the fft's, grid:" + str(np.shape(fv)[0]))
            sys.exit(1)

        lvio = []
        for i in range(len(fv)-1):
            lvio.append(fv[i+1] - fv[i])

        grid_error = np.abs(np.asanyarray(lvio)[:]) - np.abs(diff)
        if not(np.allclose(grid_error, 0, rtol=0, atol=1e-12)):
            print(np.max(grid_error))
            sys.exit("your grid is not uniform")
    return 0


class create_destroy(object):
    """
    creates and destroys temp folder that is used for computation. Both methods needs to be run
    before you initiate a new variable
    """

    def __init__(self, variable, pump_wave=''):
        self.variable = variable
        self.pump_wave = pump_wave
        return None

    def cleanup_folder(self):
        # for i in range(len(self.variable)):
        os.system('mv output'+self.pump_wave + ' output_dump/')
        return None

    def prepare_folder(self):
        for i in range(len(self.variable)):
            os.system('cp -r output'+self.pump_wave +
                      '/output/ output'+self.pump_wave+'/output'+str(i))
        return None


def power_idler(spec, fv, sim_wind, fv_id):
    """
    Set to calculate the power of the idler. The possitions
    at what you call an idler are given in fv_id
    spec: the spectrum in freequency domain
    fv: the freequency vector
    T: time window
    fv_id: tuple of the starting and
    ending index at which the idler is calculated
    """
    E_out = simps((sim_wind.t[1] - sim_wind.t[0])**2 *
                  np.abs(spec[fv_id[0]:fv_id[1], 0])**2, fv[fv_id[0]:fv_id[1]])
    P_bef = E_out/(2*np.max(sim_wind.t))
    return P_bef


class Raman_factors(object):
    def __init__(self, fr, sim_wind):

        self.H = np.array([i*0j for i in range(12)])


class Raman_factors(object):
    def __init__(self, fr, how='load'):
        how = how
        if how == 'load':
            self.get_raman = self.raman_load
        else:
            self.get_raman = self.raman_analytic
        self.fr = fr

    def get_large_grid(self, sim_wind):
        fmax = np.max(sim_wind.fv)+ sim_wind.fp
        fmin = np.min(sim_wind.fv)- sim_wind.fp
        df = sim_wind.fv[0, 1] - sim_wind.fv[0, 0]
        fv = np.arange(fmin, fmax, df, dtype=np.float64)
        nt = len(fv)
        T = 1 / df
        dt = T/nt
        t = (range(nt)-np.ones(nt)*nt/2) * dt
        t += np.abs(t.min())
        dt = sim_wind.dt[0]
        
        fv_ram = fv - sim_wind.fp
        return fv_ram, dt, t, sim_wind.F

    def set_raman_band(self, sim_wind):
        fv_ram, dt_small, t_large, F = self.get_large_grid(sim_wind)
        hfmeas = self.get_raman(t_large, dt_small)
        hfmeas_func_r = InterpolatedUnivariateSpline(fv_ram, hfmeas.real)
        hfmeas_func_i = InterpolatedUnivariateSpline(fv_ram, hfmeas.imag)
        fmids = [i*F for i in range(-6,7)]
        raman_band_factors_re = [hfmeas_func_r(f)
                                 for f in fmids]

        raman_band_factors_im = [hfmeas_func_i(f)
                                 for f in fmids]
        raman_band_factors = [(i + 1j * j)*dt_small for i, j in
                              zip(raman_band_factors_re, raman_band_factors_im)]
        self.H = raman_band_factors
        return None

    def raman_load(self, t_large, dt_small):

        mat = loadmat('loading_data/silicaRaman.mat')
        ht = mat['ht']
        t1 = mat['t1']
        htmeas_func = InterpolatedUnivariateSpline(t1*1e-3, ht)
        htmeas = htmeas_func(t_large)
        htmeas *= (t_large > 0)*(t_large < 1)
        htmeas /= (dt_small * np.sum(htmeas))
        hfmeas = fftshift(fft(htmeas))
        return hfmeas

    def raman_analytic(self, t_large, dt_small):
        t11 = 12.2e-3     # [ps]
        t2 = 32e-3        # [ps]
        # analytical response
        ht = (t11**2 + t2**2)/(t11*t2**2) * \
            np.exp(-t/t2*(t >= 0))*np.sin(t/t11)*(t >= 0)
        ht_norm = ht / (dt_small * np.sum(ht))
        htmeas = fftshift(fft(ht_norm))
        return htmeas

class Phase_modulation_FOPA(object):
    """
    Makes sure that the signal is in phase with the other
    waves so that maximum FOPA conversion can occur.
    """
    def __init__(self, fv, where,const_phi = False):

        
        self.shape1 = fv.shape[0]
        self.fv = fv
        self.p_pos = where[0]
        self.i_pos = where[2]
        self.s_pos = where[1]
        if const_phi != False:
            self.modulate = self.modulate_const
            self.dtheta = const_phi
        else:
            self.modulate = self.modulate_var
    def modulate_const(self, U):
        U[self.i_pos[0],:] = U[self.i_pos[0],:] * np.exp(1j * self.dtheta)
        return U
    
    def modulate_var(self, U):
        angles = np.angle(U[2:5])
        dtheta = self._dtheta(angles)
        U[self.i_pos[0],  :-1] = U[self.i_pos[0], :-1] * np.exp(1j * dtheta)
        return U

    def _dtheta(self, angles):

        return 2 * angles[self.p_pos[0]-2, self.p_pos[1]] \
                - angles[self.i_pos[0]-2, :-1] - angles[self.s_pos[0]-2,-2::-1] - 0.5 * pi


class Phase_modulation_infase_WDM(object):
    """
    Makes sure that the signal is in phase with the oscillating signal 
    comiing in so we can get constructive inteference.
    """
    def __init__(self, P_s,where,WDM_1):
        self.A = WDM_1.A[0]
        self.s_band_pos = where[1][0]
        if P_s ==0:
            self.modulate = self.modulate_unseeded
        else:
            self.modulate = self.modulate_seeded
        
    def modulate_unseeded(self, U1, U2):
        return U2

    def modulate_seeded(self, U1, U2):
        idx = np.argmax(np.abs(U1[self.s_band_pos,:])**2, axis = -1)
        dphi = np.angle(U1[self.s_band_pos, idx] * self.A[0][self.s_band_pos, idx]) -\
                 np.angle(U2[self.s_band_pos, idx] * self.A[1][self.s_band_pos, idx])

        return U2 * np.exp(1j * dphi)