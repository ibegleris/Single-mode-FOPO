# -*- coding: utf-8 -*-
from __future__ import division, print_function
import sys
import os
import numpy as np
from scipy.constants import pi, c
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


class raman_object(object):

    def __init__(self, a, b=None):
        self.on = a
        self.how = b
        self.hf = None

    def raman_load(self, t, dt):
        if self.on == 'on':
            #print('Raman on')
            if self.how == 'analytic':
                print(self.how)
                t11 = 12.2e-3     # [ps]
                t2 = 32e-3       # [ps]
                # analytical response
                htan = (t11**2 + t2**2)/(t11*t2**2) * \
                    np.exp(-t/t2*(t >= 0))*np.sin(t/t11)*(t >= 0)
                # Fourier transform of the analytic nonlinear response
                self.hf = fft(htan)
            elif self.how == 'load':
                # loads the measured response (Stolen et al. JOSAB 1989)
                mat = loadmat('loading_data/silicaRaman.mat')
                ht = mat['ht']
                t1 = mat['t1']
                htmeas_f = InterpolatedUnivariateSpline(t1*1e-3, ht)
                htmeas = htmeas_f(t)
                htmeas *= (t > 0)*(t < 1)  # only measured between +/- 1 ps)
                htmeas /= (dt*np.sum(htmeas))  # normalised
                # Fourier transform of the measured nonlinear response
                self.hf = fft(htmeas)
            else:
                self.hf = None

            return self.hf


def dispersion_operator(betas, lamda_c, int_fwm, sim_wind):
    """
    Calculates the dispersion operator in rad/m units
    INputed are the dispersion operators at the omega0
    Local include the taylor expansion to get these opeators at omegac 
    Returns Dispersion operator
    """
    # print(betas)
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

    Dop = np.zeros(int_fwm.nt, dtype=np.complex)
    alpha = np.reshape(int_fwm.alpha, np.shape(Dop))
    Dop -= fftshift(alpha/2)
    #Dop -=alpha/2
    betap[0] -= betap[0]
    betap[1] -= betap[1]

    for j, bb in enumerate(betap):
        Dop -= 1j*(w**j * bb / factorial(j))
    return Dop


def Q_matrixes(nm, n2, lamda, gama=None):
    """Calculates the Q matrices from importing them from a file.
     CHnages the gama if given"""
    if nm == 1:
        # loads M1 and M2 matrices
        mat = loadmat('loading_data/M1_M2_1m_new.mat')
        M1 = np.real(mat['M1'])
        M2 = mat['M2']
        M2[:, :] -= 1
        M1[0:4] -= 1
        M1[-1] -= 1
        gamma_or = 3*n2*(2*pi/lamda)*M1[4]
        if gama is not None:
            M1[4] = gama / (n2*(2*pi/lamda))
            M1[5] = gama / (n2*(2*pi/lamda))
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

    def general_options(self, maxerr, raman_object,
                        ss='1', ram='on', how='load'):
        self.maxerr = maxerr
        self.ss = ss
        self.ram = ram
        self.how = how
        return None

    def propagation_parameters(self, N, z, nplot, dz_less):
        self.N = N
        self.nt = 2**self.N
        self.z = z
        self.nplot = nplot
        self.dzstep = self.z/self.nplot
        self.dz = self.dzstep/dz_less
        return None


class sim_window(object):

    def __init__(self, fv, lamda, lamda_c, int_fwm, fv_idler_int):
        self.fv = fv
        self.lamda = lamda
        # self.lmin = 1e-3*c/np.max(fv)  # [nm]
        # self.lmax = 1e-3*c/np.min(fv)  # [nm]

        self.fmed = 0.5*(fv[-1] + fv[0])*1e12  # [Hz]
        self.deltaf = np.max(self.fv) - np.min(self.fv)  # [THz]
        self.df = self.deltaf/int_fwm.nt  # [THz]
        self.T = 1/self.df  # Time window (period)[ps]
        # print(self.fmed,c/lamda)
        # sys.exit()
        self.woffset = 2*pi*(self.fmed - c/lamda)*1e-12  # [rad/ps]
        # [rad/ps] Offset of central freequency and that of the experiment
        self.woffset2 = 2*pi*(self.fmed - c/lamda_c)*1e-12
        # wavelength limits (for plots) (nm)
        #self.fv = fv
        self.w0 = 2*pi*self.fmed  # central angular frequency [rad/s]

        self.tsh = 1/self.w0*1e12  # shock time [ps]
        self.dt = self.T/int_fwm.nt  # timestep (dt)     [ps]
        # time vector      [ps]
        self.t = (range(int_fwm.nt)-np.ones(int_fwm.nt)*int_fwm.nt/2)*self.dt
        # angular frequency vector [rad/ps]
        self.w = 2*pi * np.append(
            range(0, int(int_fwm.nt/2)),
            range(int(-int_fwm.nt/2), 0, 1))/self.T
        #self.w = fftshift(2*pi *(self.fv - 1e-12*self.fmed))
        # plt.plot(self.w)
        # plt.savefig('w.png')
        # sys.exit()
        # frequency vector[THz] (shifted for plotting)
        # wavelength vector [nm]
        self.lv = 1e-3*c/self.fv
        # space vector [m]
        self.zv = int_fwm.dzstep*np.asarray(range(0, int_fwm.nplot+1))
        self.fv_idler_int = fv_idler_int
        self.fv_idler_tuple = (
            self.fmed*1e-12 - fv_idler_int, self.fmed*1e-12 + fv_idler_int)

        # for i in (self.fv,self.t, fftshift(self.w)):
        #   check_ft_grid(i, np.abs(i[1] - i[0]))


def idler_limits(sim_wind, U_original_pump, U, noise_obj):

    size = len(U[:, 0])
    pump_pos = np.argsort(U_original_pump)[-1]
    out_int = np.argsort(U[(pump_pos + 1):, 0])[-1]

    out_int += pump_pos
    # print(1e-3*c/sim_wind.fv[pump_pos])
    # print(1e-3*c/sim_wind.fv[out_int])
    # sys.exit()

    lhs_int = np.max(
        np.where(U[pump_pos+1:out_int-1, 0] <= noise_obj.pquant_f)[0])

    rhs_int = np.min(np.where(U[out_int+1:, 0] <= noise_obj.pquant_f)[0])

    lhs_int += pump_pos
    rhs_int += out_int
    lhs_int = out_int - 20
    rhs_int = out_int + 20
    # if lhs_int > out_int:
    #    lhs_int = out_int - 10

    fv_id = (lhs_int, rhs_int)
    #print(1e-3*c/sim_wind.fv[lhs_int] - 1e-3*c/sim_wind.fv[out_int])
    return fv_id


class Loss(object):

    def __init__(self, int_fwm, sim_wind, amax=None, apart_div=8):
        """
        Initialise the calss Loss, takes in the general parameters and 
        the freequenbcy window. From that it determines where the loss will become
        freequency dependent. With the default value being an 8th of the difference
        of max and min. 

        """
        self.alpha = int_fwm.alphadB/4.343
        if amax == None:
            self.amax = self.alpha
        else:
            self.amax = amax/4.343

        self.flims_large = (np.min(sim_wind.fv), np.max(sim_wind.fv))
        try:
            temp = len(apart_div)
            self.begin = apart_div[0]
            self.end = apart_div[1]
        except TypeError:

            self.apart = np.abs(self.flims_large[1] - self.flims_large[0])
            self.apart /= apart_div
            self.begin = self.flims_large[0] + self.apart
            self.end = self.flims_large[1] - self.apart

    def atten_func_full(self, fv):
        aten = []

        a_s = ((self.amax - self.alpha) / (self.flims_large[0] - self.begin),

               (self.amax - self.alpha) / (self.flims_large[1] - self.end))
        b_s = (-a_s[0] * self.begin, -a_s[1] * self.end)

        for f in fv:
            if f <= self.begin:
                aten.append(a_s[0] * f + b_s[0])
            elif f >= self.end:
                aten.append(a_s[1] * f + b_s[1])
            else:
                aten.append(0)
        return np.asanyarray(aten) + self.alpha

    def plot(self, fv):
        fig = plt.figure()
        y = self.atten_func_full(fv)
        plt.plot(fv, y)
        plt.xlabel("Frequency (Thz)")
        plt.ylabel("Attenuation (cm -1 )")
        plt.savefig(
            "loss_function_fibre.png", bbox_inches='tight')
        plt.close(fig)


class WDM(object):

    def __init__(self, x1, x2, fv, c, modes=1):
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

        # self.A = np.array([[np.reshape(np.cos(self.fv), (len(self.fv), modes)),
        #                       np.reshape(np.sin(self.fv), (len(self.fv), modes))],
        #                      [-np.reshape(np.sin(self.fv), (len(self.fv), modes)),
        # np.reshape(np.cos(self.fv), (len(self.fv), modes))]])

        eps = np.sin(self.fv_wdm)
        eps2 = 1j*np.cos(self.fv_wdm)
        self.A = np.array([[eps, eps2],
                           [eps2, eps]])

        return None

    def U_calc(self, U_in):
        """
        Uses the array defined in __init__ to calculate 
        the outputed amplitude in arbitary units

        """

        Uout = (self.A[0, 0] * U_in[0] + self.A[0, 1] * U_in[1],)
        Uout += (self.A[1, 0] * U_in[0] + self.A[1, 1] * U_in[1],)

        return Uout

    def pass_through(self, U_in, sim_wind):
        """
        Passes the amplitudes through the object. returns the u, U and Uabs
        in a form of a tuple of (port1,port2)
        """
        # print(np.shape(U_in[0]))
        #U_in[0],U_in[1] = U_in[0][:,np.newaxis],U_in[1][:,np.newaxis]

        U_out = self.U_calc(U_in)
        u_out = ()
        for i, UU in enumerate(U_out):
            u_out += (ifft(fftshift(UU)),)
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
        plt.plot(1e-3*c/self.fv, self.il_port1(), label="%0.2f" %
                 (self.l1) + ' nm port')
        plt.plot(1e-3*c/self.fv, self.il_port2(), label="%0.1f" %
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
    folders_figures = ('/freequency', '/time', '/wavelength')
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
        self.pquant = np.sum(
            1.054e-34*(sim_wind.w*1e12 + sim_wind.w0)/(sim_wind.T*1e-12))
        # print(self.pquant**0.5)
        self.pquant = (self.pquant/2)**0.5
        self.pquant_f = np.mean(
            np.abs(self.noise_func_freq(int_fwm, sim_wind))**2)
        return None

    def noise_func(self, int_fwm):
        seed = np.random.seed(int(time()*np.random.rand()))
        noise = self.pquant * (np.random.randn(int_fwm.nt)
                               + 1j*np.random.randn(int_fwm.nt))
        return noise

    def noise_func_freq(self, int_fwm, sim_wind):
        noise = self.noise_func(int_fwm)
        noise_freq = fftshift(fft(noise))
        return noise_freq
#import warnings
# warnings.filterwarnings("error")


def pulse_propagation(u, U, int_fwm, M, sim_wind, hf, Dop, dAdzmm):
    """Pulse propagation"""
    # badz = 0  # counter for bad steps
    # goodz = 0  # counter for good steps
    dztot = 0  # total distance traveled
    #dzv = np.zeros(1)
    #dzv[0] = int_fwm.dz
    Safety = 0.95
    u1 = np.ascontiguousarray(u[:, 0])
    dz = int_fwm.dz * 1
    for jj in range(int_fwm.nplot):
        exitt = False
        while not(exitt):
            # trick to do the first iteration
            delta = 2*int_fwm.maxerr
            while delta > int_fwm.maxerr:
                u1new = ifft(np.exp(Dop*dz/2)*fft(u1))
                A, delta = RK45CK(dAdzmm, u1new, dz, M, int_fwm.n2,
                                  sim_wind.lamda, sim_wind.tsh,
                                  sim_wind.dt, hf, sim_wind.w_tiled)
                if (delta > int_fwm.maxerr):
                    # calculate the step (shorter) to redo
                    dz *= Safety*(int_fwm.maxerr/delta)**0.25
                    #badz += 1
            #####################################Successful step###############
            # propagate the remaining half step

            u1 = ifft(np.exp(Dop*dz/2)*fft(A))
            #goodz +=1
            # update the propagated distance
            dztot += dz
            # update the number of steps taken

            # store the dz just taken
            #dzv = np.append(dzv, dz)
            # calculate the next step (longer)
            # # without exceeding max dzstep
            try:
                dz = np.min(
                    [Safety*dz*(int_fwm.maxerr/delta)**0.2,
                     Safety*int_fwm.dzstep])
            except RuntimeWarning:
                dz = Safety*int_fwm.dzstep
            #dz = 0.95*dz*(int_fwm.maxerr/delta)**0.2
            # print(dz)
            ###################################################################

            if dztot == (int_fwm.dzstep*(jj+1)):
                exitt = True

            elif ((dztot + dz) >= int_fwm.dzstep*(jj+1)):
                dz = int_fwm.dzstep*(jj+1) - dztot
            #dz = np.copy(dz2)
            ###################################################################

        u[:, jj+1] = u1
        U[:, jj+1] = fftshift(fft(u[:, jj+1]))
    int_fwm.dz = dz*1

    return u, U


def dbm_nm(U, sim_wind, int_fwm):
    """
    Converts The units of freequency to units of dBm/nm
    """
    U_out = U / sim_wind.T**2
    U_out = -1*w2dbm(U_out)
    dlv = [sim_wind.lv[i+1] - sim_wind.lv[i]
           for i in range(len(sim_wind.lv) - 1)]
    dlv = np.asanyarray(dlv)
    for i in range(int_fwm.nm):
        U_out[:, i] /= dlv[i]
    return U_out


def fv_creator(lam_p1, lams, int_fwm, prot_casc=0):
    """
    Creates the freequency grid of the simmualtion and returns it.
    The conceprt is that the pump freq is the center. (N/4 - prot_casc) steps till the 
    signal and then (N/4 + prot_casc/2). After wards the rest is filled on the other side of the
    pump wavelength. 

    lam_p1 :: pump wavelength
    lams :: signal wavelength
    int_fwm :: data class with the number of points in
    prot_casc :: a safety to keep the periodic boundary condition away from the first cascade.
                    You can change it to let in more cascades but beware that you are taking 
                    points away from the original pump-signal. 
    """
    #prot_casc = 1024
    N = int_fwm.nt
    fp = 1e-3*c / lam_p1
    fs = 1e-3*c / lams

    sig_pump_shave = N//16
    f_med = np.linspace(fs, fp, sig_pump_shave - prot_casc)
    d = f_med[1] - f_med[0]
    diff = N//4 - sig_pump_shave

    f_2 = [f_med[0], ]
    for i in range(1, N//4 + 1 + diff//2 + prot_casc//2):
        f_2.append(f_2[i-1] - d)
    f_2 = f_2[1:]
    f_2.sort()
    f_1 = [f_med[-1], ]
    for i in range(1, N//2 + 1 + diff//2 + prot_casc//2):
        f_1.append(f_1[i-1] + d)
    f_1 = f_1[1:]
    f_1.sort()
    f_med.sort()

    fv = np.concatenate((f_1, f_med, f_2))
    fv.sort()
    s_pos = np.where(fv == fs)[0][0]
    p_pos = np.where(fv == fp)[0][0]
    where = [p_pos, s_pos]
    check_ft_grid(fv, d)
    return fv, where


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


def check_ft_grid(fv, diff):
    """Grid check for fft optimisation"""
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
