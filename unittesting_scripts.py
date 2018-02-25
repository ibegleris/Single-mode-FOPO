from math import factorial
from functions import *
import pytest
from scipy.fftpack import fft, ifft, fftshift, ifftshift
from fft_module import *
fft, ifft, method = pick(10, 1, 100, 1)
print('method for ffts', method)
import numpy as np
from scipy.io import loadmat
from numpy.testing import\
    assert_allclose,\
    assert_approx_equal,\
    assert_almost_equal,\
    assert_raises
from scipy.interpolate import InterpolatedUnivariateSpline
from data_plotters_animators import *
import matplotlib.pyplot as plt
from scipy.integrate import simps
import warnings
warnings.filterwarnings("ignore")
"---------------------------------W and dbm conversion tests--------------"


def test_mpi4py_futures():
    import mpi4py.futures


def test_dbm2w():
    assert dbm2w(30) == 1


def test1_w2dbm():
    assert w2dbm(1) == 30


def test2_w2dbm():
    a = np.zeros(100)
    floor = np.random.rand(1)[0]
    assert_allclose(w2dbm(a, -floor), -floor*np.ones(len(a)))


def test3_w2dbm():
    with pytest.raises(ZeroDivisionError):
        w2dbm(-1)


def FWHM_fun(X, Y):
    half_max = np.max(Y) / 2.
    # find when function crosses line half_max (when sign of diff flips)
    # take the 'derivative' of signum(half_max - Y[])
    d = np.sign(half_max - np.array(Y[0:-1])) - \
        np.sign(half_max - np.array(Y[1:]))
    # plot(X,d) #if you are interested
    # find the left and right most indexes
    left_idx = np.where(d > 0)[0]
    right_idx = np.where(d < 0)[-1]
    return X[right_idx] - X[left_idx]  # return the difference (full width)
"------------------------------------------------------fft test--------------"
try:
    fft, ifft, method = pick(10, 1, 100, 1)
    print('method for ffts', method)
    import accelerate.mkl.fftpack as mklfft
    mfft, imfft = mklfft.fft, mklfft.ifft

    def test_fft():
        x = np.random.rand(11, 10)
        assert_allclose(mfft(x), fft(x))

    def test_ifft():
        x = np.random.rand(10, 10)
        assert_allclose(imfft(x), imfft(x))
except ImportError:
    pass

"--------------------------------------------Raman response--------------"


def test_raman_off():
    ram = raman_object('off')
    ram.raman_load(np.random.rand(10), np.random.rand(1)[0])
    assert ram.hf == None


def test_raman_load():
    ram = raman_object('on', 'load')
    D = loadmat('testing_data/Raman_measured.mat')
    t = D['t']
    t = np.asanyarray([t[i][0] for i in range(t.shape[0])])
    dt = D['dt'][0][0]
    hf_exact = D['hf']
    hf_exact = np.asanyarray([hf_exact[i][0]
                              for i in range(hf_exact.shape[0])])
    hf = ram.raman_load(t, dt)

    assert_allclose(hf, hf_exact)


def test_raman_analytic():
    ram = raman_object('on', 'analytic')
    D = loadmat('testing_data/Raman_analytic.mat')
    t = D['t']
    t = np.asanyarray([t[i][0] for i in range(t.shape[0])])
    dt = D['dt'][0][0]
    hf_exact = D['hf']
    hf_exact = np.asanyarray([hf_exact[i][0]
                              for i in range(hf_exact.shape[0])])
    hf = ram.raman_load(t, dt)

    assert_allclose(hf, hf_exact)


"----------------------------Dispersion operator--------------"


class int_fwms(object):

    def __init__(self, nm, alpha, nt):
        self.nm = nm
        self.alphadB = alpha
        self.alpha = self.alphadB/4.343
        self.nt = nt


class sim_windows(object):

    def __init__(self, lamda, lv, lmin, lmax, nt):
        self.lamda = lamda
        self.lmax, self.lmin = lmax, lmin
        self.w = 2*pi*c/self.lamda*1e-12
        self.lv = lv
        self.nt = 512
        self.fv = 1e-3*c/self.lv
        self.fmed = 0.5*(max(self.fv) + min(self.fv))*1e12
        self.woffset = 2*pi*(self.fmed - c/lamda)*1e-12
        self.deltaf = 1e-3*(c/self.lmin - c/self.lmax)
        self.df = self.deltaf/len(lv)
        self.T = 1/self.df
        self.dt = self.T/len(lv)
        self.t = (range(nt)-np.ones(nt)*nt/2)*self.dt


def test_dispersion():
    nt = 512
    lmin, lmax = 1000e-9, 2000e-9
    lamda = np.linspace(lmin, lmax, nt)

    lamda0 = 1500e-9
    lamdac = 1550e-9

    sim_wind = sim_windows(lamda0, lamda, lmin, lmax, nt)
    int_fwm = int_fwms(1, 0.1, nt)

    loss = Loss(int_fwm, sim_wind, amax=0.1)
    alpha_func = loss.atten_func_full(sim_wind.fv)
    int_fwm.alphadB = alpha_func
    int_fwm.alpha = int_fwm.alphadB

    betas = np.array([0, 0, 0, 6.755e-2, -1.001e-4])*1e-3

    betas_disp = dispersion_operator(betas, lamdac, int_fwm, sim_wind)

    betas_exact = np.loadtxt('testing_data/exact_dispersion.txt').view(complex)
    assert_allclose(betas_disp, betas_exact)


def pulse_propagations(ram, ss, N_sol=1):
    "SOLITON TEST. IF THIS FAILS GOD HELP YOU!"

    n2 = 2.5e-20                                # n2 for silica [m/W]
    nm = 1                                  # number of modes
    alphadB = 0  # 0.0011666666666666668             # loss [dB/m]
    gama = 1e-3                                 # w/m

    maxerr = 1e-13              # maximum tolerable error per step

    N = 14
    z = 70                  # total distance [m]
    nplot = 1                 # number of plots
    nt = 2**N                   # number of grid points
    dzstep = z/nplot            # distance per step
    dz_less = 1e30
    dz = dzstep/dz_less      # starting guess value of the step

    lam_p1 = 835
    lamda_c = 835e-9
    lamda = lam_p1*1e-9

    beta2 = -1e-3
    P0_p1 = 1

    T0 = (N_sol**2 * np.abs(beta2) / (gama * P0_p1))**0.5
    TFWHM = (2*np.log(1+2**0.5)) * T0
    print(TFWHM)

    int_fwm = sim_parameters(n2, nm, alphadB)
    int_fwm.general_options(maxerr, raman_object, ss, ram)
    int_fwm.propagation_parameters(N, z, nplot, dz_less)

    fv, where = fv_creator(lam_p1 - 25, lam_p1, int_fwm)
    sim_wind = sim_window(fv, lamda, lamda_c, int_fwm, fv_idler_int=1)

    loss = Loss(int_fwm, sim_wind, amax=int_fwm.alphadB)
    alpha_func = loss.atten_func_full(sim_wind.fv)
    int_fwm.alphadB = alpha_func
    int_fwm.alpha = int_fwm.alphadB
    betas = np.array([0, 0, beta2])  # betas at ps/m
    Dop = dispersion_operator(betas, lamda_c, int_fwm, sim_wind)

    string = "dAdzmm_r"+str(ram)+"_s"+str(ss)
    func_dict = {'dAdzmm_ron_s1': dAdzmm_ron_s1,
                 'dAdzmm_ron_s0': dAdzmm_ron_s0,
                 'dAdzmm_roff_s0': dAdzmm_roff_s0,
                 'dAdzmm_roff_s1': dAdzmm_roff_s1}
    pulse_pos_dict_or = ('after propagation', "pass WDM2",
                         "pass WDM1 on port2 (remove pump)",
                         'add more pump', 'out')

    dAdzmm = func_dict[string]

    M = Q_matrixes(1, n2, lamda, gama=gama)
    raman = raman_object(int_fwm.ram, int_fwm.how)
    raman.raman_load(sim_wind.t, sim_wind.dt)

    if raman.on == 'on':
        hf = raman.hf
    else:
        hf = None

    u = np.zeros([len(sim_wind.t), len(sim_wind.zv)], dtype='complex128')
    U = np.zeros([len(sim_wind.t), len(sim_wind.zv)], dtype='complex128')

    sim_wind.w_tiled = sim_wind.w
    print(sim_wind.woffset)

    u[:, 0] = ((P0_p1)**0.5 / np.cosh(sim_wind.t/T0)) * \
        np.exp(-1j*(sim_wind.woffset)*sim_wind.t)
    U[:, 0] = fftshift(sim_wind.dt*fft(u[:, 0]))
    fwhm2 = FWHM_fun(sim_wind.t, np.abs(u)**2)
    u, U = pulse_propagation(u, U, int_fwm, M, sim_wind, hf, Dop, dAdzmm)

    U_start = np.abs(U[:, 0])**2

    u[:, -1] = u[:, -1]*np.exp(1j*z/2)*np.exp(-1j *
                                              (sim_wind.woffset)*sim_wind.t)
    """
    fig1 = plt.figure()
    plt.plot(sim_wind.fv,np.abs(U[:,0])**2)
    plt.savefig('1.png')

    fig2 = plt.figure()
    plt.plot(sim_wind.fv,np.abs(U[:,-1])**2)
    plt.savefig('2.png')    
    
    
    fig3 = plt.figure()
    plt.plot(sim_wind.t,np.abs(u[:,0])**2)
    plt.xlim(-10*T0, 10*T0)
    plt.savefig('3.png')

    fig4 = plt.figure()
    plt.plot(sim_wind.t,np.abs(u[:,-1])**2)
    plt.xlim(-10*T0, 10*T0)
    plt.savefig('4.png')    
    

    fig5 = plt.figure()
    plt.plot(fftshift(sim_wind.w),(np.abs(U[:,-1])**2 - np.abs(U[:,0])**2 ))
    plt.savefig('error.png')

    
    fig6 = plt.figure()
    plt.plot(sim_wind.t,np.abs(u[:,-1])**2 - np.abs(u[:,0])**2)
    plt.xlim(-10*T0, 10*T0)
    plt.savefig('error2.png')
    plt.show()
    """
    return u, U, maxerr


def test_solit_r0_ss0():
    u, U, maxerr = pulse_propagations('off', 0)
    print(np.linalg.norm(np.abs(u[:, 0])**2 - np.abs(u[:, -1])**2, 2))

    assert_allclose(np.abs(u[:, 0])**2, np.abs(u[:, -1])**2, atol=9e-4)


def test_energy_r0_ss0():
    u, U, maxerr = pulse_propagations(
        'off', 0, N_sol=np.abs(10*np.random.randn()))
    E = []
    for i in range(np.shape(u)[1]):
        E.append(np.linalg.norm(u[:, i], 2)**2)
    assert np.all(x == E[0] for x in E)


def test_energy_r0_ss1():
    u, U, maxerr = pulse_propagations(
        'off', 1, N_sol=np.abs(10*np.random.randn()))
    E = []
    for i in range(np.shape(u)[1]):
        E.append(np.linalg.norm(u[:, i], 2)**2)

    assert np.all(x == E[0] for x in E)


def test_energy_r1_ss0():
    u, U, maxerr = pulse_propagations(
        'on', 0, N_sol=np.abs(10*np.random.randn()))
    E = []
    for i in range(np.shape(u)[1]):
        E.append(np.linalg.norm(u[:, i], 2)**2)
    assert np.all(x == E[0] for x in E)


def test_energy_r1_ss1():
    u, U, maxerr = pulse_propagations(
        'on', 1, N_sol=np.abs(10*np.random.randn()))
    E = []
    for i in range(np.shape(u)[1]):
        E.append(np.linalg.norm(u[:, i], 2)**2)
    assert np.all(x == E[0] for x in E)


def test_time_frequency():
    nt = 3
    dt = np.abs(np.random.rand())*10
    u1 = 10*(np.random.randn(2**nt) + 1j * np.random.randn(2**nt))
    U = fftshift(dt*fft(u1))
    u2 = ifft(ifftshift(U)/dt)
    assert_allclose(u1, u2)

"-------------------------------WDM------------------------------------"


class Test_WDM(object):
    """
    Tests conservation of energy in freequency and time space as well as the 
    absolute square value I cary around in the code.
    """

    def test1_WDM_freq(self):
        self.x1 = 950
        self.x2 = 1050
        self.nt = 2**10
        l1, l2 = 900, 1250
        f1, f2 = 1e-3 * c / l1, 1e-3 * c / l2

        self.fv = np.linspace(f1, f2, self.nt)
        self.lv = 1e3 * c / self.fv

        lamda = (self.lv[-1] + self.lv[0])/2
        sim_wind = sim_windows(lamda, self.lv, 900, 1250, self.nt)
        WDMS = WDM(self.x1, self.x2, self.fv, c)

        U1 = 10*(np.random.randn(self.nt) + 1j * np.random.randn(self.nt))
        U2 = 0 * (np.random.randn(self.nt) + 1j * np.random.randn(self.nt))
        U_in = (U1, U2)

        a, b = WDMS.pass_through(U_in, sim_wind)
        U_out1, U_out2 = a[1], b[1]

        U_in_tot = np.abs(U1)**2 + np.abs(U2)**2
        U_out_tot = np.abs(U_out1)**2 + np.abs(U_out2)**2
        assert_allclose(U_in_tot, U_out_tot)

    def test2_WDM_time(self):
        self.x1 = 950
        self.x2 = 1050
        self.nt = 2**24
        l1, l2 = 900, 1250
        f1, f2 = 1e-3 * c / l1, 1e-3 * c / l2

        self.fv = np.linspace(f1, f2, self.nt)
        self.lv = 1e3 * c / self.fv

        lamda = (self.lv[-1] + self.lv[0])/2
        sim_wind = sim_windows(lamda, self.lv, 900, 1250, self.nt)
        WDMS = WDM(self.x1, self.x2, self.fv, c)

        U1 = 10*(np.random.randn(self.nt) + 1j * np.random.randn(self.nt))
        U2 = 10*(np.random.randn(self.nt) + 1j * np.random.randn(self.nt))
        U_in = (U1, U2)

        u_in1 = ifft(fftshift(U1))
        u_in2 = ifft(fftshift(U2))
        u_in_tot = simps(np.abs(u_in1)**2, sim_wind.t) + \
            simps(np.abs(u_in2)**2, sim_wind.t)

        a, b = WDMS.pass_through(U_in, sim_wind)
        u_out1, u_out2 = a[0], b[0]

        u_out_tot = simps(np.abs(u_out1)**2, sim_wind.t) + \
            simps(np.abs(u_out2)**2, sim_wind.t)
        assert_allclose(u_in_tot, u_out_tot)


class int_fwmss(object):

    def __init__(self, alphadB):
        self.alphadB = alphadB


class sim_windowss(object):

    def __init__(self, fv):
        self.fv = fv


class Test_loss:

    def test_loss1(a):
        fv = np.linspace(200, 600, 1024)
        alphadB = 1
        sim_wind = sim_windowss(fv)
        int_fwm = int_fwmss(alphadB)
        loss = Loss(int_fwm, sim_wind, amax=alphadB)
        alpha_func = loss.atten_func_full(sim_wind.fv)
        assert_allclose(alpha_func, np.ones_like(alpha_func)*alphadB/4.343)

    def test_loss2(a):
        fv = np.linspace(200, 600, 1024)
        alphadB = 1
        sim_wind = sim_windowss(fv)
        int_fwm = int_fwmss(alphadB)
        loss = Loss(int_fwm, sim_wind, amax=2*alphadB)
        alpha_func = loss.atten_func_full(sim_wind.fv)
        maxim = np.max(alpha_func)
        assert maxim == 2*alphadB/4.343

    def test_loss3(a):
        fv = np.linspace(200, 600, 1024)
        alphadB = 1
        sim_wind = sim_windowss(fv)
        int_fwm = int_fwmss(alphadB)
        loss = Loss(int_fwm, sim_wind, amax=2*alphadB)
        alpha_func = loss.atten_func_full(sim_wind.fv)
        minim = np.min(alpha_func)
        assert minim == alphadB/4.343


class Test_splicer():

    def test1_splicer_freq(self):
        self.x1 = 930
        self.x2 = 1050
        self.nt = 2**3

        self.lv = np.linspace(900, 1250, 2**self.nt)
        lamda = (self.lv[-1] + self.lv[0])/2
        sim_wind = sim_windows(lamda, self.lv, 900, 1250, self.nt)
        splicer = Splicer(loss=np.random.rand()*10)

        U1 = 10*(np.random.randn(2**self.nt) +
                 1j * np.random.randn(2**self.nt))
        U2 = 10 * (np.random.randn(2**self.nt) +
                   1j * np.random.randn(2**self.nt))

        U_in = (U1, U2)
        U1 = U1[:, np.newaxis]
        U2 = U2[:, np.newaxis]
        a, b = splicer.pass_through(U_in, sim_wind)
        U_out1, U_out2 = a[1], b[1]

        U_in_tot = np.abs(U1)**2 + np.abs(U2)**2
        U_out_tot = np.abs(U_out1)**2 + np.abs(U_out2)**2

        assert_allclose(U_in_tot[:, 0], U_out_tot)

    def test2_splicer_time(self):
        self.x1 = 930
        self.x2 = 1050
        self.nt = 2**3

        self.lv = np.linspace(900, 1250, 2**self.nt)
        lamda = (self.lv[-1] + self.lv[0])/2
        sim_wind = sim_windows(lamda, self.lv, 900, 1250, self.nt)
        splicer = Splicer(loss=np.random.rand()*10)

        U1 = 10*(np.random.randn(2**self.nt) +
                 1j * np.random.randn(2**self.nt))
        U2 = 10 * (np.random.randn(2**self.nt) +
                   1j * np.random.randn(2**self.nt))

        U_in = (U1, U2)
        U1 = U1  # [:,np.newaxis]
        U2 = U2  # [:,np.newaxis]
        u_in1 = ifft(ifftshift(U1))
        u_in2 = ifft(ifftshift(U2))
        u_in_tot = np.abs(u_in1)**2 + np.abs(u_in2)**2

        a, b = splicer.pass_through(U_in, sim_wind)
        u_out1, u_out2 = a[0], b[0]

        u_out_tot = np.abs(u_out1)**2 + np.abs(u_out2)**2

        assert_allclose(u_in_tot, u_out_tot)


def test_read_write1():
    #os.system('rm testing_data/hh51_test.hdf5')
    A = np.random.rand(10, 3, 5) + 1j * np.random.rand(10, 3, 5)
    B = np.random.rand(10)
    C = 1
    save_variables('hh51_test', '0', filepath='testing_data/',
                   A=A, B=B, C=C)
    A_copy, B_copy, C_copy = np.copy(A), np.copy(B), np.copy(C)
    del A, B, C
    D = read_variables('hh51_test', '0', filepath='testing_data/')

    A, B, C = D['A'], D['B'], D['C']
    os.system('rm testing_data/hh51_test.hdf5')
    assert_allclose(A, A_copy)


def test_read_write2():

    #os.system('rm testing_data/hh52_test.hdf5')
    A = np.random.rand(10, 3, 5) + 1j * np.random.rand(10, 3, 5)
    B = np.random.rand(10)
    C = 1
    save_variables('hh52_test', '0', filepath='testing_data/',
                   A=A, B=B, C=C)
    A_copy, B_copy, C_copy = np.copy(A), np.copy(B), np.copy(C)
    del A, B, C
    D = read_variables('hh52_test', '0', filepath='testing_data/')
    A, B, C = D['A'], D['B'], D['C']
    # locals().update(D)
    os.system('rm testing_data/hh52_test.hdf5')
    return None


def test_read_write3():

    A = np.random.rand(10, 3, 5) + 1j * np.random.rand(10, 3, 5)
    B = np.random.rand(10)
    C = 1
    save_variables('hh53_test', '0', filepath='testing_data/',
                   A=A, B=B, C=C)
    A_copy, B_copy, C_copy = np.copy(A), np.copy(B), np.copy(C)
    del A, B, C
    D = read_variables('hh53_test', '0', filepath='testing_data/')
    A, B, C = D['A'], D['B'], D['C']
    os.system('rm testing_data/hh53_test.hdf5')
    assert C == C_copy
    return None


def test_fv_creator():
    """
    Checks whether the first order cascade is in the freequency window.
    """
    class int_fwm1(object):

        def __init__(self):
            self.N = 14
            self.nt = 2**self.N

    int_fwm = int_fwm1()
    lam_p1 = 1000
    lam_s = 1200
    fv, where = fv_creator(lam_p1, lam_s, int_fwm)
    mins = np.min(1e-3*c/fv)
    f1 = 1e-3*c/lam_p1
    fs = 1e-3*c/lam_s
    diff = abs(f1 - fs)
    assert(all(i < max(fv) and i > min(fv)
               for i in (f1, fs, fs + diff, f1 - diff, f1 - 2*diff)))


def test_noise():
    class sim_windows(object):

        def __init__(self):
            self.w = 10
            self.T = 0.1
            self.w0 = 9
    class int_fwms(object):

        def __init__(self):
            self.nt = 1024
            self.nm = 1
    int_fwm = int_fwms()
    sim_wind = sim_windows()
    noise = Noise(int_fwm, sim_wind)
    n1 = noise.noise_func(int_fwm)
    n2 = noise.noise_func(int_fwm)
    print(n1, n2)
    assert_raises(AssertionError, assert_almost_equal, n1, n2)


def test_full_trans_in_cavity():
    N = 12
    nt = 2**N
    #fft,ifft,method = pick(nt, 1,100, 1)
    from scipy.constants import c, pi
    int_fwm = sim_parameters(2.5e-20, 1, 0)
    int_fwm.general_options(1e-6, raman_object, 0, 0)
    int_fwm.propagation_parameters(N, 18, 1, 1)

    lam_p1 = 1048.17107345
    fv, where = fv_creator(850, lam_p1, int_fwm)
    lv = 1e-3*c/fv
    sim_wind = sim_window(fv, lam_p1, lam_p1, int_fwm, 0)
    noise_obj = Noise(int_fwm, sim_wind)
    print(fv)
    WDM1 = WDM(1050, 1200, sim_wind.fv, c)
    WDM2 = WDM(930, 1200, sim_wind.fv, c)
    WDM3 = WDM(930, 1050, sim_wind.fv, c)
    WDM4 = WDM(930, 1200, sim_wind.fv, c)
    splicer1 = Splicer(loss=0.4895)
    splicer2 = Splicer(loss=0.142225011896)

    U = (1/2)**0.5 * (1 + 1j) * np.ones(nt)

    U = splicer1.pass_through((U, np.zeros_like(U)), sim_wind)[0][1]
    U = splicer1.pass_through((U, np.zeros_like(U)), sim_wind)[0][1]
    U = splicer2.pass_through((U, np.zeros_like(U)), sim_wind)[0][1]
    U = splicer2.pass_through((U, np.zeros_like(U)), sim_wind)[0][1]

    U = WDM2.pass_through((U, np.zeros_like(U)), sim_wind)[1][1]

    U = splicer2.pass_through((U, np.zeros_like(U)), sim_wind)[0][1]

    U = WDM1.pass_through((np.zeros_like(U), U), sim_wind)[0][1]

    assert_allclose(max(np.abs(U)**2), 0.7234722042243035)
