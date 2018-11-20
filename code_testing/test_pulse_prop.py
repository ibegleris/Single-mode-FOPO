import sys
sys.path.append('src')
from functions import *
import numpy as np
from numpy.testing import assert_allclose
def test_half_disp():

    dz = np.random.randn()
    shape1 = 7
    shape2 = 2**12
    u1 = np.random.randn(shape1, shape2) + 1j * np.random.randn(shape1, shape2)
    u1 *= 10
    Dop = np.random.randn(shape1, shape2) + 1j * np.random.randn(shape1, shape2)
    
    u_python = np.fft.ifft(np.exp(Dop*dz/2) * np.fft.fft(u1))
    u_cython = half_disp_step(u1, Dop/2, dz, shape1, shape2)


    assert_allclose(np.asarray(u_cython), u_python)


def test_cython_norm():
    shape1 = 7
    shape2 = 2**12

    A = np.random.randint(0,100)* np.random.randn(shape1, shape2) + np.random.randint(0,100)* 1j * np.random.randn(shape1, shape2)
    cython_norm = np.asarray(norm(A,shape1,shape2))
    python_norm = np.linalg.norm(A,2, axis = -1).max()

    assert_allclose(cython_norm, python_norm)

def test_fftishit():
    shape1 = 7
    shape2 = 2**12
    A = np.random.randn(shape1, shape2) + 1j * np.random.randn(shape1, shape2)

    cython_shift = np.asarray(cyfftshift(A))
    python_shift = np.fft.fftshift(A, axes = -1)
    assert_allclose(cython_shift, python_shift)

def test_fft():
    shape1 = 7
    shape2 = 2**12
    A = np.random.randn(shape1, shape2) + 1j * np.random.randn(shape1, shape2)

    cython_fft = cyfft(A)
    python_fft = np.fft.fft(A)
    assert_allclose(cython_fft, python_fft)

def test_ifft():
    shape1 = 7
    shape2 = 2**12
    A = np.random.randn(shape1, shape2) + 1j * np.random.randn(shape1, shape2)

    cython_fft = cyifft(A)
    python_fft = np.fft.ifft(A)
    assert_allclose(cython_fft, python_fft)


class Test_CK_operators:

    shape1 = 7
    shape2 = 2**12
    u1 = np.random.randn(shape1, shape2) + 1j * np.random.randn(shape1, shape2)
    A1 = np.random.randn(shape1, shape2) + 1j * np.random.randn(shape1, shape2)
    A2 = np.asarray(A2_temp(u1, A1, shape1, shape2))
    
    A3 = np.asarray(A3_temp(u1, A1, A2, shape1,shape2))
    A4 = np.asarray(A4_temp(u1, A1, A2, A3, shape1,shape2))
    A5 = np.asarray(A5_temp(u1, A1, A2, A3, A4, shape1,shape2))
    A6 = np.asarray(A6_temp(u1, A1, A2, A3, A4, A5, shape1,shape2))
    A = np.asarray(A_temp(u1, A1, A3, A4, A6, shape1,shape2))
    Afourth = np.asarray(Afourth_temp(u1, A1, A3, A4, A5, A6, A, shape1,shape2))
    

    def test_A2(self):
        A2_python = self.u1 + (1./5)*self.A1
        assert_allclose(self.A2, A2_python)

    def test_A3(self):
        A3_python = self.u1 + (3./40)*self.A1 + (9./40)*self.A2
        assert_allclose(self.A3, A3_python)

    def test_A4(self):
        A4_python = self.u1 + (3./10)*self.A1 - (9./10)*self.A2 + (6./5)*self.A3
        assert_allclose(self.A4, A4_python)

    def test_A5(self):
        A5_python = self.u1 - (11./54)*self.A1 + (5./2)*self.A2 - (70./27)*self.A3 + (35./27)*self.A4
        assert_allclose(self.A5, A5_python)

    def test_A6(self):
        A6_python = self.u1 + (1631./55296)*self.A1 + (175./512)*self.A2 + (575./13824)*self.A3 +\
                   (44275./110592)*self.A4 + (253./4096)*self.A5
        assert_allclose(self.A6, A6_python)

    def test_A(self):
        A_python = self.u1 + (37./378)*self.A1 + (250./621)*self.A3 + (125./594) * \
                    self.A4 + (512./1771)*self.A6
        assert_allclose(self.A, A_python)

    def test_Afourth(self):
        Afourth_python = self.u1 + (2825./27648)*self.A1 + (18575./48384)*self.A3 + (13525./55296) * \
        self.A4 + (277./14336)*self.A5 + (1./4)*self.A6
        Afourth_python = self.A - Afourth_python
        assert_allclose(self.Afourth, Afourth_python)

def pulse_prop(P_p, betas, ss, lamda_c, lamp, lams, N, z, type='CW'):

    u, U, int_fwm, sim_wind, Dop, non_integrand = \
        wave_setup(P_p, betas, ss, lamda_c, lamp, lams, N, z, type='CW')
    

    factors_xpm, factors_fwm,gama,tsh, w_tiled = \
            non_integrand.factors_xpm, non_integrand.factors_fwm,\
             non_integrand.gama, non_integrand.tsh, non_integrand.w_tiled
    dz,dzstep,maxerr = int_fwm.dz,int_fwm.dzstep,int_fwm.maxerr
    Dop = np.ascontiguousarray(Dop)
    factors_xpm = np.ascontiguousarray(factors_xpm)
    factors_fwm = np.ascontiguousarray(factors_fwm)
    gama = np.ascontiguousarray(gama)
    tsh = np.ascontiguousarray(tsh)
    w_tiled = np.ascontiguousarray(w_tiled)


    u_or, U_or = np.copy(u), np.copy(U)
    U, dz = pulse_propagation(u,dz,dzstep,maxerr, Dop,factors_xpm, factors_fwm, gama,tsh,w_tiled)
    u = np.fft.ifft(np.fft.ifftshift(U, axes = -1))

    return u_or, U_or, u, U


def wave_setup(P_p, betas, ss, lamda_c, lamp, lams, N, z, type='CW'):
    n2 = 2.5e-20
    alphadB = 0
    maxerr = 1e-13
    dz_less = 1e10
    gama = 10e-3
    fr = 0.18
    int_fwm = sim_parameters(n2, 1, alphadB)
    int_fwm.general_options(maxerr, ss)
    int_fwm.propagation_parameters(N, z, 2, dz_less)
    lamda = lamp * 1e-9  # central wavelength of the grid[m]

    M = Q_matrixes(int_fwm.nm, int_fwm.n2, lamda, gama)

    fv, where, f_centrals = fv_creator(
        lamda * 1e9, lams, lamda_c, int_fwm, betas, M, 5,0)
    sim_wind = sim_window(fv, lamda, f_centrals, lamda_c, int_fwm)

    fv, where, f_centrals = fv_creator(
        lamp, lams, lamda_c, int_fwm, betas, M, P_p,0, Df_band=25)
    p_pos, s_pos, i_pos = where
    sim_wind = sim_window(fv, lamda, f_centrals, lamda_c, int_fwm)
    "----------------------------------------------------------"

    "---------------------Loss-in-fibres-----------------------"
    slice_from_edge = (sim_wind.fv[-1] - sim_wind.fv[0]) / 100
    loss = Loss(int_fwm, sim_wind, amax=0)
    int_fwm.alpha = loss.atten_func_full(fv)
    int_fwm.gama = np.array(
        [-1j * n2 * 2 * M * pi * (1e12 * f_c) / (c) for f_c in f_centrals])
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

    keys = ['loading_data/green_dot_fopo/pngs/' +
            str(i) + str('.png') for i in range(7)]
    D_pic = [plt.imread(i) for i in keys]

    ex = Plotter_saver(True, False, sim_wind.fv, sim_wind.t)

    non_integrand = Integrand(int_fwm.gama, sim_wind.tsh,
                              sim_wind.w_tiled, ss,ram, cython_tick=True,
                              timer=False)

    noise_new = noise_obj.noise_func(int_fwm)
    u = np.copy(noise_new)

    if type == 'CW':
        u[3, :] += (P_p)**0.5
        # print(np.max(u))
        u[2, :] += (0.000001)**0.5

    U = fftshift(fft(u), axes=-1)

    return u, U, int_fwm, sim_wind, Dop, non_integrand


class Test_energy_conserve():
    lamda_c = 1051.85e-9
    lamp = 1048
    lams = 1245.98
    betas = np.array([0, 0, 0, 6.756e-2,
                      -1.002e-4, 3.671e-7]) * 1e-3
    N = 10
    P_p = 10
    z = 20

    def test_energy_conserve_s0(self):
        ss = 0
        u_or, U_or, u, U =\
            pulse_prop(self.P_p, self.betas, ss,
                       self.lamda_c, self.lamp, self.lams, self.N, self.z, type='CW')
        E1 = np.sum(np.linalg.norm(u_or, 2, axis = -1)**2)
        E2 = np.sum(np.linalg.norm(u, 2, axis = -1)**2)

        assert_allclose(E1, E2)

    def test_energy_conserve_s1(self):
        ss = 1

        u_or, U_or, u, U =\
            pulse_prop(self.P_p, self.betas, ss,
                       self.lamda_c, self.lamp, self.lams, self.N, self.z, type='CW')
        E1 = np.sum(np.linalg.norm(u_or, 2, axis = -1)**2)
        E2 = np.sum(np.linalg.norm(u, 2, axis = -1)**2)
        assert_allclose(E1, E2)


class Test_cython():
    lamda_c = 1051.85e-9
    lamp = 1048
    lams = 1245.98
    betas = np.array([0, 0, 0, 6.756e-2,
                      -1.002e-4, 3.671e-7]) * 1e-3
    N = 10
    P_p = 10
    z = 20
    dz = 0.01
    def test_s1(self):
        ss = 1
        u, U, int_fwm, sim_wind, Dop, non_integrand = \
            wave_setup(self.P_p, self.betas, ss, self.lamda_c, self.lamp,
                       self.lams, self.N, self.z, type='CW')
        N1 = non_integrand.cython_s1(u, self.dz)
        N2 = non_integrand.python_s1(u, self.dz)
        assert_allclose(N1, N2)

    def test_s0(self):
        ss = 0
        u, U, int_fwm, sim_wind, Dop, non_integrand = \
            wave_setup(self.P_p, self.betas, ss, self.lamda_c, self.lamp,
                       self.lams, self.N, self.z, type='CW')
        N1 = non_integrand.cython_s0(u, self.dz)
        N2 = non_integrand.python_s0(u, self.dz)
        assert_allclose(N1, N2)

    
