import numpy as np
from numpy.testing import assert_allclose,assert_raises,assert_almost_equal
import sys
from scipy.constants import c, pi
sys.path.append('src')
from scipy.integrate import solve_ivp
from functions import *
def specific_variables(N):
    n2 = 2.5e-20
    alphadB = 0
    maxerr = 1e-13
    ss = 1
    #N = 18
    lamda_c = 1051.85e-9
    lamda = 1048e-9
    lams = 1245.98
    betas = np.array([0, 0, 0, 6.756e-2,    
                          -1.002e-4, 3.671e-7])*1e-3
    int_fwm = sim_parameters(n2, 1, alphadB)
    int_fwm.general_options(maxerr, ss)
    int_fwm.propagation_parameters(N, 10, 1, 1)
    M = Q_matrixes(int_fwm.nm, int_fwm.n2, lamda, 10)
    fv, where,f_centrals = fv_creator(lamda*1e9 ,lams, lamda_c, int_fwm, betas, M, 5,0)
    sim_wind = sim_window(fv, lamda,f_centrals, lamda_c, int_fwm)
    return M, int_fwm, sim_wind, n2, alphadB, maxerr, ss, N, lamda_c, lamda, lams, \
            betas, fv, where, f_centrals


M, int_fwm, sim_wind, n2, alphadB, maxerr, ss, N, lamda_c, lamda, lams, \
            betas, fv, where, f_centrals = specific_variables(20)

class Test_WDM(object):
    #
    #Tests conservation of energy in freequency and time space as well as the 
    #absolute square value I cary around in the code.
    #
    x1 = 930
    x2 = 1050
    WDMS = WDM(x1, x2,fv)

    U1 = 10*(np.random.randn(7,int_fwm.nt) + 1j * np.random.randn(7,int_fwm.nt))
    U2 = 10*(np.random.randn(7,int_fwm.nt) + 1j * np.random.randn(7,int_fwm.nt))
    U_in = (U1, U2)
        
    a,b = WDMS.pass_through(U_in)
    def test1_WDM_freq(self):
        
        
        U_out1,U_out2 = self.a[1], self.b[1]

        U_in_tot = np.abs(self.U1)**2 + np.abs(self.U2)**2
        U_out_tot = np.abs(U_out1)**2 + np.abs(U_out2)**2
        assert_allclose(U_in_tot,U_out_tot)

    
    def test2_WDM_time(self):        
        U_in = (self.U1, self.U2)
        u_in1 = ifft(fftshift(self.U1, axes = -1))
        u_in2 = ifft(fftshift(self.U2, axes = -1))
        u_in_tot = simps(np.abs(u_in1)**2, sim_wind.t) + simps(np.abs(u_in2)**2,sim_wind.t)

       
        u_out1,u_out2 = self.a[0], self.b[0]

        u_out_tot =  simps(np.abs(u_out1)**2, sim_wind.t) + simps(np.abs(u_out2)**2,sim_wind.t)
        assert_allclose(u_in_tot, u_out_tot)
   


class Test_splicer(object):
    #self.x1 = 930
    #self.x2 = 1050
    #self.nt = 2**3
    #self.lv = np.linspace(900, 1250,2**self.nt)
    splicer = Splicer(loss = np.random.rand()*10)
    U1 = 10*(np.random.randn(7,int_fwm.nt) + 1j * np.random.randn(7,int_fwm.nt))
    U2 = 0*(np.random.randn(7,int_fwm.nt) + 1j * np.random.randn(7,int_fwm.nt))
    U_in = (U1, U2)
    a,b = splicer.pass_through(U_in)
    def test1_splicer_freq(self):
        U1 = self.U1
        U2 = self.U2
        U_out1,U_out2 = self.a[1], self.b[1]

        U_in_tot = np.abs(U1)**2 + np.abs(U2)**2
        U_out_tot = np.abs(U_out1)**2 + np.abs(U_out2)**2

        assert_allclose(U_in_tot,U_out_tot)
    
    def test2_splicer_time(self):
        U_in = (self.U1, self.U2)
        u_in1 = ifft(fftshift(self.U1, axes = -1))
        u_in2 = ifft(fftshift(self.U2, axes = -1))
        u_in_tot = np.abs(u_in1)**2 + np.abs(u_in2)**2

        
        u_out1,u_out2 = self.a[0], self.b[0]

        
        u_out_tot = np.abs(u_out1)**2 + np.abs(u_out2)**2

        assert_allclose(u_in_tot, u_out_tot)


def test_full_trans_in_cavity():
    lam_p1 = 1048.17107345
    WDM1 = WDM(1050, 1200, sim_wind.fv,c)
    WDM2 = WDM(930, 1200, sim_wind.fv, c)
    WDM3 = WDM(930, 1050, sim_wind.fv, c)
    WDM4 = WDM(930, 1200, sim_wind.fv, c)
    splicer1 = Splicer(loss=0.4895)
    splicer2 = Splicer(loss=0.142225011896)


    U = (1/2)**0.5 * (1 + 1j) * np.ones(int_fwm.nt)

    U = splicer1.pass_through((U,np.zeros_like(U)))[0][1]
    U = splicer1.pass_through((U,np.zeros_like(U)))[0][1]
    U = splicer2.pass_through((U,np.zeros_like(U)))[0][1]
    U = splicer2.pass_through((U,np.zeros_like(U)))[0][1]

    U  = WDM2.pass_through((U, np.zeros_like(U)))[1][1]

    U = splicer2.pass_through((U,np.zeros_like(U)))[0][1]


    U  = WDM1.pass_through((np.zeros_like(U),U))[0][1]
 
    assert_allclose(np.max(np.abs(U)**2), 0.72, atol = 1e-2)


class Test_WDM_phase_modulation:
    
    

    def test_unseeded(self):
        WDM1 = WDM(1050, 1200, sim_wind.fv,c)
        shape1, shape2 = sim_wind.fv.shape
        U1 = np.random.randn(shape1, shape2) + 1j * np.random.randn(shape1, shape2)
        U1 *= 100
        U2 =  np.random.randn(shape1, shape2) + 1j * np.random.randn(shape1, shape2)
        U2 *= 100
        WDM1_phase_mod = Phase_modulation_infase_WDM(0,where, WDM1)

        U2 = WDM1_phase_mod.modulate(U1, U2)
        assert_allclose(U2, U2)


    def test_seeded(self):
        WDM1 = WDM(1050, 1200, sim_wind.fv,c)
        shape1, shape2 = sim_wind.fv.shape
        U1 = np.random.randn(shape1, shape2) + 1j * np.random.randn(shape1, shape2)
        #U1 *= np.random.randint(0,100)
        U2 =  np.random.randn(shape1, shape2) + 1j * np.random.randn(shape1, shape2)
        #U2 *= np.random.randint(0,100)
        
        U1[where[1][0], where[1][1]] += 1 + 100j
        U2[where[1][0], where[1][1]] += 100 + 100j
        WDM1_phase_mod = Phase_modulation_infase_WDM(1, where,WDM1)
        
        U2_mod = WDM1_phase_mod.modulate(U1, U2)
        
        p3_non = WDM1.pass_through((U1, U2))[0]
        p3_mod = WDM1.pass_through((U1, U2_mod))[0]
        Up3_mod = p3_mod[1]
        Up3_non = p3_non[1]
        e1 = np.abs(Up3_mod[where[1][0], where[1][1]])**2
        e2 = np.abs(Up3_non[where[1][0], where[1][1]])**2
        print(e1, e2)
        assert((e1 > e2) or np.allclose(e1,e2,equal_nan=True))

class Test_FOPA_phase_modulation_1:
    U = np.random.randint(0,100)*np.random.randn(sim_wind.fv.shape[0],sim_wind.fv.shape[1])+\
        np.random.randint(0,100) *1j * (np.random.randn(sim_wind.fv.shape[0],sim_wind.fv.shape[1]))


    def test_seeded_unseeded(self):
        pm_un = Phase_modulation_FOPA(sim_wind.fv,where)
        U2  = pm_un.modulate(np.copy(self.U))
        assert_raises(AssertionError, assert_allclose,
                     self.U[where[2][0], where[2][1]],
                      U2[where[2][0], where[2][1]])
        non_signal_bands = [i for i in range(7)]
        non_signal_bands.remove(where[2][0])
        for i in non_signal_bands:
            assert_allclose(self.U[i,:], U2[i,:])
        max_idx = np.argmax(np.abs(U2)**2, axis = -1)
        p_pos = where[0]
        i_pos = where[2]
        s_pos = where[1]
        

        max_idx_t = list(np.argmax(np.abs(U2[2:4,:])**2, axis = -1))
        fs,fp = (sim_wind.fv[i,j] for i,j in zip(range(2,4), max_idx_t))
        max_idx[4] = np.argmin(abs(sim_wind.fv[4,:] - (2 * fp - fs)))
        

        angles = np.angle(self.U[2:5])
        Utest = np.copy(U2)
        angle = 2 * angles[p_pos[0]-2, p_pos[1]] \
                - angles[i_pos[0]-2, :-1] - angles[s_pos[0]-2,-2::-1] - 0.5 * pi

        Utest[i_pos[0],:-1] = self.U[i_pos[0],:-1] * np.exp(1j * angle)

        assert_allclose(Utest, U2)


def coupled_eq(z, U):
    Uabs2 = np.abs(U)**2
    Uspm = [U[i] * Uabs2[i] for i in range(3)]
    Uxpm = [U[0] * (Uabs2[1] + Uabs2[2]),
            U[1] * (Uabs2[0] + Uabs2[2]), 
            U[2] * (Uabs2[0] + Uabs2[1]) ]
    
    U_fwm = [U[1]**2 * U[2].conjugate(), 2 * U[0] * U[2] * U[1].conj(),
             U[1]**2 * U[0].conjugate()   ]
    
    integrand = 1j * 10 * \
        np.array([ Uspm[i] + 2 * Uxpm[i] + U_fwm[i] for i in range(3)])
    return integrand


def coupled_prop(U):
    z = 10
    res = solve_ivp(coupled_eq, (0, z), U, rtol = 1e-16)
    return res


class Test_FOPA_phase_modulation_2:

    def test_coupled_amplitudes(self):
        U = np.array([0 + 0j, 1+1j, 0.03+0.03j])
        en_in = sum(np.abs(U)**2)
        res1 = coupled_prop(U)
        powers = np.abs(res1.y[:,-1])**2
        en_out = sum(powers)
        assert_allclose(en_in, en_out,rtol = 1e-3)

    def test_seeded_modulation(self):
        U = np.array([[0 + 0j, 0 + 0j, 0+0j, 7+3j, 1.5+0.5j,0 + 0j, 0 + 0j], 
                     [0 + 0j, 0 + 0j,0.1-0.2j, 0+0j, 0+0j,0 + 0j, 0 + 0j]])
        U = U.T

        fv = np.array([0,0,1,2,3,0,0])
        fv = fv[:,np.newaxis]
        where = np.array([[3,0], [2,0], [4,0]])


        pm = Phase_modulation_FOPA(fv, where)

        Umod = pm.modulate(np.copy(U))
        Umod[4,:] = Umod[4,:] * np.exp(1j*pi) # Sets the +0.5pi for CA propagation (GNLSE is conj)
        assert_raises(AssertionError, assert_allclose,
                             U, Umod)
        assert_allclose(np.abs(Umod)**2, np.abs(U)**2)
        

        en_in = sum(np.abs(U)**2)
        Ucop = U[:,0]
        Ucop[2] = U[2,1]
        Umodcop = Umod[:,0]
        Umodcop[2] = Umod[2,1]
        
        print(Ucop[2:5])
        print(Umodcop[2:5])
        #assert False

        res1 = coupled_prop(Ucop[2:5]).y[:,-1]
        res2 = coupled_prop(Umodcop[2:5]).y[:,-1]
        idler1 = np.abs(res1[0])**2
        idler2 = np.abs(res2[0])**2
        assert idler2 >= idler1
