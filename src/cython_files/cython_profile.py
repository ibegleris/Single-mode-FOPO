import pstats, cProfile
import cython_integrand
import pickle
import os
with open('../../loading_data/cython_prop.pickl', 'rb') as f:
	D = pickle.load(f)

names = ('u','dz','dzstep','maxerr', 'Dop','factors_xpm', 'factors_fwm', 'gama','tsh','w_tiled') 
u,dz,dzstep,maxerr, Dop,factors_xpm, factors_fwm, gama,tsh,w_tiled = [D[i] for i in names]

cProfile.runctx("cython_integrand.pulse_propagation(u,dz,dzstep,maxerr, Dop,factors_xpm, factors_fwm, gama,tsh,w_tiled)", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
os.system('rm Profile.prof')