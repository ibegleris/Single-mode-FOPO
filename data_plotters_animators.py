import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.constants import c
import h5py

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

plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)


def plotter_dbm(index, nm, sim_wind, u, U, P0_p, P0_s, f_p, f_s, which,ro, pump_wave = '',filename=None, title=None, im=0, plots = True):
	#u, U = np.reshape(u,(np.shape(u)[-1], np.shape(u)[0])), np.reshape(u,(np.shape(U)[-1], np.shape(U)[0]))
	if plots == True:
		fig = plt.figure(figsize=(20.0, 10.0))

		plt.plot(1e-3*c/sim_wind.fv, 
				w2dbm(np.abs(U[:,which])**2), '-*')
		#plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
		#plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
		plt.xlabel(r'$\lambda (nm)$', fontsize=18)
		plt.ylabel(r'$Spectrum (a.u.)$', fontsize=18)
		plt.ylim([-80, 100])
		plt.xlim([np.min(sim_wind.lv), np.max(sim_wind.lv)])
		plt.xlim([900, 1250])
		plt.title(title)
		plt.grid()
		if type(im) != int:
			newax = fig.add_axes([0.8, 0.8, 0.2, 0.2], anchor='NE')
			newax.imshow(im)
			newax.axis('off')
		if filename == None:
			plt.show()
		else:
			plt.savefig('output'+pump_wave+'/output'+str(index)+'/figures/wavelength/'+filename, bbox_inched='tight')

		plt.close(fig)

		fig = plt.figure(figsize=(20.0, 10.0))
		plt.plot(sim_wind.fv, w2dbm(np.abs(U[:,which])**2), '-*')
		#plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
		#plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
		plt.xlabel(r'$f (THz)$', fontsize=18)
		plt.ylabel(r'$Spectrum (a.u.)$', fontsize=18)
		#plt.xlim([np.min(sim_wind.fv), np.max(sim_wind.fv)])
		plt.ylim([-20, 120])
		#plt.xlim(270,300)
		plt.title(str(f_p)+' ' +str(f_s))
		plt.grid()
		if type(im) != int:
			newax = fig.add_axes([0.8, 0.8, 0.2, 0.2], anchor='NE')
			newax.imshow(im)
			newax.axis('off')
		if filename == None:
			plt.show()
		else:
			plt.savefig('output'+pump_wave+'/output'+str(index)+'/figures/freequency/'+filename, bbox_inched='tight')
		plt.close(fig)

		fig = plt.figure(figsize=(20.0, 10.0))
		
		plt.plot(sim_wind.t,np.abs(u[:, which])**2, '*-')
		#plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
		plt.title('time space')
		#plt.ylim([0, 160])
		plt.grid()
		plt.xlabel(r'$t(ps)$')
		plt.ylabel(r'$Spectrum$')
		if type(im) != int:
			newax = fig.add_axes([0.8, 0.8, 0.2, 0.2], anchor='NE')
			newax.imshow(im)
			newax.axis('off')

		if filename == None:
			plt.show()
		else:

			plt.savefig('output'+pump_wave+'/output'+str(index)+'/figures/time/'+filename)
		plt.close(fig)
			

	if filename is not(None):
		if filename[:4] != 'port': 
			layer = filename[-1]+'/'+filename[:-1]
		else:
			layer = filename
		try:
			
			save_variables('data_large', layer, filepath='output'+pump_wave+'/output'+str(index)+'/data/', U = U[:,which], t=sim_wind.t, u=u[:,which],
						   fv=sim_wind.fv, lv=sim_wind.lv,
						   which=which, nm=nm, P0_p=P0_p, P0_s=P0_s, f_p=f_p, f_s=f_s, ro = ro)
		except RuntimeError:
			os.system('rm output'+pump_wave+'/output'+str(index)+'/data/data_large.hdf5')
			save_variables('data_large', layer, filepath='output'+pump_wave+'/output'+str(index)+'/data/', U=U[:,which], t=sim_wind.t, u=u[:,which],
						   fv=sim_wind.fv, lv=sim_wind.lv,
						   which=which, nm=nm, P0_p=P0_p, P0_s=P0_s, f_p=f_p, f_s=f_s, ro = ro)
			pass

	return 0


def plotter_dbm_load():
	# class sim_window(object):
	plotter_dbm(nm, sim_wind, Uabs, u, which)
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
