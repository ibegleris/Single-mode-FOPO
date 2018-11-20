# SM-FOPO
[![Build Status](https://travis-ci.com/ibegleris/strict-SM-FOPO_dev.svg?token=UafEdqSJuFtM7z2nYK1k&branch=master)](https://travis-ci.com/ibegleris/strict-SM-FOPO_dev)

The repository holds the model of a unidirectional single mode FOPO based upon the system.

Updates in Version 2.0:
* Introduction of the 7-band Banded Generalised Nonlinear Schrodinger equation for pulse propagation
* Introduction of the Phase-modulation at every round trip for resonance to be reached
* Use of Cython Memoryviews and Intel MKL libraries for compilation




* Requirements:
  * Tested on Ubuntu Xenian. 
  * Python 3.6 tested
  * mpi4py futures
  * The Conda Intel Python distribution found [here](https://software.intel.com/en-us/articles/using-intel-distribution-for-python-with-anaconda)
  * MKL libraries for compilation of the Cython code [here](https://software.intel.com/en-us/articles/intel-math-kernel-library-intel-mkl-2018-install-guide)

Installation:

	bash build_install.sh

Note: Pass 'cluster' as an argument if you do not have sudo. sudo is used to install mpich for multinode computations. 
	
Testing (unit-tests):

	source activate intel && pytest code_testing/test*.py

Execution of FOPO:

	bash run_oscilator.sh rounds cores library && bash run_posprocess.sh
Note:
* rounds: Number of round trips 
* cores: Number of cores to use
* library: mpi (Code uses mpi4py, needed for multiple nodes on cluster), joblib: Multicore computations, MKL: Runs with MKL libraries 

Example: bash run_oscilator.sh 1000 4 mpi; will run the 
oscillator for 1000 rounds over 4 cores using mpi4py.  




There is now a build script that downloads and installs miniconda with all the equilavent packadges needed. Beware to have sudo now for MPICH if not already installed.

In particular the repository has been branched to create a more efficient way to do the pulse propagation within the fibre since the grid cannot resolve all waves at the moment. 
