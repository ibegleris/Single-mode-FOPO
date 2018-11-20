#!/bin/bash
#This bash file handles the running of the oscillator code.
#Example chmod +X run_oscilator.sh; ./run_oscilator.sh 1000 4 mpi; will run the 
#oscillator for 1000 rounds over 4 cores using mpi.  
source activate intel
echo 'starting...'
rm -r output*
rm -r *src/__*
cd src/cython_files
#rm -rf build *so *c *html

python setup.py build_ext --inplace
cython -a cython_integrand.pyx
cd ../..
rm error_log
if [ "$3" == "mpi" ]
then
	export MKL_NUM_THREADS=1
	echo "running with" 1 "MKL core and" $2 "MPI cores, for" $1 "rounds"
	mpiexec -n $2 python -m mpi4py.futures src/oscillator.py $3 $1 $2 0
elif [ "$3" == "joblib" ]
then
	export MKL_NUM_THREADS=1
	echo "running with" 1 "MKL core and" $2 "Multiprocessing cores, for" $1 "rounds"
	python src/oscillator.py $3 $1 $2 0
else
	echo "running with MKL cores for" $1 "rounds" 
	python src/oscillator.py $3 $1 1 0
fi
