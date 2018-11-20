#!/bin/bash
echo 'starting...'
rm -r output*
rm -r *__*
rm error_log
cd src/cython_files
python setup.py build_ext --inplace
cython -a cython_integrand.pyx
cd ../..
source activate intel
export MKL_NUM_THREADS=1
kernprof -l -v src/oscillator.py single 10 1 1
rm oscillator.py.lprof