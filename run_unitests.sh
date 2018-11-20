#!/bin/bash
source activate intel

cd src/cython_files
python setup.py build_ext --inplace
cython -a cython_integrand.pyx
cd ../..


pytest code_testing/test*.py
pytest code_testing/mpi_test.py
rm -r .pytest_cache
rm -r .hypothesis
#pytest code_testing/test_pulse_prop.py
#pytest code_testing/unittesting_scripts.py
#pytest code_testing/test_small.py
#pytest --disable-pytest-warnings code_testing/unit_testing_integrand.py 
#pytest code_testing/test_WDM_splice.py