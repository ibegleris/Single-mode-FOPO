wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
rm minconda.sh
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update conda -y
conda config --add channels intel
conda create -n intel intelpython3_core python=3
source activate intel
conda install cython numpy scipy matplotlib pandas h5py pytables jupyter joblib numba pytest nose -y
source deactivate
rm -rf ../.condarc
mv build_data/.condarc_default ../.condarc
mv build_data/.condarc_default ~/miniconda/envs/intel/.condarc
conda update conda -y
conda install numpy scipy matplotlib jupyter pandas h5py pytables jupyter numba pytest nose -y
conda install python=3.6 -y
if [ "$1" != 'cluster' ]; then
	sudo apt-get update
	sudo apt-get install mpich -y 
fi
echo 'export PATH="/home/$USER/miniconda/bin:$PATH"' >> ~/.bashrc
source activate intel
git clone https://github.com/mpi4py/mpi4py.git
cd mpi4py
python setup.py build
python setup.py install
cd ..
rm -rf mpi4py
pytest unittesting_scripts.py
