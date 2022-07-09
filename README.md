# zhang_annp-gpu-lammps
The package is the implementation of artificial neural newtwork potential (ANNP), which can be accelerated by using GPU card. You can compile these into LAMMPS package according to the following procedures. It can support the CUDA- and OpenCL-enabled GPU card. All the potential parameters are obtained from Dr. H. Mori, but the format are defined by us, as can be see the "fe_annp_potential.ann" file.

Installation:
1. clon this package to your directory
   git clon https://github.com/Zhang-CMS/annp-gpu-lammps.git

2. copy all files in "lib" directory into lammps/lib/gpu directory
   cp ./lal_* ../lammps/lib/gpu;

   (a) set correct value of GPU_ARCH in "Makefile.linux"
   (b) make -f Makefile.linux
   Note: 
   (a) if there is an error about no enough memory for __shared__ memory, please remove the __shared__ numtyp dG_dkx or dG_dky in "lal_annp.cu" file, then try again.
   (b) the "n_Block" in "lal_annp.cpp" file can be changed to make sure that the n_Block*BX/t_per_atom (mostly, BX = 256, t_per_atom = 4) large than cores on your GPU card

3. copy "pair_annp.*" in "src" directory into lammps/src/MANYBODY directory
   cp ./pair_annp.* ../lammps/src/MANYBODY;

4. copy "pair_annp_gpu.*" in "src" directory into lammps/src/GPU directory 
   cp ./pair_annp.* ../lammps/src/GPU;
   add the two "pair_annp*" files into Install.h file in GPU directory

5. make mpi

6. If you want to use OpenCL library, it's better using the cmake to compile lammps by following precedures:
   mkdir build_opencl
   cmake ../cmake -C ../cmake/presets/basic.cmake -D PKG_GPU=on -D GPU_API=opencl -D GPU_PREC=mixed -D GPU_ARCH=sm_61;
   make;
   sudo make install
   Note: the definitation of shared memeory in cuda must be changed into local (__shared__ ----> __local) in the "lal_annp.cu" file  

How to run LAMMPS with ANNP_GPU version
1. the Newton thrid law must be open 
   newton on

2. pair_style	annp
   pair_style	* * fe_annp_potential.ann Fe

Reference:
1. H. Mori, T. Ozaki, Phys Rev Mater 4(4) (2020).
2. N. Artrith and A. Urban, Comput. Mater. Sci. 114, 135 (2016).
3. S. Plimpton, J. Comput. Phys. 117, 1 (1995).

Author & contact information
Author: Meng Zhang，Junya Inoue, Institute of Industral Science, The university of Tokyo, Japan
Email: meng_zhang@metall.t.u-tokyo.ac.jp (M. Zhang), junya_inoue@metall.t.u-tokyo.ac.jp (J. Inoue)
