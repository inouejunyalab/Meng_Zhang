## This is new GPU implementation for ANNP of Fe (References 1, 2, and 3), which has a 12-14% speedup increasing than the previous one, as can be seen from performance test. You can compile this into LAMMPS (Reference 4) according to the procedures in "fe" file.

## Update:
1) removed redundant computation of symmetry functions 
2) modified the force updating for atom i

## Installation:
1) download all files in "lib", "src" directors and the potential file  

2) copy all files in the "lib" directory into lammps/lib/gpu directory \
   cp ./lal_* &#8195; lammps_PATH/lib/gpu
   
   Note: \
   (a) set correct value of GPU_ARCH in "Makefile.linux" \
   (b) make -f Makefile.linux \
   (c) the "n_Block" in "lal_annp.cpp" file can be changed to make sure that the n_Block*BX/t_per_atom (mostly, BX = 256, t_per_atom = 4) large than cores on your GPU card

3) copy "pair_annp.h and pair_annp.cpp" in the "src" directory into lammps/src/MANYBODY directory \
   cp ./pair_annp.h &#8195; &#8194; lammps_PATH/src/MANYBODY \
   cp ./pair_annp.cpp &#8195; lammps_PATH/src/MANYBODY 
4) copy "pair_annp_gpu.h and pair_annp_gpu.cpp" in the "src" directory into lammps/src/GPU directory \
   cp ./pair_annp_gpu.h &#8195; &#8194; lammps_PATH/src/GPU \
   cp ./pair_annp_gpu.cpp &#8195; lammps_PATH/src/GPU \
   add the name of the two "pair_annp_gpu*" files into Install.h file in GPU directory

5) make mpi

6) If you want to use OpenCL library, it's better using the cmake to compile lammps by following procedures: \
   mkdir build_opencl \
   cd build_opencl \
   cmake ../cmake -C ../cmake/presets/basic.cmake -D PKG_GPU=on -D GPU_API=opencl -D GPU_PREC=mixed -D GPU_ARCH=sm_61 \
   make \
   sudo make install \
   Note: the definition of shared memory in cuda must be changed into local (shared__ ----> local) in the "lal_annp.cu" file  


## MD simulation in Lammps:
1) the Newton third law must be opened: \
   newton on
2) pair_style	annp \
   pair_coeff	* * fe_annp_potential.ann Fe

## Tested systems, GPU cards, Lammps version:
1) Ubuntu 16.04 (System)
2) Nvidia RTX A5000 (GPU card)
3) August 02 2023 stable version (Lammps)

## References:
1) H. Mori, T. Ozaki, Phys Rev Mater. 4, 040601(R) (2020).
2) N. Artrith and A. Urban, Comput. Mater. Sci. 114, 135 (2016).
3) M. Zhang, K, Hibi, J. Inoue, Comput. Phys. Commun. 108655, 285 (2023).
4) S. Plimpton, J. Comput. Phys. 117, 1 (1995).

## Update and release:
[GitHub] (https://github.com/inouejunyalab/Meng_Zhang/tree/main/annp-gpu-lammps/fe_v2)

## Contact Information:
Email: meng_zhang@metall.t.u-tokyo.ac.jp (M. Zhang), junya_inoue@metall.t.u-tokyo.ac.jp (J. Inoue)
Please contact us if you have any questions or suggestion for the implementation

## License:
Copyright (C) 2022 Meng Zhang (meng_zhang@metall.t.u-tokyo.ac.jp), Koki Hibi (koki_hibi@metall.t.u-tokyo.ac.jp), Junya Inoue (junya_inoue@metall.t.u-tokyo.ac.jp).
The source code is distributed under a Mozilla Public License, V. 2.0.
