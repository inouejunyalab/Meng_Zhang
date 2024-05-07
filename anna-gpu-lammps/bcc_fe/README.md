## Impelementation of ANNA potential for bcc_fe

## Description:
-This package is used to implement a ANNA potential for BCC fe in LAMMPS package (Reference 1), which can be accelerated by using GPU card.\
-You can compile this package according to the following procedures. It can support the CUDA- and OpenCL-enabled GPU card. 

-The files in the lib folder are the library, which should be complied into lammps/lib/GPU package. \
-The files in the src folder are the source files, which provide the interface to lammps. \
-If you compile the lammps package, please be careful about the GPU_ARCH, which must be consistent with your GPU cards. 

## Installation:
1) download all files in "lib", "src" directors and the potential file
2) copy all files in the "lib" directory into lammps/lib/gpu directory \
   cp ./lal_*   lammps_PATH/lib/gpu

   Note: \
   (a) set correct value of GPU_ARCH in "Makefile.linux" \
   (b) make -f Makefile.linux
  
3) copy "pair_anna.h and pair_anna.cpp" in the "src" directory into lammps/src/MANYBODY directory \
   cp ./pair_anna.h     lammps_PATH/src/MANYBODY \
   cp ./pair_anna.cpp   lammps_PATH/src/MANYBODY
  
4) copy "pair_anna_gpu.h and pair_anna_gpu.cpp" in the "src" directory into lammps/src/GPU directory \
   cp ./pair_anna_gpu.h     lammps_PATH/src/GPU \
   cp ./pair_anna_gpu.cpp   lammps_PATH/src/GPU \
   add the name of the two "pair_anna_gpu*" files into Install.h file in GPU directory

5) make mpi

6) If you want to use OpenCL library, it's better using the cmake to compile lammps by following procedures: \
   mkdir build_opencl \
   cd build_opencl \
   cmake ../cmake -C ../cmake/presets/basic.cmake -D PKG_GPU=on -D GPU_API=opencl -D GPU_PREC=mixed -D GPU_ARCH=sm_86 \
   make \
   sudo make install \

## MD simulation in Lammps:
1) the Newton third law must be on and off for CPU-runs and GPU-runs, respectively: \
   newton on(or off)
3) pair_style anna_adp \
   pair_style * * fe_adp_potential_2310.anna Fe
   
## Tested systems, GPU cards, Lammps version:
1) Ubuntu 20.04 (System)
2) Nvidia RTX A5000 (GPU card)
3) September 2021 stable version (Lammps)

## References:
1) S. Plimpton, J. Comput. Phys. 117, 1 (1995).

## Update and release:
[GitHub] (https://github.com/inouejunyalab/Meng_Zhang/tree/main/anna-gpu-lammps/bcc_fe)

## Contact Information:
Email: meng_zhang@metall.t.u-tokyo.ac.jp (M. Zhang), junya_inoue@metall.t.u-tokyo.ac.jp (J. Inoue) Please contact us if you have any questions or suggestion for the implementation

## License:
Copyright (C) 2022 Meng Zhang (meng_zhang@metall.t.u-tokyo.ac.jp), Koki Hibi (koki_hibi@metall.t.u-tokyo.ac.jp), Junya Inoue (junya_inoue@metall.t.u-tokyo.ac.jp). The source code is distributed under a Mozilla Public License, V. 2.0.
