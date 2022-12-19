//* Host c++ *------------------------------------------
//      Artifical Neural Network Potential
//             Accelerated by GPU
//______________________________________________________        
//  begin:  Wed February 16, 2022
//  email:  
//______________________________________________________
//------------------------------------------------------

#ifdef PAIR_CLASS
// clang-format off
PairStyle(annp/gpu, PairANNPGPU);
// clang-format on
#else

#ifndef LMP_PAIR_ANNP_GPU_H
#define LMP_PAIR_ANNP_GPU_H

#include "pair_annp.h"

namespace LAMMPS_NS {
    class PairANNPGPU : public PairANNP {
    public:
        PairANNPGPU(class LAMMPS*);
        virtual ~PairANNPGPU();
        void compute(int, int);
        void init_style();
        double memory_usage();

        enum { GPU_FORCE, GPU_NEIGH, GPU_HYB_NEIGH };

    protected:
        // void allocate();                                                                         // this is for allocating the cutghost, that used in sw, tersoff potentials                                        
        int gpu_mode;
        double cpu_time;
        int _max_nf = 0;
        double** neigF = nullptr;                                                                   // 1r: idj, 2r: x, 3r: y, 4r: z
    };
}

#endif
#endif






