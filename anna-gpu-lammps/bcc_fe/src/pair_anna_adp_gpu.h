//* Host c++ *------------------------------------------
//      Physics-informed Neural Network Potential
//             Accelerated by GPU
//______________________________________________________        
//  begin:  Monday August 07, 2023
//  email:  meng_zhang@metall.t.u-tokyo.ac.jp
//          junya_inoue@metall.t.u-tokyo.ac.jp  
//______________________________________________________
//------------------------------------------------------

#ifdef PAIR_CLASS
// clang-format off
PairStyle(anna_adp/gpu, PairANNAADPGPU);
// clang-format on
#else

#ifndef LMP_PAIR_ANNA_ADP_GPU_H
#define LMP_PAIR_ANNA_ADP_GPU_H

#include "pair_anna_adp.h"

namespace LAMMPS_NS {
    class PairANNAADPGPU : public PairANNA_ADP {
    public:
        PairANNAADPGPU(class LAMMPS*);
        virtual ~PairANNAADPGPU();
        void compute(int, int);
        void init_style();
        double memory_usage();
        void* extract(const char*, int&) { return nullptr; }

        int pack_forward_comm(int, int*, double*, int, int*);             
        void unpack_forward_comm(int, int, double*);
        enum { GPU_FORCE, GPU_NEIGH, GPU_HYB_NEIGH };

    protected:
        int gpu_mode;
        static const int n_mu = 3, n_lamb = 6, nout = 2;                 
        double cpu_time;
        void* temp_adp_comm;                                             
        void* adp_rho;                                                   
        void* adp_mu[n_mu];                                              
        void* adp_lambda[n_lamb];                                        
        void* ladp_params[nout];                                         
        bool adp_single;
    };
}

#endif
#endif
