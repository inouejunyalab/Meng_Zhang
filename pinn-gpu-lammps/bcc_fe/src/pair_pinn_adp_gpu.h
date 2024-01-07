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
PairStyle(pinn_adp/gpu, PairPINNADPGPU);
// clang-format on
#else

#ifndef LMP_PAIR_PINN_ADP_GPU_H
#define LMP_PAIR_PINN_ADP_GPU_H

#include "pair_pinn_adp.h"

namespace LAMMPS_NS {
    class PairPINNADPGPU : public PairPINN_ADP {
    public:
        PairPINNADPGPU(class LAMMPS*);
        virtual ~PairPINNADPGPU();
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