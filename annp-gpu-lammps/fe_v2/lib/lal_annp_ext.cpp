//* Host c++ *----------------------------------------
//      Artifical Neural Network Potential
//             Accelerated by GPU
//______________________________________________________
//  begin:  Wed February 16, 2022
//  email:  
//______________________________________________________
//------------------------------------------------------

#include <iostream>
#include <cassert>
#include <cmath>

#include "lal_annp.h"

using namespace std;
using namespace LAMMPS_AL;

static ANNP<PRECISION, ACC_PRECISION> ANNPMF;

/*---------------------------------------------------------------------
    allocate memory on host and device and copy constants to device
----------------------------------------------------------------------*/
int annp_gpu_init(const int ntypes, const int inum, const int nall, const int max_nbors,
                  const double cell_size, int& gpu_mode, FILE* screen, const int ntl, 
                  const int nhl, const int nnod, const int nsf, const int npsf, 
                  const int ntsf, const double e_scale, const double e_shift, 
                  const double e_atom, const int flagsym, int* flagact, 
                  double* sfnor_scal, double* sfnor_avg, double** host_cutsq, 
                  int* host_map, double*** host_weight_all, double*** host_bias_all) {

    ANNPMF.clear();
    gpu_mode = ANNPMF.device->gpu_mode();

    double gpu_split = ANNPMF.device->particle_split();
    int first_gpu = ANNPMF.device->first_device();                                                  // Index of first device used by a node
    int last_gpu = ANNPMF.device->last_device();                                                    // Index of last device used by a node
    int world_me = ANNPMF.device->world_me();                                                       // My rank within all processes
    int gpu_rank = ANNPMF.device->gpu_rank();                                                       // Return my rank in the device communicator
    int procs_per_gpu = ANNPMF.device->procs_per_gpu();                                             // Return the number of procs sharing a device (size of device communicator)

    if (gpu_split != 1.0)
        return -8;
    ANNPMF.device->init_message(screen, "annp/gpu", first_gpu, last_gpu);                           // printf some information about the GPU. Using "./deviceQuery" in "~/NVIDIA_CUDA-XXX_XX/bin/x86_64/linux/release/" file

    bool message = false;
    if (ANNPMF.device->replica_me() == 0 && screen)
        message = true;

    if (message) {
        fprintf(screen, "Initializing Device and compiling on process 0...");
        fflush(screen);
    }
    int init_ok = 0;
    if (world_me == 0)
        init_ok = ANNPMF.init(ntypes, inum, nall, max_nbors, cell_size,
                              gpu_split, screen, ntl, nhl, nnod, nsf,
                              npsf, ntsf, e_scale, e_shift, e_atom,
                              flagsym, flagact, sfnor_scal, sfnor_avg,
                              host_cutsq, host_map, host_weight_all, host_bias_all);
    ANNPMF.device->world_barrier();                                                                 // Quadro p5000, cuda cores(SP) 2560, 20 SM, 128 SP/SM, one sp = one thread, 32 thread = 1 wrap.
    if (message)                                                                                    // Thus, theoretically, it can run 4 wrap/SM, 80 wrap/p5000
        fprintf(screen, "Done.\n");                                                                 // Thinkpad, Quadro, T1000, cuda cores(SP) 896, 14SM, 64 SP/SM, it can run 2 wrap/SM

    for (int i = 0; i < procs_per_gpu; i++) {
        if (message)
        {
            if (last_gpu - first_gpu == 0)
                fprintf(screen, "Initializing Device %d on core %d...", first_gpu, i);
            else
                fprintf(screen, "Initializing Devices %d-%d on core %d...", first_gpu, last_gpu, i);
            fflush(screen);
        }
        if (gpu_rank == i && world_me != 0)
            init_ok = ANNPMF.init(ntypes, inum, nall, max_nbors, cell_size,
                                  gpu_split, screen, ntl, nhl, nnod, nsf,
                                  npsf, ntsf, e_scale, e_shift, e_atom, 
                                  flagsym, flagact, sfnor_scal, sfnor_avg,
                                  host_cutsq, host_map, host_weight_all, host_bias_all);
        ANNPMF.device->gpu_barrier();
        if (message)
            fprintf(screen, "Done.\n");
    }
    if (message)
        fprintf(screen, "\n");

    if (init_ok == 0)
        ANNPMF.estimate_gpu_overhead();
    return init_ok;
}

void annp_gpu_clear() {
    ANNPMF.clear();
}

int** annp_gpu_compute_n(double *eatom, double &eng_vdwl, double** f, const int ago, 
                         const int inum, const int nall, const int nghost, double** host_x, 
                         int* host_type, double* sublo, double* subhi, tagint* tag, 
                         int** nspecial, tagint** special, const bool eflag, const bool vflag, 
                         const bool ea_flag, const bool va_flag, int& host_start, int** ilist,
                         int** jnum, const double cpu_time, bool& success, double **vatom_annp) {                        // should be initialized in "lal_base_annp.h" file

    return ANNPMF.compute(eatom, eng_vdwl, f, ago, inum, nall, nghost, host_x, 
                          host_type, sublo, subhi, tag, nspecial, special, 
                          eflag, vflag, ea_flag, va_flag, host_start, 
                          ilist, jnum, cpu_time, success, vatom_annp);
}
void annp_gpu_compute(double* eatom, double& eng_vdwl, double** f, const int ago, 
                      const int inum, const int nall, const int nghost, double** host_x, 
                      int* host_type, int* ilist, int* numj, int** firstneigh, 
                      const bool eflag, const bool vflag, const bool ea_flag, 
                      const bool va_flag, int& host_start, const double cpu_time, 
                      bool& success, double **vatom_annp) {                                                             // should be initialized in "lal_base_annp.h" file
    
    ANNPMF.compute(eatom, eng_vdwl, f, ago, inum, nall, nghost, host_x, 
                   host_type, ilist, numj, firstneigh, eflag, vflag, 
                   ea_flag, va_flag, host_start, cpu_time, success, vatom_annp);
}

double annp_gpu_bytes() {
    return ANNPMF.host_memory_usage();
}
