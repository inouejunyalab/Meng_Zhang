//* Host c++ *----------------------------------------
//      Artifical Neural Network Potential
//             Accelerated by GPU
//______________________________________________________
//  begin:  Wed February 16, 2022
//  email:  meng_zhang@metall.t.u-tokyo.ac.jp
//          junya_inoue@metall.t.u-tokyo.ac.jp   
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
    int first_gpu = ANNPMF.device->first_device();                                                  
    int last_gpu = ANNPMF.device->last_device();                                                    
    int world_me = ANNPMF.device->world_me();                                                       
    int gpu_rank = ANNPMF.device->gpu_rank();                                                       
    int procs_per_gpu = ANNPMF.device->procs_per_gpu();                                             

    if (gpu_split != 1.0)
        return -8;
    ANNPMF.device->init_message(screen, "annp/gpu", first_gpu, last_gpu);                           

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
    ANNPMF.device->world_barrier();                                                                 
    if (message)                                                                                    
        fprintf(screen, "Done.\n");                                                                 

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
                         int** jnum, const double cpu_time, bool& success, double **vatom_annp) {                        

    return ANNPMF.compute(eatom, eng_vdwl, f, ago, inum, nall, nghost, 
                          host_x, host_type, sublo, subhi, tag, nspecial, 
                          special, eflag, vflag, ea_flag, va_flag, 
                          host_start, ilist, jnum, cpu_time, success, vatom_annp);
}
void annp_gpu_compute(double* eatom, double& eng_vdwl, double** f, const int ago, 
                      const int inum, const int nall, const int nghost, double** host_x, 
                      int* host_type, int* ilist, int* numj, int** firstneigh, 
                      const bool eflag, const bool vflag, const bool ea_flag, 
                      const bool va_flag, int& host_start, const double cpu_time, 
                      bool& success, double **vatom_annp) {  
    
    ANNPMF.compute(eatom, eng_vdwl, f, ago, inum, nall, nghost, host_x, 
                   host_type, ilist, numj, firstneigh, eflag, vflag, 
                   ea_flag, va_flag, host_start, cpu_time, success, vatom_annp);
}

double annp_gpu_bytes() {
    return ANNPMF.host_memory_usage();
}
