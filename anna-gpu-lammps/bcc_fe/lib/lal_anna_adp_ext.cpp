//* Host c++ *----------------------------------------
//      Physics-informed Neural Network Potential
//             Accelerated by GPU
//______________________________________________________
//  begin:  Monday August 07, 2023
//  email:  meng_zhang@metall.t.u-tokyo.ac.jp
//          junya_inoue@metall.t.u-tokyo.ac.jp  
//______________________________________________________
//------------------------------------------------------

#include <iostream>
#include <cassert>
#include <cmath>

#include "lal_anna_adp.h"

using namespace std;
using namespace LAMMPS_AL;

static ANNAADP<PRECISION, ACC_PRECISION> ANNAADPMF;

/*---------------------------------------------------------------------
    allocate memory on host and device and copy constants to device
----------------------------------------------------------------------*/
int anna_adp_gpu_init(const int ntypes, const int inum, const int nall, 
                      const int max_nbors, const double cell_size, 
                      int& gpu_mode, FILE* screen, const int maxspecial,
                      const int ntl, const int nhl, const int nnod, 
                      const int nsf, const int npsf, const int ntsf, 
                      const int nout, const int ngp, const int n_mu, const int n_lamb, 
                      const double e_base, const int flagsym, int* flagact, 
                      double** host_cutsq, int* host_map, double*** host_weight_all,
                      double*** host_bias_all, double *host_gadp_params, int &adp_size) {

    ANNAADPMF.clear();
    gpu_mode = ANNAADPMF.device->gpu_mode();

    double gpu_split = ANNAADPMF.device->particle_split();
    int first_gpu = ANNAADPMF.device->first_device();                            
    int last_gpu = ANNAADPMF.device->last_device();                              
    int world_me = ANNAADPMF.device->world_me();                                 
    int gpu_rank = ANNAADPMF.device->gpu_rank();                                 
    int procs_per_gpu = ANNAADPMF.device->procs_per_gpu();                       

    if (gpu_split != 1.0)
        return -8;

    adp_size = sizeof(PRECISION);                                                
    ANNAADPMF.device->init_message(screen, "anna_adp/gpu", first_gpu, last_gpu); 

    bool message = false;
    if (ANNAADPMF.device->replica_me() == 0 && screen)
        message = true;

    if (message) {
        fprintf(screen, "Initializing Device and compiling on process 0...");
        fflush(screen);
    }
    int init_ok = 0;
    if (world_me == 0)
        init_ok = ANNAADPMF.init(ntypes, inum, nall, max_nbors, cell_size,
                                 gpu_split, screen, maxspecial, ntl, nhl, 
                                 nnod, nsf, npsf, ntsf, nout, ngp, n_mu, n_lamb, 
                                 e_base, flagsym, flagact, host_cutsq, host_map, 
                                 host_weight_all, host_bias_all, host_gadp_params);

    ANNAADPMF.device->world_barrier();                                            
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
            init_ok = ANNAADPMF.init(ntypes, inum, nall, max_nbors, cell_size,
                                     gpu_split, screen, maxspecial, ntl, nhl, 
                                     nnod, nsf, npsf, ntsf, nout, ngp, n_mu, n_lamb, 
                                     e_base, flagsym, flagact, host_cutsq, host_map, 
                                     host_weight_all, host_bias_all, host_gadp_params);

        ANNAADPMF.device->gpu_barrier();
        if (message)
            fprintf(screen, "Done.\n");
    }
    if (message)
        fprintf(screen, "\n");

    if (init_ok == 0)
        ANNAADPMF.estimate_gpu_overhead();
    return init_ok;
}

void anna_adp_gpu_clear() {
    ANNAADPMF.clear();
}

int** anna_adp_gpu_compute_n(const int ago, const int inum_full, const int nall, 
                             const int nlocal, double** host_x, int* host_type, 
                             double* sublo, double* subhi, tagint* tag, 
                             int** nspecial, tagint** special, const bool eflag, 
                             const bool vflag, const bool ea_flag, const bool va_flag, 
                             void** adp_rho, void* adp_mu[], void* adp_lambda[], 
                             void* ladp_params[], int& host_start, int** ilist, 
                             int** jnum, bool& success, const double cpu_time) {                   

    return ANNAADPMF.compute(ago, inum_full, nall, nlocal, host_x, host_type, 
                             sublo, subhi, tag, nspecial, special, eflag, vflag, 
                             ea_flag, va_flag, adp_rho, adp_mu, adp_lambda, 
                             ladp_params, host_start, ilist, jnum, success, cpu_time);
}
void anna_adp_gpu_compute(const int ago, const int inum_full, const int nall, 
                          const int nlocal, double** host_x, int* host_type, 
                          int* ilist, int* numj, int** firstneigh, 
                          const bool eflag, const bool vflag, 
                          const bool ea_flag, const bool va_flag, void** adp_rho, 
                          void* adp_mu[], void* adp_lambda[], void* ladp_params[], 
                          int& host_start, bool& success, const double cpu_time) {                 

    ANNAADPMF.compute(ago, inum_full, nall, nlocal, host_x, host_type, 
                      ilist, numj, firstneigh, eflag, vflag, ea_flag, 
                      va_flag, adp_rho, adp_mu, adp_lambda, ladp_params, 
                      host_start, success, cpu_time);
}

void anna_adp_gpu_compute_force(const int nall, int* ilist, const bool eflag, 
                                const bool vflag, const bool ea_flag, const bool va_flag) {

    ANNAADPMF.compute_force(nall, ilist, eflag, vflag, ea_flag, va_flag);
}

double anna_adp_gpu_bytes() {
    return ANNAADPMF.host_memory_usage();
}
