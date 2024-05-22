//* Host c++ *------------------------------------------
//      Physics-informed Neural Network Potential
//             Accelerated by GPU
//______________________________________________________
//  begin:  Monday August 07, 2023
//  email:  meng_zhang@metall.t.u-tokyo.ac.jp
//          junya_inoue@metall.t.u-tokyo.ac.jp  
//______________________________________________________
//------------------------------------------------------

#include "pair_anna_adp_gpu.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "gpu_extra.h"
#include "memory.h"
#include "suffix.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"

#include <cmath>
#include <iostream>

using namespace LAMMPS_NS;

// External functions from cuda library for atom decomposition
int anna_adp_gpu_init(const int ntypes, const int inum, const int nall, 
                      const int max_nbors, const double cell_size, 
                      int& gpu_mode, FILE* screen, const int maxspecial,
                      const int ntl, const int nhl, const int nnod, 
                      const int nsf, const int npsf, const int ntsf, 
                      const int nout, const int ngp, const int n_mu, const int n_lamb, 
                      const double e_base, const int flagsym, int* flagact, 
                      double** host_cutsq, int* host_map, double*** host_weight_all,
                      double*** host_bias_all, double *gadp_params, int &adp_size);       

void anna_adp_gpu_clear();

// build nighbord list on gpu, then calcualte energy
int** anna_adp_gpu_compute_n(const int ago, const int inum_full, const int nall, 
                             const int nlocal, double** host_x, int* host_type, 
                             double* sublo,double* subhi, tagint* tag, int** nspecial, 
                             tagint** special, const bool eflag, const bool vflag, 
                             const bool ea_flag, const bool va_flag, 
                             void** adp_rho, void* adp_mu[], void* adp_lambda[], 
                             void *ladp_params[], int& host_start, int** ilist, 
                             int** jnum, bool& success, const double cpu_time);

// copy neighbor list from cpu, then calcualte energy
void anna_adp_gpu_compute(const int ago, const int inum_full, const int nall, 
                          const int nlocal, double** host_x, int* host_type, 
                          int* ilist, int* numj, int** firstneigh, 
                          const bool eflag, const bool vflag, 
                          const bool ea_flag, const bool va_flag, void** adp_rho, 
                          void* adp_mu[], void* adp_lambda[], void *ladp_params[], 
                          int& host_start, bool& success, const double cpu_time);

// calculating the force
void anna_adp_gpu_compute_force(const int nall, int* ilist, const bool eflag, 
                                const bool vflag, const bool ea_flag, const bool va_flag);

double anna_adp_gpu_bytes();

/*----------------------------------------------------------------------*/
PairANNAADPGPU::PairANNAADPGPU(LAMMPS *lmp) : PairANNA_ADP(lmp), gpu_mode(GPU_FORCE) {
    respa_enable = 0;                                                                   
    cpu_time = 0.0;                                              
    suffix_flag |= Suffix::GPU;
    GPU_EXTRA::gpu_ready(lmp->modify, lmp->error);
}

/*---------------------------------------------------------------------
    check if allocated, since class can be destructed when incomplete
----------------------------------------------------------------------*/
PairANNAADPGPU::~PairANNAADPGPU() {
    anna_adp_gpu_clear();
}    

/*--------------------------------------------------------------------*/
double PairANNAADPGPU::memory_usage() {
    double bytes = Pair::memory_usage();
    return bytes + anna_adp_gpu_bytes();
}

/*---------------------------------------------------------------------
                    compute force and energy
----------------------------------------------------------------------*/
void PairANNAADPGPU::compute(int eflag, int vflag) {
    ev_init(eflag, vflag);
    int nghost = atom->nghost;
    int nlocal = atom->nlocal;
    int nall = nlocal + nghost;                                                       
    int inum, host_start, inum_dev;
    bool success = true;
    int *ilist, *numneigh, **firstneigh;
    if(gpu_mode != GPU_FORCE) {
        double sublo[3],subhi[3];
        if (domain->triclinic == 0) {
            sublo[0] = domain->sublo[0];
            sublo[1] = domain->sublo[1];
            sublo[2] = domain->sublo[2];
            subhi[0] = domain->subhi[0];
            subhi[1] = domain->subhi[1];
            subhi[2] = domain->subhi[2];
        } else {
            domain->bbox(domain->sublo_lamda,domain->subhi_lamda,sublo,subhi);
        }
        inum = atom->nlocal;
        firstneigh = anna_adp_gpu_compute_n(neighbor->ago, inum, nall, nlocal, 
                                            atom->x, atom->type, sublo, subhi, 
                                            atom->tag, atom->nspecial, 
                                            atom->special, eflag, vflag, eflag_atom, 
                                            vflag_atom, &adp_rho, adp_mu, 
                                            adp_lambda, ladp_params, host_start, 
                                            &ilist, &numneigh, success, cpu_time);
    } else {
        inum = list->inum;
        ilist = list->ilist;
        numneigh = list->numneigh;
        firstneigh = list->firstneigh;
        anna_adp_gpu_compute(neighbor->ago, inum, nall, nlocal, 
                             atom->x, atom->type, ilist, numneigh, 
                             firstneigh, eflag, vflag, eflag_atom, 
                             vflag_atom, &adp_rho, adp_mu, 
                             adp_lambda, ladp_params,
                             host_start, success, cpu_time);
    }
    if(!success)
        error->one(FLERR,"Insufficient memory on accelerator");

    temp_adp_comm = adp_rho;
    //comm->forward_comm_pair(this);
    comm->forward_comm(this);
 
    for(int k = 0; k < n_mu; k++) {
        temp_adp_comm = adp_mu[k];
        //comm->forward_comm_pair(this);
        comm->forward_comm(this);
    }
    for(int k = 0; k < n_lamb; k++) {
        temp_adp_comm = adp_lambda[k];
        //comm->forward_comm_pair(this);
        comm->forward_comm(this);
    }
    for(int k = 0; k < nout; k++) {
        temp_adp_comm = ladp_params[k];
        //comm->forward_comm_pair(this);
        comm->forward_comm(this);
    }
    // compute force of each atom on GPU
    if (gpu_mode != GPU_FORCE)
        anna_adp_gpu_compute_force(nall, nullptr, eflag, vflag, eflag_atom, vflag_atom);
    else
        anna_adp_gpu_compute_force(nall, ilist, eflag, vflag, eflag_atom, vflag_atom);
}

/*---------------------------------------------------------------------
                 init specific to this pair style
----------------------------------------------------------------------*/
void PairANNAADPGPU::init_style() {
    if (atom->tag_enable == 0)
        error->all(FLERR, "Pair style anna_adp/gpu requires atom IDs");
    if (force->newton_pair == 1)
        error->all(FLERR, "Pair style anna_adp/gpu requires newton pair off");

    // parameters that need to send to GPU
    int ntl, nhl, nnod, nout, nsf, npsf, ntsf, ngp, nelements;
    int flagsym, * flagact;
    double e_base;
    double* gadp_params;
    double*** weight_all, *** bias_all;                                       
    weight_all = nullptr;                                                     
    bias_all = nullptr;                                                       
    flagact = nullptr;
    gadp_params = nullptr;

    ntl = params[0].ntl;                                                      
    nhl = params[0].nhl;
    nnod = params[0].nnod;
    nout = params[0].nout;
    nsf = params[0].nsf;
    npsf = params[0].npsf;
    ntsf = params[0].ntsf;
    ngp = params[0].ngp;
    
    nelements = params[0].nelements;                                         
    e_base = params[0].e_base;
    flagsym = params[0].flagsym;

    flagact = new int[ntl - 1];
    weight_all = new double** [nelements];
    bias_all = new double** [nelements];
    c_3d_matrix(nelements, ntl - 1, nnod * nsf, weight_all);
    c_3d_matrix(nelements, ntl - 1, nnod, bias_all);

    double cut;
    for (int i = 1; i <= atom->ntypes; i++) {
        for (int j = i; j <= atom->ntypes; j++) {
            if (setflag[i][j] != 0 || (setflag[i][i] != 0 && setflag[j][j] != 0)) {
                cut = init_one(i, j);
                cut *= cut;
                cutsq[i][j] = cut;
                cutsq[j][i] = cut;
            } else {
                cutsq[i][j] = 0.0;
                cutsq[j][i] = 0.0;
            }
        }
    }

    for (int nt = 1; nt <= atom->ntypes; nt++) {                                                    
        int rtype = map[nt];                                                  

        if (nt > 1 && rtype == map[nt - 1])
            continue;
        for (int i = 0; i < ntl - 1; i++) {                                   
            int nrow_w = nnod, ncol_w = nnod, nrow_b = 1, ncol_b = nnod;
            if (i == 0) ncol_w = nsf;
            if (i == ntl - 2) {                                               
                nrow_w = nout;                                                
                ncol_b = nout;
            }
            for (int j = 0; j < nrow_w; j++)
                for (int k = 0; k < ncol_w; k++)
                    weight_all[rtype][i][k + j * ncol_w] = params[0].all_anna[rtype].weight_all[i][j][k];
            for (int j = 0; j < nrow_b; j++)
                for (int k = 0; k < ncol_b; k++)
                    bias_all[rtype][i][k + j * ncol_b] = params[0].all_anna[rtype].bias_all[i][j][k];
            flagact[i] = params[0].flagact[i];
        }
    }

    gadp_params = new double[ngp];
    for (int i = 0; i < ngp; i++) {
        gadp_params[i] = params[0].gparams[i];                                 
    }

    int maxspecial = 0;
    if (atom->molecular != Atom::ATOMIC)
        maxspecial = atom->maxspecial;
    int adp_size;                                                              
    double cell_size = cutmax + neighbor->skin;
    int mnf = 5e-2 * neighbor->oneatom;
    int nall = atom->nlocal + atom->nghost;                                    
    int success = anna_adp_gpu_init(atom->ntypes, atom->nlocal, nall, mnf,     
                                    cell_size, gpu_mode, screen, maxspecial,
                                    ntl, nhl, nnod, nsf, npsf, ntsf, nout, ngp, 
                                    n_mu, n_lamb, e_base, flagsym, flagact, cutsq, 
                                    map, weight_all, bias_all, gadp_params, adp_size);  

    // free array
    delete[] flagact;
    delete[] gadp_params;
    d_3d_matrix(nelements, ntl - 1, weight_all);
    d_3d_matrix(nelements, ntl - 1, bias_all);

    GPU_EXTRA::check_flag(success, error, world);

    if (gpu_mode == GPU_FORCE) {
        //int irequest = neighbor->request(this);
        //neighbor->requests[irequest]->half = 0;
        //neighbor->requests[irequest]->full = 1;
        neighbor->add_request(this, NeighConst::REQ_FULL);
    }

    if (adp_size == sizeof(double))
        adp_single = false;
    else
        adp_single = true;                                                   
}

/*---------------------------------------------------------------------
        packing the value of _fp, _mu, _lambda for communication
----------------------------------------------------------------------*/
int PairANNAADPGPU::pack_forward_comm(int n, int* list, double* buf_adp, int /*pbc_flag*/, int* /*pbc*/) {
    
    int i, j, m = 0;
    if (adp_single) {
        float* adp_ptr = (float *)temp_adp_comm;
        for (i = 0; i < n; i++) {
            j = list[i];
            buf_adp[m++] = static_cast<double>(adp_ptr[j]);
        }
    } 
    else {
        double* adp_ptr = (double *)temp_adp_comm;
        for (i = 0; i < n; i++) {
            j = list[i];
            buf_adp[m++] = adp_ptr[j];
        }
    }
    return m;                                                               
}

/*---------------------------------------------------------------------
                unpacking the buf of _fp, _mu, _lambda
----------------------------------------------------------------------*/
void PairANNAADPGPU::unpack_forward_comm(int n, int first, double* buf_adp) {

    int i, last, m = 0;
    last = first + n;
    if (adp_single) {
        float* adp_ptr = (float *)temp_adp_comm;
        for (i = first; i < last; i++) {
            adp_ptr[i] = buf_adp[m++];
        }
    } 
    else {
        double* adp_ptr = (double *)temp_adp_comm;
        for (i = first; i < last; i++) {
            adp_ptr[i] = buf_adp[m++];
        }
    }
}
