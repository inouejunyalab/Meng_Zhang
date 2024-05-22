//* Host c++ *------------------------------------------
//      Artifical Neural Network Potential
//             Accelerated by GPU
//______________________________________________________
//  begin:  Mon Oct 23, 2022
//  email:  meng_zhang@metall.t.u-tokyo.ac.jp
//          junya_inoue@metall.t.u-tokyo.ac.jp 
//______________________________________________________
//------------------------------------------------------

#include "pair_annp_gpu.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "gpu_extra.h"
#include "memory.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "suffix.h"

#include <cmath>
#include <iostream>

using namespace LAMMPS_NS;

// External functions from cuda library for atom decomposition
int annp_gpu_init(const int ntypes, const int inum, const int nall, 
                  const int max_nbors, const double cell_size, 
                  int& gpu_mode, FILE* screen, const int ntl, 
                  const int nhl, const int nnod, const int nsf,
                  const int npsf, const int ntsf, const double e_scale,
                  const double e_shift, const double e_atom, 
                  const int flagsym, int* flagact, double* sf_scal, 
                  double* sf_min, double** host_cutsq, int* host_map, 
                  double*** host_weight_all, double*** host_bias_all,
                  double** host_cofsymrad, double** host_cofsymang);

void annp_gpu_clear();

// build nighbord list on gpu, then calcualte energy and force
int** annp_gpu_compute_n(double *eatom_annp, double& eng_vdwl_annp, double** f, const int ago, 
                         const int inum, const int nall, const int nghost, double** host_x, 
                         int* host_type, double* sublo, double* subhi, tagint* tag, int** nspecial, 
                         tagint** special, const bool eflag, const bool vflag, const bool ea_flag, 
                         const bool va_flag, int& host_start, int** ilist, int** jnum, 
                         const double cpu_time, bool& success, double **vatom_annp);

// copy neighbor list from cpu, then calcualte energy and force
void annp_gpu_compute(double* eatom_annp, double& eng_vdwl_annp, double** f, const int ago, 
                      const int inum, const int nall, const int nghost, double** host_x, 
                      int* host_type, int* ilist, int* numj, int** firstneigh, const bool eflag, 
                      const bool vflag, const bool ea_flag, const bool va_flag, int& host_start, 
                      const double cpu_time, bool& success, double **vatom_annp);

double annp_gpu_bytes();

/*----------------------------------------------------------------------*/
PairANNPGPU::PairANNPGPU(LAMMPS *lmp) : PairANNP(lmp), gpu_mode(GPU_FORCE) {
    respa_enable = 0;                                                                              
    cpu_time = 0.0;                                              
    suffix_flag |= Suffix::GPU;
    GPU_EXTRA::gpu_ready(lmp->modify, lmp->error);
}

/*---------------------------------------------------------------------
    check if allocated, since class can be destructed when incomplete
----------------------------------------------------------------------*/
PairANNPGPU::~PairANNPGPU() {
    annp_gpu_clear();
}    

/*--------------------------------------------------------------------*/
double PairANNPGPU::memory_usage() {
    double bytes = Pair::memory_usage();
    return bytes + annp_gpu_bytes();
}

/*---------------------------------------------------------------------
    compute force and energy
----------------------------------------------------------------------*/
void PairANNPGPU::compute(int eflag, int vflag) {
    ev_init(eflag, vflag);
    int nghost = atom->nghost;
    int nall = atom->nlocal + nghost;
    int inum, host_start, inum_dev;
    bool success = true;
    int *ilist, *numneigh, **firstneigh;
    double** f = atom->f;                                                                         

    double eng_vdwl_annp = 0.0; 
    double* eatom_annp = LAMMPS_NS::Pair::eatom;
    double** vatom_annp = LAMMPS_NS::Pair::vatom;

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
        firstneigh = annp_gpu_compute_n(eatom_annp, eng_vdwl_annp, f, neighbor->ago, 
                                        inum, nall, nghost, atom->x, atom->type, sublo, 
                                        subhi, atom->tag, atom->nspecial, atom->special, 
                                        eflag, vflag, eflag_atom, vflag_atom, host_start, 
                                        &ilist, &numneigh, cpu_time, success, vatom_annp);
    } else {
        inum = list->inum;
        ilist = list->ilist;
        numneigh = list->numneigh;
        firstneigh = list->firstneigh;
        annp_gpu_compute(eatom_annp, eng_vdwl_annp, f, neighbor->ago, 
                         inum, nall, nghost, atom->x, atom->type, ilist, 
                         numneigh, firstneigh, eflag, vflag, eflag_atom, 
                         vflag_atom, host_start, cpu_time, success, vatom_annp);
    }
    if(!success)
        error->one(FLERR,"Insufficient memory on accelerator");

    // updating the total energy, energy of each atoms, and virial force
    if (success && vflag)    virial_fdotr_compute(); 
    if (eflag) eng_vdwl = eng_vdwl_annp;
}

/*---------------------------------------------------------------------
    init specific to this pair style
----------------------------------------------------------------------*/
void PairANNPGPU::init_style() {
    if (atom->tag_enable == 0)
        error->all(FLERR, "Pair style annp/gpu requires atom IDs");
    if (force->newton_pair == 0)
        error->all(FLERR, "Pair style annp/gpu requires newton pair on");

    // parameters that need to send to GPU
    int ntl, nhl, nnod, nsf, npsf, ntsf, nelements;
    int flagsym, * flagact;
    double e_scale, e_shift, e_atom;
    double* sf_scal, * sf_min;                                                            
    double*** weight_all, *** bias_all;                                                        
    weight_all = nullptr;                                                                      
    bias_all = nullptr;                                                                           
    flagact = nullptr;

    // get the parameters from potential file
    ntl = params[0].ntl;                                                               
    nhl = params[0].nhl;
    nnod = params[0].nnod;
    nsf = params[0].nsf;
    npsf = params[0].npsf;
    ntsf = params[0].ntsf;
    
    nelements = params[0].nelements;                                                    
    e_scale = params[0].e_scale;
    e_shift = params[0].e_shift;
    e_atom = params[0].e_atom;
    flagsym = params[0].flagsym;

    // initinalization of annp matrix
    sf_scal = new double[nsf];
    sf_min = new double[nsf];
    memset(sf_scal, 0, sizeof(double) * nsf);
    memset(sf_min, 0, sizeof(double) * nsf);
    flagact = new int[ntl - 1];
    weight_all = new double** [nelements];
    bias_all = new double** [nelements];
    c_3d_matrix(nelements, ntl - 1, nnod * nsf, weight_all);
    c_3d_matrix(nelements, ntl - 1, nnod, bias_all);

    double** cofsymrad = new double* [npsf];
    double** cofsymang = new double* [ntsf];
    c_2d_matrix(npsf, 3, cofsymrad);
    c_2d_matrix(ntsf, 4, cofsymang);

    // for cutsq array
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

    // for weight_all, bias_all, flagact matrix
    for (int nt = 1; nt <= atom->ntypes; nt++) {                                                    
        int rtype = map[nt];                                                            

        if (nt > 1 && rtype == map[nt - 1])    continue;                                            
        for (int i = 0; i < ntl - 1; i++) {
            int nrow_w = nnod, ncol_w = nnod, nrow_b = 1, ncol_b = nnod;
            if (i == 0) ncol_w = nsf;
            if (i == ntl - 2) {                                                                 
                nrow_w = 1;
                ncol_b = 1;
            }
            for (int j = 0; j < nrow_w; j++)
                for (int k = 0; k < ncol_w; k++)
                    weight_all[rtype][i][k + j * ncol_w] = params[0].all_annp[rtype].weight_all[i][j][k];
            for (int j = 0; j < nrow_b; j++)
                for (int k = 0; k < ncol_b; k++)
                    bias_all[rtype][i][k + j * ncol_b] = params[0].all_annp[rtype].bias_all[i][j][k];
            flagact[i] = params[0].flagact[i];
        }
    }

    // for coefficient of the symmetry function
    for (int i = 0; i < ntsf; i++) {
        if (i < npsf) {
            for (int j = 0; j < 3; j++)
                cofsymrad[i][j] = params[0].sym_coerad[i][j];
        }
        for (int j = 0; j < 4; j++)
            cofsymang[i][j] = params[0].sym_coeang[i][j];
    }

    // for normalizing the symmetry function
    for (int i = 0; i < nsf; i++) {
        sf_min[i] = params[0].sf_min[i];
        sf_scal[i] = params[0].sf_max[i] - params[0].sf_min[i];
    }

    // initialization paramaters , "maxspecial" is needed or not.....check it 
    double cell_size = cutmax + neighbor->skin;
    int mnf = 5e-2 * neighbor->oneatom;
    int success = annp_gpu_init(atom->ntypes, atom->nlocal, atom->nlocal + atom->nghost,          
                                mnf, cell_size, gpu_mode, screen, ntl, nhl, nnod, nsf,
                                npsf, ntsf, e_scale, e_shift, e_atom, flagsym, flagact, 
                                sf_scal, sf_min, cutsq, map, weight_all, bias_all,
                                cofsymrad, cofsymang);
    // free array
    delete[] flagact;
    delete[] sf_scal;
    delete[] sf_min;
    d_3d_matrix(nelements, ntl - 1, weight_all);
    d_3d_matrix(nelements, ntl - 1, bias_all);
    d_2d_matrix(npsf, cofsymrad);
    d_2d_matrix(ntsf, cofsymang);

    GPU_EXTRA::check_flag(success, error, world);

    if (gpu_mode == GPU_FORCE) {
        //int irequest = neighbor->request(this);
        //neighbor->requests[irequest]->half = 0;
        //neighbor->requests[irequest]->full = 1;
        neighbor->add_request(this, NeighConst::REQ_FULL);
    }
}

