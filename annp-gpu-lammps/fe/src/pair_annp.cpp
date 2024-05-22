//* Host c++ *------------------------------------------
//      Artifical Neural Network Potential
//             CPU version
//______________________________________________________        
//  begin:  Wed February 16, 2022 (1/16/2022)
//  email:  meng_zhang@metall.t.u-tokyo.ac.jp
//          junya_inoue@metall.t.u-tokyo.ac.jp     
//______________________________________________________
//------------------------------------------------------

#include "pair_annp.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"
#include "memory.h"
#include "math_const.h"
#include "math_extra.h"
#include "math_special.h"
#include "suffix.h"
#include "tokenizer.h"

#include <string>
#include <cmath>
#include <iostream>
#include <fstream>
#include <cstring>
#include <cctype>
#include <iomanip>
//#include <mpi.h>                                                                 

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace MathSpecial;
using namespace MathExtra;

/*----------------------------------------------------------------------*/
PairANNP::PairANNP(LAMMPS *lmp) : Pair(lmp)
{
    restartinfo = 0;                                                                
    one_coeff = 1;                                                               
    manybody_flag = 1;                                                           

    elements_coeff = nullptr;
    params = nullptr;
}
/*---------------------------------------------------------------------
    check if allocated, since class can be destructed when incomplete
----------------------------------------------------------------------*/
PairANNP::~PairANNP()
{
    if (copymode)    return;                                                     
    if (allocated) {
        memory->destroy(cutsq);
        memory->destroy(setflag);

        delete[]elements_coeff;
        elements_coeff = nullptr;
    }
    if (params) {                                                                
        delete[] params;
        params = nullptr;
    }
}

/*---------------------------------------------------------------------
    compute force and energy
----------------------------------------------------------------------*/
void PairANNP::compute(int eflag, int vflag)                            
{
    int i, j, k, ii, jj, kk, inum, jnum;
    int itype, jtype, ktype, ritype, rjtype, rktype;
    double xtmp, ytmp, ztmp, evdwl;
    int* ilist, * jlist, * numneigh, ** firstneigh;

    evdwl = 0.0;    
    ev_init(eflag, vflag);                                  
    double** x = atom->x;                                   
    double** f = atom->f;                                   
    int* type = atom->type;                                
    int nlocal = atom->nlocal;                      
    int nall = nlocal + atom->nghost;               

    inum = list->inum;                                  
    ilist = list->ilist;                                
    numneigh = list->numneigh;                          
    firstneigh = list->firstneigh;                      

    int dimG = params[0].nsf;
    int dimGp = params[0].npsf;
    int dimGt = params[0].ntsf;

    double* sf_scale = new double[dimG];                          
    for (int i = 0; i < dimG; i++) {
        double t_avg = params[0].sfnor_avg[i];
        double t_scale = sqrt(params[0].sfnor_cov[i] - t_avg * t_avg);
        if (t_scale <= 1.0e-10) {
            std::cout << "Warning: Invalid scaling encountered in 'sfv_normalize()'" << std::endl;
            sf_scale[i] = 0.0;
        }
        else
            sf_scale[i] = 1.0 / t_scale;
    }
    for (ii = 0; ii < inum; ii++) {
        i = ilist[ii];                                            
        ritype = type[i];                                         
        itype = map[ritype];                                      
                                                    
        jnum = numneigh[i];                                       
        jlist = firstneigh[i];                                    
        xtmp = x[i][0];
        ytmp = x[i][1];
        ztmp = x[i][2];

        double rsqij, rsqik, xij[3], xik[3], rij_unit[3], rik_unit[3];
        double fcij, fcik, dfcij, dfcik;
        double* dr_dj = new double[3];
        double* dE_dG = new double[dimG];
        double* G = new double[dimG];    
        double*** dG_dij = new double** [nall + 2];                  

        memset(dr_dj, 0, sizeof(double) * 3);
        memset(dE_dG, 0, sizeof(double)* dimG);                      
        memset(G, 0, sizeof(double)* dimG);                                                                                         
        c_3d_matrix(nall + 2, dimG, 3, dG_dij);
        
        for (jj = 0; jj < jnum; jj++) {
            j = jlist[jj];
            j &= NEIGHMASK;                                
            rjtype = type[j];
            jtype = map[rjtype];                           

            xij[0] = xtmp - x[j][0];
            xij[1] = ytmp - x[j][1];
            xij[2] = ztmp - x[j][2];
            rsqij = xij[0] * xij[0] + xij[1] * xij[1] + xij[2] * xij[2];
            if (rsqij > cutsq[ritype][rjtype] || rsqij < 1.0e-12)  continue;
            printf("neighbor list....%d %d %f %f %f %f %f %f\n", i, j, x[i][0], x[i][1], x[i][2], x[j][0], x[j][1], x[j][2]);

            const double rijinv = 1.0 / sqrt(dot3(xij, xij));        
            scale3(rijinv, xij, rij_unit);                           
            
            double rij = sqrt(rsqij);
            double Rc = sqrt(cutsq[ritype][rjtype]);
            annp_fc(rij, Rc, fcij, dfcij);
            annp_dr_dij(1, rij, xij, dr_dj);
            annp_symmetry_pair(j, rij, fcij, dfcij, dr_dj, sf_scale, G, dG_dij, &params[0]);

            for (kk = jj + 1; kk < jnum; kk++) {
                k = jlist[kk];
                rktype = type[k];
                ktype = map[rktype];

                xik[0] = xtmp - x[k][0];
                xik[1] = ytmp - x[k][1];
                xik[2] = ztmp - x[k][2];
                rsqik = xik[0] * xik[0] + xik[1] * xik[1] + xik[2] * xik[2];
                if (rsqik > cutsq[ritype][rktype] || rsqik < 1.0e-12)  continue;
                
                const double rikinv = 1.0 / sqrt(dot3(xik, xik));
                scale3(rikinv, xik, rik_unit);
                double cos_theta = dot3(rij_unit, rik_unit);

                double rik = sqrt(rsqik);
                Rc = sqrt(cutsq[ritype][rktype]);
                annp_fc(rik, Rc, fcik, dfcik);
                annp_symmetry_trip(j, k, rij, rik, cos_theta, fcij, dfcij, fcik, dfcik, 
                                   xij, xik, dr_dj, sf_scale, G, dG_dij, &params[0]);  
            }
        }
        for (int k = 0; k < dimG; k++) {
            G[k] = G[k] - sf_scale[k] * params[0].sfnor_avg[k];
        }
        annp_feed_forward(itype, G, dE_dG, eflag, evdwl, &params[0]); 

        if (eflag) {
            if (eflag_global) eng_vdwl += evdwl;
            if (eflag_atom) eatom[i] += evdwl;
        }

        double Fi[3] = { 0.0, 0.0, 0.0 };
        for (jj = 0; jj < jnum; jj++) {
            double Fj[3] = { 0.0, 0.0, 0.0 };
            j = jlist[jj];
            j &= NEIGHMASK;
            for (int k = 0; k < 3; k++) {
                for (int n = 0; n < dimG; n++)
                   Fj[k] += (-1.0) * dE_dG[n] * dG_dij[j][n][k] * params[0].e_scale;
                Fi[k] += Fj[k];
                f[j][k] += Fj[k];
            }
            if(evflag) {
                double delx = x[i][0]-x[j][0];
                double dely = x[i][1]-x[j][1];
                double delz = x[i][2]-x[j][2];
                double fx = -Fj[0];
                double fy = -Fj[1];
                double fz = -Fj[2];
                ev_tally_xyz(i, j, nlocal, force->newton_pair, 0.0, 0.0, fx, fy, fz, delx, dely, delz);
            }
        }
        f[i][0] -= Fi[0];
        f[i][1] -= Fi[1];
        f[i][2] -= Fi[2];

        delete[]dr_dj;
        delete[]G;
        delete[]dE_dG;
        d_3d_matrix(nall + 2, dimG, dG_dij);
    }
    if (vflag_fdotr)    virial_fdotr_compute();                                 
    delete[]sf_scale;
}

/*---------------------------------------------------------------------
    allocate all arrys
----------------------------------------------------------------------*/
void PairANNP::allocate()                                      
{
    allocated = 1;
    int n = atom->ntypes;

    memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
    memory->create(setflag, n + 1, n + 1, "pair:setflag");
    for (int i = 1; i <= n; i++)                                                
        for (int j = i; j <= n; j++) {
            setflag[i][j] = 0;                                                  
        }

    elements_coeff = new std::string[n + 1];                                    
    delete[] map;
    map = new int[n + 1];                                                       
    for (int i = 1; i <= n; i++)   map[i] = -1;                                 
}

/*---------------------------------------------------------------------
    global setting
----------------------------------------------------------------------*/
void PairANNP::settings(int narg, char **arg)
{
    if (narg != 0)    error->all(FLERR, "Illegal pair_style command");
}

/*---------------------------------------------------------------------
    set coeffs for one or more type pairs     
----------------------------------------------------------------------*/
void PairANNP::coeff(int narg, char** arg)                                      
{
    if (!allocated)   allocate();
    int ntypes = atom->ntypes;
    int i, j;

    if (narg != 3 + ntypes)
        error->all(FLERR, "Incorrect args for pair coefficients");
    if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
        error->all(FLERR, "Incorrect args for pair coefficients");
    nelements_coeff = 0;
    for (i = 3; i < narg; i++) {
        if (strcmp(arg[i], "") == 0)   continue;
        for (j = 0; j < nelements_coeff; j++) {
            if (std::string(arg[i]) == elements_coeff[j])   break;
        }
        map[i - 2] = j;                                                         
        if (j == nelements_coeff) {
            elements_coeff[j] = std::string(arg[i]);
            nelements_coeff++;
        }
    }

    nparams = 0;                                                                
    params = new Param_ANNP[2];                                                                                        
    read_file(arg[2]);
    nparams++;                                                                  
    std::cout << "number of potentials.......: " << nparams << std::endl;
    for(int i = 0; i < nparams; i++)
        if(nelements_coeff != params[i].nelements)
            error->all(FLERR, "Incorrect args for pair coefficients");       

    cutmax = 0.0;  
    if (cutmax < params[0].cut)  cutmax = params[0].cut;                        

    int count = 0;
    for (i = 1; i <= ntypes; i++) {
        for (j = i; j <= ntypes; j++) {                                         
            if ((map[i] >= 0) && (map[j] >= 0)) {                            
                setflag[i][j] = 1;
                count++;
            }                                  
        }
    }
    if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");
}

/*---------------------------------------------------------------------
    init specfic to this pair style     
----------------------------------------------------------------------*/
void PairANNP::init_style()
{
    if (force->newton_pair == 0)
        error->all(FLERR, "Pair style Neural Network Potential requires newton pair on");
    
    //int irequest = neighbor->request(this, instance_me);
    //neighbor->requests[irequest]->half = 0;
    //neighbor->requests[irequest]->full = 1;
    neighbor->add_request(this, NeighConst::REQ_FULL);
}

/*---------------------------------------------------------------------
    init for one type pair i,j and corresponding j,i     
----------------------------------------------------------------------*/
double PairANNP::init_one(int i, int j)                                       
{
    if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");
    return cutmax;
}

/*---------------------------------------------------------------------
    read potential file    
----------------------------------------------------------------------*/
void PairANNP::read_file(char *filename)
{         
    Param_ANNP *file = &params[0];                  
                                                     
    if(comm->me == 0)                               
    {
        std::fstream fin;
        fin.open(filename, std::ios::in);
        if (!fin.is_open()) {
            error->one(FLERR, "Cannot open neural network potential file");
        }
        try {
            std::string t_string = "";
            for (int i = 0; i < 21 + nelements_coeff; i++) {
                getline(fin, t_string);
                if (i == 5) {
                    file->nelements = atoi(&t_string[0]);
                    Element* all_elem = new Element[file->nelements];           
                    file->all_elem = all_elem;                                  
                }
                if (i >= 6 && i < 6 + file->nelements) {
                    file->all_elem[i - 6].id_elem = atoi(&t_string[0]);

                    for (int j = 0; j < t_string.size(); j++) {
                        int j_next = j + 1;
                        if (isalpha(t_string[j])) {
                            file->all_elem[i - 6].elements += t_string[j];
                        }
                        if (t_string[j] == '\t' && isdigit(t_string[j_next])) {
                            file->all_elem[i - 6].mass = atof(&t_string[j_next]);
                        }
                    }
                }
                if (i == 8 + file->nelements) {
                    int num_para_ann = 0;
                    file->ntl = atoi(&t_string[0]);
                    num_para_ann++;
                    for (int j = 0; j < t_string.size(); j++) {                 
                        int j_next = j + 1;
                        if (t_string[j] == '\t' && isdigit(t_string[j_next]))
                        {
                            if (num_para_ann == 1)  file->nhl = atoi(&t_string[j_next]);
                            if (num_para_ann == 2)  file->nnod = atoi(&t_string[j_next]);
                            if (num_para_ann == 3)  file->nsf = atoi(&t_string[j_next]);
                            if (num_para_ann == 4)  file->npsf = atoi(&t_string[j_next]);
                            if (num_para_ann == 5)  file->ntsf = atoi(&t_string[j_next]);
                            if (num_para_ann == 6)  file->cut = atof(&t_string[j_next]);
                            num_para_ann++;
                        }
                    }
                }
                if (i == 11) {
                    int n_sf = file->nsf;
                    file->sfnor_cov = new double[n_sf];
                    file->sfnor_avg = new double[n_sf];
                }
                if (i >= 11 + file->nelements && i <= 12 + file->nelements) {   
                    int num_values = 0;
                    if (i == 11 + file->nelements)
                        file->sfnor_cov[num_values] = atof(&t_string[0]);
                    else
                        file->sfnor_avg[num_values] = atof(&t_string[0]);

                    num_values++;
                    for (int j = 0; j < t_string.size(); j++) {
                        int j_next = j + 1;
                        if (t_string[j] == '\t' && ((isdigit(t_string[j_next]) || (int)t_string[j_next] == 45))) {
                            if (i == 11 + file->nelements)
                                file->sfnor_cov[num_values] = atof(&t_string[j_next]);
                            else
                                file->sfnor_avg[num_values] = atof(&t_string[j_next]);
                            num_values++;
                        }
                    }
                }
                if (i == 15 + file->nelements) {                                 
                    int nact = 0;
                    int nlayer = file->ntl - 1;
                    file->flagact = new int[nlayer];                            
                    for (int j = 0; j < t_string.size(); j++) {
                        int j_next = j + 1;
                        if (t_string[j] == 'C' && t_string[j_next] == 'h') file->flagsym = 0;
                        if ((t_string[j] == 'B' && t_string[j_next] == 'e') || (t_string[j] == 'B' && t_string[j_next] == 'P')) file->flagsym = 1;
                        if (t_string[j] == 'C' && t_string[j_next] == 'u') file->flagsym = 2;

                        if (t_string[j] == 'l' && t_string[j_next] == 'i') file->flagact[nact++] = 0;
                        if (t_string[j] == 'h' && t_string[j_next] == 'y') file->flagact[nact++] = 1;
                        if (t_string[j] == 's' && t_string[j_next] == 'i') file->flagact[nact++] = 2;
                        if (t_string[j] == 'm' && t_string[j_next] == 'o') file->flagact[nact++] = 3;
                        if (t_string[j] == 't' && t_string[j_next] == 'a') file->flagact[nact++] = 4;
                    }
                }
                if (i == 18 + file->nelements || i == 19 + file->nelements || i == 20 + file->nelements) {
                    if (i == 18 + file->nelements)
                        file->e_scale = atof(&t_string[0]);
                    else if (i == 19 + file->nelements)
                        file->e_shift = atof(&t_string[0]);
                    else
                        file->e_atom = atof(&t_string[0]);
                }
            }

            int n_elem = file->nelements;
            int n_lay = file->ntl - 1;
            int n_nod = file->nnod;
            int n_sf = file->nsf;

            Annparm* all_annp = new Annparm[n_elem];
            for (int i = 0; i < n_elem; i++) {
                all_annp[i].weight_all = new double** [n_lay];
                all_annp[i].bias_all = new double** [n_lay];
                c_3d_matrix(n_lay, n_nod, n_sf, all_annp[i].weight_all);
                c_3d_matrix(n_lay, 1, n_nod, all_annp[i].bias_all);
            }
            file->all_annp = all_annp;                                          

            while (true) {
                int no_layer = 0;
                bool flag_wb = 0;
                getline(fin, t_string);

                int type_elem = 0;
                std::string name_elem = "";
                for (int i = 0; i < nelements_coeff; i++) {
                    if (t_string[0] == '#' && isupper(t_string[1])) {
                        for (int j = 0; j < t_string.size(); j++)
                            if (t_string[j] != '#')
                                name_elem += t_string[j];

                    }
                    if (name_elem == elements_coeff[i]) {
                        type_elem = i;
                    }
                }
                if (t_string[0] == '#' && isdigit(t_string[1])) {
                    for (int i = 0; i < t_string.size(); i++) {
                        if ((int)t_string[i] > 47 && (int)t_string[i] < 58)     
                        {
                            no_layer *= 10;
                            no_layer += (int)t_string[i] - 48;
                        }
                        if (t_string[i] == 'w')  flag_wb = 0;
                        if (t_string[i] == 'b')  flag_wb = 1;
                    }

                    int nrow_w = file->nnod, ncol_w = file->nnod, nrow_b = 1, ncol_b = file->nnod;   
                    if (no_layer == 1) {                                        
                        nrow_w = file->nnod;    ncol_w = file->nsf;
                    }
                    if (no_layer == file->ntl - 1) {                            
                        nrow_w = 1;    ncol_w = file->nnod;
                        nrow_b = 1;    ncol_b = 1;
                    }

                    int nol = no_layer - 1;                                     
                    if (flag_wb == 0) {
                        for (int i = 0; i < nrow_w; i++) {
                            int num_values = 0;
                            getline(fin, t_string);
                            file->all_annp[type_elem].weight_all[nol][i][num_values] = atof(&t_string[0]); num_values++;
                            for (int j = 0; j < t_string.size(); j++) {
                                int j_next = j + 1;
                                if (t_string[j] == '\t' && ((isdigit(t_string[j_next]) || (int)t_string[j_next] == 45))) {
                                    file->all_annp[type_elem].weight_all[nol][i][num_values] = atof(&t_string[j_next]);   num_values++;
                                }
                            }
                        }
                    }
                    else {                                                      
                        for (int i = 0; i < nrow_b; i++) {
                            int num_values = 0;
                            getline(fin, t_string);
                            file->all_annp[type_elem].bias_all[nol][i][num_values] = atof(&t_string[0]);   num_values++;
                            for (int j = 0; j < t_string.size(); j++) {
                                int j_next = j + 1;
                                if (t_string[j] == '\t' && ((isdigit(t_string[j_next]) || (int)t_string[j_next] == 45))) {
                                    file->all_annp[type_elem].bias_all[nol][i][num_values] = atof(&t_string[j_next]);   num_values++;
                                }
                            }
                        }
                    }
                }
                if (fin.peek() == EOF)  break;
            }
        }
        catch (TokenizerException &e) {
            error->one(FLERR, e.what());
        }
        fin.close();
    }

    MPI_Bcast(&file->nelements, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&file->cut, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&file->e_scale, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&file->e_shift, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&file->e_atom, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&file->flagsym, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&file->ntl, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&file->nhl, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&file->nnod, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&file->nsf, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&file->npsf, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&file->ntsf, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (comm->me != 0) {            
        int nele = file->nelements;
        int ntl = file->ntl - 1;
        int nsf = file->nsf;
        Element* all_elem = new Element[nele];
        Annparm* all_annp = new Annparm[nele];

        for (int i = 0; i < nele; i++) {
            all_elem[i].elements = "";
            all_annp[i].weight_all = new double** [ntl];
            all_annp[i].bias_all = new double** [ntl];

            c_3d_matrix(ntl, file->nnod, file->nsf, all_annp[i].weight_all);
            c_3d_matrix(ntl, 1, file->nnod, all_annp[i].bias_all);
        }
        file->all_elem = all_elem;
        file->all_annp = all_annp;
        file->flagact = new int[ntl];
        file->sfnor_cov = new double[nsf];
        file->sfnor_avg = new double[nsf];
    }

    int ntl = file->ntl - 1;
    int nnod = file->nnod;
    int nsf = file->nsf;
    for (int i = 0; i < file->nelements; i++) {
        int ns = 0;
        if (comm->me == 0) {
            ns = (int)strlen(&file->all_elem[i].elements[0]) + 1;
        }
        MPI_Bcast(&ns, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (comm->me != 0) file->all_elem[i].elements.resize(ns);

        MPI_Bcast(const_cast<char*>(file->all_elem[i].elements.data()), ns, MPI_CHAR, 0, MPI_COMM_WORLD);
        MPI_Bcast(&file->all_elem[i].id_elem, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&file->all_elem[i].mass, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(file->sfnor_cov, nsf, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(file->sfnor_avg, nsf, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        for (int j = 0; j < ntl; j++) {
            for (int k = 0; k < nnod; k++) {
                MPI_Bcast(file->all_annp[i].weight_all[j][k], nsf, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            }
            MPI_Bcast(file->all_annp[i].bias_all[j][0], nnod, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    }
    MPI_Bcast(file->flagact, ntl, MPI_INT, 0, MPI_COMM_WORLD);
}

/*---------------------------------------------------------------------
    other subfunctions
----------------------------------------------------------------------*/
void PairANNP::annp_fc(double rij, double Rc, double &fc, double &dfc) {                                             
    double coeff_a = MY_PI / Rc * rij;
    fc = 0.5 * (cos(coeff_a) + 1);
    dfc = -0.5 * MY_PI / Rc * sin(coeff_a);
}

void PairANNP::annp_Tx(double x, int npsf, double *Tx, double *dTx) {
    for (int i = 0; i < npsf; i++) {
        if (i == 0) {
            Tx[i] = 1;
            dTx[i] = 0;
        }
        else if (i == 1) {
            Tx[i] = x;
            dTx[i] = 1;
        }
        else {
            Tx[i] = 2 * x * Tx[i - 1] - Tx[i - 2];
            dTx[i] = 2 * Tx[i - 1] + 2 * x * dTx[i - 1] - dTx[i - 2];
        }
    }
}

void PairANNP::annp_dr_dij(int f_ijk, double rsqij, double* rij, double* dr_dj) {
    for (int i = 0; i < 3; i++)
        dr_dj[i] = pow(-1, f_ijk) * rij[i] / rsqij;
}

void PairANNP::annp_dct_djk(double rsqij, double rsqik, double *rij, 
                            double *rik, double cos_theta, 
                            double *dct_dj, double *dct_dk) {
    double B = rsqij * rsqik;
    double term1 = cos_theta / (rsqij * rsqij);
    double term2 = cos_theta / (rsqik * rsqik);
    for (int i = 0; i < 3; i++) {
        dct_dj[i] = (-1.0) * rik[i] / B + term1 * rij[i];
        dct_dk[i] = (-1.0) * rij[i] / B + term2 * rik[i];
    }
}

/*---------------------------------------------------------------------
    symmetry function (pair, trip)
----------------------------------------------------------------------*/
void PairANNP::annp_symmetry_pair(int j, double rij, double fcij, double dfcij, 
                                  double *dr_dj, double *sf_scale, double* G, 
                                  double*** dG_dij, Param_ANNP* params) {
    int npsf = params->npsf;
    double Rc = params->cut;
    double x;
    double* Tx = new double[npsf], * dTx = new double[npsf];
    memset(Tx, 0, sizeof(double) * npsf);
    memset(dTx, 0, sizeof(double) * npsf);

    x = 2 * rij / Rc - 1;
    annp_Tx(x, npsf, Tx, dTx);                                                  
    for (int m = 0; m < npsf; m++) {
        double t_avg = params->sfnor_avg[m];
        G[m] += sf_scale[m] * Tx[m] * fcij;
        double term1 = (dTx[m] * 2 / Rc * fcij + Tx[m] * dfcij) * sf_scale[m];
        for (int n = 0; n < 3; n++) {
            double t_dG_dj = term1 * dr_dj[n];
            dG_dij[j][m][n] += t_dG_dj;
        }
    }
    delete[]Tx;
    delete[]dTx;
}

void PairANNP::annp_symmetry_trip(int j, int k, double rij, double rik, double cos_theta, 
                                  double fcij, double dfcij, double fcik, double dfcik,
                                  double *xij, double *xik, double *dr_dj, double *sf_scale,
                                  double* G, double*** dG_dij, Param_ANNP* params) {
    int npsf = params->npsf;
    int ntsf = params->ntsf;
    double Rc = params->cut;
    double x;
    double dct_dj[3] = { 0 }, dct_dk[3] = { 0 }, dr_dk[3] = { 0 };
    double* Tx = new double[ntsf], * dTx = new double[ntsf];
    memset(Tx, 0, sizeof(double) * ntsf);
    memset(dTx, 0, sizeof(double) * ntsf);

    x = 0.5 * (cos_theta + 1);
    annp_Tx(x, ntsf, Tx, dTx); 
    annp_dr_dij(1, rik, xik, dr_dk); 
    annp_dct_djk(rij, rik, xij, xik, cos_theta, dct_dj, dct_dk);

    for (int n = 0; n < ntsf; n++) {
        double t_avg = params->sfnor_avg[n + npsf];
        G[n + npsf] += sf_scale[n + npsf] * Tx[n] * fcij * fcik;
        double term1 = dTx[n] * 0.5 * fcij * fcik;
        double term2 = Tx[n] * dfcij * fcik;
        double term3 = Tx[n] * fcij * dfcik;
        for (int m = 0; m < 3; m++) {
            double t_dG_dj = term1 * dct_dj[m] + term2 * dr_dj[m];
            double t_dG_dk = term1 * dct_dk[m] + term3 * dr_dk[m];
            double t_dG_di = (-1.0) * (t_dG_dj + t_dG_dk);
            if (fabs(t_dG_di) < 1.0e-12)
                t_dG_di = 0.0;
            dG_dij[j][n + npsf][m] += sf_scale[n + npsf] * t_dG_dj;
            dG_dij[k][n + npsf][m] += sf_scale[n + npsf] * t_dG_dk;
        }

    }
    delete[]Tx;
    delete[]dTx;
}

/*---------------------------------------------------------------------
    feed_forward function
----------------------------------------------------------------------*/
void PairANNP::dot_add_wxb(int nr, int nc, double** w, double* x, 
                           double** b, double* ans) {                           
    for (int i = 0; i < nr; i++) {
        for (int j = 0; j < nc; j++)
            ans[i] += w[i][j] * x[j];
        ans[i] += b[0][i];
    }
}

void PairANNP::annp_actf(int flag_l, int flag_act, int ntl, int nr_w, 
                         double* wxb, double* hidly, double** hidly_d) {        
    double coeff_a = 1.7159;
    double coeff_b = 0.666666666666667;
    double coeff_c = 0.1;
    double t_tanhx = 0.0;
    for (int i = 0; i < nr_w; i++) {
        if (flag_act == 0) {                                                    
            hidly[i] = wxb[i];
            hidly_d[i][i] = 1;
        }
        if (flag_act == 1) {                                                    
            hidly[i] = tanh(wxb[i]);
            hidly_d[i][i] = 1 - hidly[i] * hidly[i];                            
        }
        if (flag_act == 2) {                                                    
            hidly[i] = 1.0 / (1.0 + exp(wxb[i]));
            hidly_d[i][i] = hidly[i] * (1 - hidly[i]);
        }
        if (flag_act == 3) {                                                    
            t_tanhx = tanh(coeff_b * wxb[i]);
            hidly[i] = coeff_a * t_tanhx;
            hidly_d[i][i] = coeff_a * (1.0 - t_tanhx * t_tanhx) * coeff_b;
        }
        if (flag_act == 4) {                                                    
            t_tanhx = tanh(coeff_b * wxb[i]);
            hidly[i] = coeff_a * t_tanhx + coeff_c * wxb[i];
            hidly_d[i][i] = coeff_a * (1.0 - t_tanhx * t_tanhx) * coeff_b + coeff_c;
        }
    }
}

void PairANNP::annp_feed_forward(int itype, double* G, double* dE_dG, int eflag, 
                                 double& evdwl, Param_ANNP* params) {           
    int ntl = params->ntl;
    int nntl = ntl - 1;
    int nsf = params->nsf;
    int nnod = params->nnod;

    double** tdE_dG = new double* [nsf];
    double** tdE_dG1 = new double* [nsf];
    double** hidly = new double* [nntl];                                          
    double** hidly_d = new double* [nnod];                                      
    double** hidly_dw = new double* [nnod];                                     
    c_2d_matrix(nsf, nsf, tdE_dG);
    c_2d_matrix(nsf, nsf, tdE_dG1);
    c_2d_matrix(ntl - 1, nnod, hidly);
    c_2d_matrix(nnod, nnod, hidly_d);
    c_2d_matrix(nnod, nsf, hidly_dw);
    
    for (int i = 0; i < nsf; i++)                                               
        for (int j = 0; j < nsf; j++)
            if (i == j) tdE_dG[i][j] = 1.0;

    for (int i = 0; i < ntl - 1; i++) {
        int flag_act = params->flagact[i];                                      
        double* wxb = new double[nnod];                                           
        memset(wxb, 0, sizeof(double) * nnod);
        double** weight = params->all_annp[itype].weight_all[i];
        double** bias = params->all_annp[itype].bias_all[i];
        int nr_w = nnod, nc_w = nnod;
        if (i == 0) {
            nc_w = nsf;
            dot_add_wxb(nr_w, nc_w, weight, G, bias, wxb);
        }
        else {
            if (i == ntl - 2)    nr_w = 1;
            dot_add_wxb(nr_w, nc_w, weight, hidly[i - 1], bias, wxb);
        }
        annp_actf(i, flag_act, ntl, nr_w, wxb, hidly[i], hidly_d);

        dot_mat_2d(nr_w, nr_w, nc_w, hidly_d, weight, hidly_dw);
        dot_mat_2d(nr_w, nc_w, nsf, hidly_dw, tdE_dG, tdE_dG1);
        for (int j = 0; j < nr_w; j++) {
            for (int k = 0; k < nsf; k++)
                tdE_dG[j][k] = tdE_dG1[j][k];
        }
        delete[]wxb;
    }
    if (eflag) {
        evdwl = hidly[ntl - 2][0];
        evdwl = params->e_scale * evdwl + params->e_shift + params->e_atom;
    }
    for (int i = 0; i < nsf; i++) {
        dE_dG[i] = tdE_dG[0][i];
    }

    d_2d_matrix(nsf, tdE_dG);
    d_2d_matrix(nsf, tdE_dG1);
    d_2d_matrix(ntl - 1, hidly);
    d_2d_matrix(nnod, hidly_d);                                                 
    d_2d_matrix(nnod, hidly_dw);
}


/*---------------------------------------------------------------------
    mathematic function for matrix
----------------------------------------------------------------------*/
void PairANNP::copy_i(int n, double s, double* v, double* ans) {
    for (int i = 0; i < n; i++)
        ans[i] = s*v[i];
}

void PairANNP::copy_i3(int index, int n, double s, double** v, double** ans) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < 3; j++) {
            ans[i + index][j] = s * v[i][j];
        }
}

void PairANNP::dot_mat_2d(int nr1, int nc1, int nc2, double** v1, 
                          double** v2, double** ans) {
    for (int i = 0; i < nr1; i++) {
        for (int j = 0; j < nc2; j++) {
            double t_ans = 0.0;
            for (int k = 0; k < nc1; k++) {
                t_ans += v1[i][k] * v2[k][j];
            }
            ans[i][j] = t_ans;
        }
    }
}
