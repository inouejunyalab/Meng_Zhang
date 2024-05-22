//* Host c++ *------------------------------------------
//      Physics-informed Neural Network Potential
//             CPU version
//______________________________________________________        
//  begin:  Sun July 9, 2023
//  email:  meng_zhang@metall.t.u-tokyo.ac.jp
//          junya_inoue@metall.t.u-tokyo.ac.jp  
//______________________________________________________
//------------------------------------------------------

#include "pair_anna_adp.h"

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

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace MathSpecial;
using namespace MathExtra;

/*----------------------------------------------------------------------*/
PairANNA_ADP::PairANNA_ADP(LAMMPS *lmp) : Pair(lmp) {
    restartinfo = 0;                                                            
    one_coeff = 1;                                                              
    manybody_flag = 1;                                                          

    elements_coeff = nullptr;
    params = nullptr;

    comm_forward = 1;                                                           
}
/*---------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
----------------------------------------------------------------------*/
PairANNA_ADP::~PairANNA_ADP() {
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
void PairANNA_ADP::compute(int eflag, int vflag) {
    int i, j, k, ii, jj, kk, inum, jnum;
    int itype, jtype, ktype, ritype, rjtype, rktype;
    double xtmp, ytmp, ztmp, evdwl;
    int* ilist, * jlist, * numneigh, ** firstneigh;
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

    // calculate for each atom on this proc
    int nout = params[0].nout;
    int nsf = params[0].nsf;
    int npsf = params[0].npsf;
    int ntsf = params[0].ntsf;
    double Rc = params[0].cut;
    double coeff_b = MY_PI / Rc;
    double* gparams = params[0].gparams;
    double A0 = gparams[0], yy = gparams[1], gamma = gparams[2], C0 = gparams[3];
    double c1F = gparams[4], c2F = gparams[5], V0 = gparams[6], b1 = gparams[7]; 
    double b2 = gparams[8], delta = gparams[9], r0 = gparams[10], r1 = gparams[11];
    double hc = gparams[12], d1 = gparams[13], q1 = gparams[14], d3 = gparams[15], q3 = gparams[16];
    double E_base = params[0].e_base, E_scal = params[0].e_scal;
    
    // starting the calculation
    for (ii = 0; ii < inum; ii++) {
        double rsqij, rsqik, rij, rik, cos_theta, fcij, fcik;
        double xij[3], xik[3], rij_unit[3], rik_unit[3];
        i = ilist[ii];                                                         
        ritype = type[i];                                                      
        itype = map[ritype];                                                   
        jnum = numneigh[i];                                                    
        jlist = firstneigh[i];                                                 
        xtmp = x[i][0];
        ytmp = x[i][1];
        ztmp = x[i][2];

        double* G = new double[nsf];
        double* lparams = new double[nout];
        double** all_xij = new double* [jnum];                             
        c_2d_matrix(jnum, 4, all_xij);
        memset(G, 0, sizeof(double) * nsf);
        memset(lparams, 0, sizeof(double) * nout);

        for (jj = 0; jj < jnum; jj++) {
            j = jlist[jj];
            j &= NEIGHMASK;                                                
            rjtype = type[j];
            jtype = map[rjtype];                                           

            xij[0] = xtmp - x[j][0];
            xij[1] = ytmp - x[j][1];
            xij[2] = ztmp - x[j][2];
            rsqij = xij[0] * xij[0] + xij[1] * xij[1] + xij[2] * xij[2];
            all_xij[jj][0] = xij[0];                                         
            all_xij[jj][1] = xij[1];
            all_xij[jj][2] = xij[2];
            all_xij[jj][3] = sqrt(rsqij);
            if (rsqij > cutsq[ritype][rjtype] || rsqij < 1.0e-12)   continue;

            const double rijinv = 1.0 / sqrt(dot3(xij, xij));                
            scale3(rijinv, xij, rij_unit);                                   
            rij = all_xij[jj][3];
            double fcij = 0.5 * (cos(coeff_b * rij) + 1.0);
            anna_adp_symmetry_pair(rij, fcij, G, &params[0]);
            for (kk = jj + 1; kk < jnum; kk++) {
                k = jlist[kk];
                rktype = type[k];
                ktype = map[rktype];

                xik[0] = xtmp - x[k][0];
                xik[1] = ytmp - x[k][1];
                xik[2] = ztmp - x[k][2];
                rsqik = xik[0] * xik[0] + xik[1] * xik[1] + xik[2] * xik[2];
                if (rsqik > cutsq[ritype][rktype] || rsqik < 1.0e-12)   continue;

                const double rikinv = 1.0 / sqrt(dot3(xik, xik));
                scale3(rikinv, xik, rik_unit);
                cos_theta = dot3(rij_unit, rik_unit);
                rik = sqrt(rsqik);
                double fcik = 0.5 * (cos(coeff_b * rik) + 1.0);
                anna_adp_symmetry_trip(cos_theta, fcij, fcik, G, &params[0]);
            }
        }

        // getting the local parameters
        anna_adp_feed_forward(itype, lparams, G, &params[0]);               
        double d2 = lparams[0], q2 = lparams[1];                            

        // energy and force
        double* mu_i = new double[3], ** lambda_i = new double* [3];
        double rho_i = 0.0;
        double coeff_repul = V0 / (b2 - b1);
        double adp_repul_eng = 0.0;
        memset(mu_i, 0, sizeof(double) * 3);
        c_2d_matrix(3, 3, lambda_i);

        for (jj = 0; jj < jnum; jj++) {
            if (all_xij[jj][3] > Rc || all_xij[jj][3] < 1.0e-12)
                continue;

            double stpf_x = (all_xij[jj][3] - Rc) / hc;
            double adp_stpf = pow(stpf_x, 4) / (1 + pow(stpf_x, 4));
            double adp_u = adp_stpf * (d1 * exp(-d2 * all_xij[jj][3]) + d3);
            double adp_w = adp_stpf * (q1 * exp(-q2 * all_xij[jj][3]) + q3);
            mu_i[0] += adp_u * all_xij[jj][0];
            mu_i[1] += adp_u * all_xij[jj][1];
            mu_i[2] += adp_u * all_xij[jj][2];
            for (int row = 0; row < 3; row++) {
                for (int col = 0; col < 3; col++) {
                    lambda_i[row][col] += adp_w * all_xij[jj][row] * all_xij[jj][col];
                }
            }
            double rho_z = all_xij[jj][3] - r0;
            double exp_z = exp(-gamma * rho_z);
            rho_i += adp_stpf * (A0 * pow(rho_z, yy) * exp_z * (1 + exp_z) + C0);

            double repul_z = all_xij[jj][3] / r1;
            adp_repul_eng += adp_stpf * (coeff_repul * (b2 / pow(repul_z, b1) - b1 / pow(repul_z, b2)) + delta);
        }

        double v_i = lambda_i[0][0] + lambda_i[1][1] + lambda_i[2][2];
        double sum_mu_i = 0.0, sum_lambda_i = 0.0;
        for (int row = 0; row < 3; row++) {
            sum_mu_i += mu_i[row] * mu_i[row];
            for (int col = 0; col < 3; col++) {
                sum_lambda_i += pow(lambda_i[row][col], 2);
            }
        }
        double f_v = -1.0 / 3.0 * v_i;                                                                
        double rep_coeff = V0 / (b2 - b1);                                                            
        double adp_angular_eng = 0.5 * sum_mu_i + 0.5 * sum_lambda_i - 1.0 / 6.0 * v_i * v_i;
        double adp_embed_eng = c1F * sqrt(rho_i) + c2F * pow(rho_i, 2);
        evdwl = 0.5 * (adp_repul_eng) + adp_embed_eng + adp_angular_eng + E_base;

        //-------------force calculation
        for (jj = 0; jj < jnum; jj++) {
            j = jlist[jj];
            j &= NEIGHMASK;                                                                         
            if (all_xij[jj][3] > Rc || all_xij[jj][3] < 1.0e-12)
                continue;

            double xi = all_xij[jj][0], yi = all_xij[jj][1], zi = all_xij[jj][2], rij = all_xij[jj][3];
            // step function
            double stpf_x = (rij - Rc) / hc;
            double adp_stpf_t1 = 1 + pow(stpf_x, 4);
            double adp_stpf = pow(stpf_x, 4) / adp_stpf_t1;
            double d_adp_stpf = 4 * pow(stpf_x, 3) / pow(adp_stpf_t1, 2) / hc;

            // rho function
            double rho_z = rij - r0;
            double exp_z = exp(-gamma * rho_z);
            double z_yy = A0 * pow(rho_z, yy);
            double ga_zyy = z_yy * gamma;
            double d_adp_rho = exp_z * (1.0 + exp_z) * (z_yy * (d_adp_stpf + adp_stpf * yy / rho_z) - ga_zyy) + C0 * d_adp_stpf - ga_zyy * exp_z * exp_z;
            
            // embed part
            double d_embed_eng = (0.5 * c1F * pow(rho_i, -0.5) + 2.0 * c2F * rho_i) * d_adp_rho;
            
            // pair part
            double repul_z = rij / r1;
            double zb1 = pow(repul_z, b1);
            double zb2 = pow(repul_z, b2);
            double drep_t = b2 * b1 / r1;
            double rep_t1 = rep_coeff * (b2 / zb1 - b1 / zb2) + delta;
            double d_repul_eng = d_adp_stpf * rep_t1 +  adp_stpf * rep_coeff * (drep_t / repul_z * (-1.0 / zb1 + 1.0 / zb2));

            // angular part----------------------------------old implementation
            double adp_u_term = d1 * exp(-d2 * rij);
            double adp_w_term = q1 * exp(-q2 * rij);
            double adp_u = adp_stpf * (adp_u_term + d3);
            double adp_w = 2.0 * adp_stpf * (adp_w_term + q3);
            double d_adp_u = d_adp_stpf * (adp_u_term + d3) + adp_stpf * (-d2 * adp_u_term);
            double d_adp_w = d_adp_stpf * (adp_w_term + q3) + adp_stpf * (-q2 * adp_w_term);
            double d_angular_lamb1 = d_adp_w * (lambda_i[0][0] * xi * xi + lambda_i[1][1] * yi * yi + lambda_i[2][2] * zi * zi);
            double d_angular_lamb2 = d_adp_w * (lambda_i[0][1] * xi * yi + lambda_i[0][2] * xi * zi + lambda_i[1][2] * yi * zi) * 2.0 + d_angular_lamb1;
            double df_term1 = 0.5 * d_repul_eng + d_embed_eng + d_adp_u * (mu_i[0] * xi + mu_i[1] * yi + mu_i[2] * zi) + d_angular_lamb2;
            double df_term3 = f_v * (d_adp_w * rij + adp_w);

            double fx = df_term1 * xi / rij +adp_w * (yi * lambda_i[0][1] + zi * lambda_i[0][2] + xi * lambda_i[0][0]) + mu_i[0] * adp_u + xi * df_term3;
            double fy = df_term1 * yi / rij +adp_w * (yi * lambda_i[1][1] + zi * lambda_i[1][2] + xi * lambda_i[0][1]) + mu_i[1] * adp_u + yi * df_term3;
            double fz = df_term1 * zi / rij +adp_w * (yi * lambda_i[1][2] + zi * lambda_i[2][2] + xi * lambda_i[0][2]) + mu_i[2] * adp_u + zi * df_term3;

            f[i][0] -= fx;
            f[i][1] -= fy;
            f[i][2] -= fz;
            f[j][0] += fx;
            f[j][1] += fy;
            f[j][2] += fz;
            if (evflag) {
                ev_tally_xyz(i, j, nlocal, force->newton_pair, 0.0, 0.0, -fx, -fy, -fz, all_xij[jj][0], all_xij[jj][1], all_xij[jj][2]);
            }
        }
        if (eflag) {
            if (eflag_global) eng_vdwl += evdwl;
            if (eflag_atom) eatom[i] += evdwl;
        }
        // free all array
        delete[] G;
        delete[] mu_i;
        delete[] lparams;
        d_2d_matrix(3, lambda_i);
        d_2d_matrix(jnum, all_xij);
    }

    if (vflag_fdotr)    virial_fdotr_compute();                        
}

/*---------------------------------------------------------------------
                        allocate all arrys
----------------------------------------------------------------------*/
void PairANNA_ADP::allocate() {
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
void PairANNA_ADP::settings(int narg, char **arg) {
    if (narg != 0)    error->all(FLERR, "Illegal pair_style command");
}

/*---------------------------------------------------------------------
              set coeffs for one or more type pairs     
----------------------------------------------------------------------*/
void PairANNA_ADP::coeff(int narg, char** arg) {
    if (!allocated)   allocate();
    int ntypes = atom->ntypes;
    int i, j;

    if (narg != 3 + ntypes)
        error->all(FLERR, "Incorrect args for pair coefficients");
    if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
        error->all(FLERR, "Incorrect args for pair coefficients");
    // map the Ith atom tpe is, -1 if NULL
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

    // read_file from the potential
    nparams = 0;                                                           
    params = new ANNAPARA[2];                                              
    read_file(arg[2]);
    nparams++;                                                             
    std::cout << "number of potentials.......: " << comm->me << " " << nparams << std::endl;
    for(int i = 0; i < nparams; i++)
        if(nelements_coeff != params[i].nelements)
            error->all(FLERR, "Incorrect args for pair coefficients");       

    // setting the /*cutoff*/ for different elements
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
void PairANNA_ADP::init_style() {
    if (force->newton_pair == 0)
        error->all(FLERR, "Pair style Neural Network Potential requires newton pair on");
    
    // need a full neighbor list
    //int irequest = neighbor->request(this, instance_me);
    //neighbor->requests[irequest]->half = 0;
    //neighbor->requests[irequest]->full = 1;
    neighbor->add_request(this, NeighConst::REQ_FULL);
}

/*---------------------------------------------------------------------
           init for one type pair i,j and corresponding j,i     
----------------------------------------------------------------------*/
double PairANNA_ADP::init_one(int i, int j) {
    if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");
    return cutmax;
}

/*---------------------------------------------------------------------
                         read potential file    
----------------------------------------------------------------------*/
void PairANNA_ADP::read_file(char *filename) {
    ANNAPARA*file = &params[0];                    
                                                   
    //read potential file
    if(comm->me == 0) {
        std::fstream fin;
        fin.open(filename, std::ios::in);
        if (!fin.is_open()) {
            error->one(FLERR, "Cannot open physically informed neural network potential file");
        }
        // start reading
        try {
            std::string t_string = "";
            for (int i = 0; i < 19 + nelements_coeff; i++) {
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
                    int num_para_anna = 0;
                    file->ntl = atoi(&t_string[0]);
                    num_para_anna++;
                    for (int j = 0; j < t_string.size(); j++) {                                  
                        int j_next = j + 1;
                        if (t_string[j] == '\t' && isdigit(t_string[j_next])) {
                            if (num_para_anna == 1)  file->nhl = atoi(&t_string[j_next]);
                            if (num_para_anna == 2)  file->nnod = atoi(&t_string[j_next]);
                            if (num_para_anna == 3)  file->nout = atoi(&t_string[j_next]);
                            if (num_para_anna == 4)  file->nsf = atoi(&t_string[j_next]);
                            if (num_para_anna == 5)  file->npsf = atoi(&t_string[j_next]);
                            if (num_para_anna == 6)  file->ntsf = atoi(&t_string[j_next]);
                            if (num_para_anna == 7)  file->cut = atof(&t_string[j_next]);
                            num_para_anna++;
                        }
                    }
                }
                if (i == 11 + file->nelements) {                                                 
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
                if (i == 14 + file->nelements) {
                    file->e_base = atof(&t_string[0]);
                    for (int j = 0; j < t_string.size(); j++) {                                
                        int j_next = j + 1;
                        if (t_string[j] == '\t' && isdigit(t_string[j_next])) {
                            file->e_scal = atof(&t_string[j_next]);
                        }
                    }
                }
                if (i == 17 + file->nelements) {
                    file->ngp = atoi(&t_string[0]);
                }
                if (i == 18 + file->nelements) {                                               
                    file->gparams = new double[file->ngp];
                    int id = 0;
                    file->gparams[id] = atof(&t_string[0]); id++;
                    for (int j = 0; j < t_string.size(); j++) {
                        int j_next = j + 1;
                        if (t_string[j] == '\t' && ((isdigit(t_string[j_next]) || (int)t_string[j_next] == 45))) {
                            file->gparams[id] = atof(&t_string[j_next]);    id++;
                        }
                    }
                }
            }

            // for reading the weight and bias
            int n_elem = file->nelements;
            int n_lay = file->ntl - 1;
            int n_nod = file->nnod;
            int n_sf = file->nsf;
            ANNA* all_anna = new ANNA[n_elem];
            for (int i = 0; i < n_elem; i++) {
                all_anna[i].weight_all = new double** [n_lay];
                all_anna[i].bias_all = new double** [n_lay];
                c_3d_matrix(n_lay, n_nod, n_sf, all_anna[i].weight_all);
                c_3d_matrix(n_lay, 1, n_nod, all_anna[i].bias_all);
            }
            file->all_anna = all_anna;
            while (true) {
                int no_layer = 0;
                bool flag_wb = 0;
                getline(fin, t_string);

                int type_elem = 0;
                std::string name_elem = "";
                for (int i = 0; i < nparams; i++) {
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
                        if ((int)t_string[i] > 47 && (int)t_string[i] < 58) {                      
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
                        nrow_w = file->nout;    ncol_w = file->nnod;
                        nrow_b = 1;    ncol_b = file->nout;
                    }
                    int n1d = no_layer - 1;
                    if (flag_wb == 0) {
                        for (int i = 0; i < nrow_w; i++) {
                            int num_values = 0;
                            getline(fin, t_string);
                            file->all_anna[type_elem].weight_all[n1d][i][num_values] = atof(&t_string[0]); num_values++;
                            for (int j = 0; j < t_string.size(); j++) {
                                int j_next = j + 1;
                                if (t_string[j] == '\t' && ((isdigit(t_string[j_next]) || (int)t_string[j_next] == 45))) {
                                    file->all_anna[type_elem].weight_all[n1d][i][num_values] = atof(&t_string[j_next]);   num_values++;
                                }
                            }
                        }
                    }
                    else {                                                                      
                        for (int i = 0; i < nrow_b; i++) {
                            int num_values = 0;
                            getline(fin, t_string);
                            file->all_anna[type_elem].bias_all[n1d][i][num_values] = atof(&t_string[0]);   num_values++;
                            for (int j = 0; j < t_string.size(); j++) {
                                int j_next = j + 1;
                                if (t_string[j] == '\t' && ((isdigit(t_string[j_next]) || (int)t_string[j_next] == 45))) {
                                    file->all_anna[type_elem].bias_all[n1d][i][num_values] = atof(&t_string[j_next]);   num_values++;
                                }
                            }
                        }
                    }
                }
                if (fin.peek() == EOF)
                    break;
            }
        }
        catch (TokenizerException &e) {
            error->one(FLERR, e.what());
        }
        fin.close();
    }

    // broadcate all the paramets 
    std::cout << "MPI_checking....." << std::endl;
    MPI_Bcast(&file->nelements, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&file->cut, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&file->e_base, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&file->e_scal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&file->flagsym, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&file->ntl, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&file->nhl, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&file->nnod, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&file->nout, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&file->nsf, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&file->npsf, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&file->ntsf, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&file->ngp, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // broadcast potential potential
    if (comm->me != 0) {
        int n_ele = file->nelements;
        int ntl = file->ntl - 1;
        int nsf = file->nsf;
        int ngp = file->ngp;
        Element* all_elem = new Element[n_ele];
        ANNA* all_anna = new ANNA[n_ele];

        for (int i = 0; i < n_ele; i++) {
            all_elem[i].elements = "";
            all_anna[i].weight_all = new double** [ntl];
            all_anna[i].bias_all = new double** [ntl];

            c_3d_matrix(ntl, file->nnod, file->nsf, all_anna[i].weight_all);
            c_3d_matrix(ntl, 1, file->nnod, all_anna[i].bias_all);
        }
        file->all_elem = all_elem;
        file->all_anna = all_anna;
        file->flagact = new int[ntl];
        file->gparams = new double[ngp];
    }

    int ntl = file->ntl - 1;
    int nnod = file->nnod;
    int nsf = file->nsf;
    int ngp = file->ngp;
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

        for (int j = 0; j < ntl; j++) {
            for (int k = 0; k < nnod; k++) {
                MPI_Bcast(file->all_anna[i].weight_all[j][k], nsf, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            }
            MPI_Bcast(file->all_anna[i].bias_all[j][0], nnod, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    }
    MPI_Bcast(file->flagact, ntl, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(file->gparams, ngp, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

/*---------------------------------------------------------------------
                          other subfunctions
----------------------------------------------------------------------*/
void PairANNA_ADP::anna_adp_Tx(double x, int n, double *Tx) {
    for (int i = 0; i < n; i++) {
        if (i == 0)
            Tx[i] = 1;
        else if (i == 1)
            Tx[i] = x;
        else
            Tx[i] = 2 * x * Tx[i - 1] - Tx[i - 2];
    }
}

/*---------------------------------------------------------------------
                    symmetry function (pair, trip)
----------------------------------------------------------------------*/
void PairANNA_ADP::anna_adp_symmetry_pair(double rij, double fcij, double* G, ANNAPARA* params) {
    int npsf = params->npsf;
    double Rc = params->cut;
    double* Tx = new double[npsf];
    memset(Tx, 0, sizeof(double) * npsf);

    double x = 2 * rij / Rc - 1;
    anna_adp_Tx(x, npsf, Tx);                                                                 
    for (int m = 0; m < npsf; m++) {
        G[m] += Tx[m] * fcij;
    }
    delete[]Tx;
}

void PairANNA_ADP::anna_adp_symmetry_trip(double cos_theta, double fcij, double fcik,
                                          double* G, ANNAPARA* params) {
    int npsf = params->npsf;
    int ntsf = params->ntsf;
    double* Tx = new double[ntsf];
    memset(Tx, 0, sizeof(double) * ntsf);

    double x = 0.5 * (cos_theta + 1);                                                         
    anna_adp_Tx(x, ntsf, Tx);
    for (int n = 0; n < ntsf; n++) {
        G[n + npsf] += Tx[n] * fcij * fcik;
    }
    delete[]Tx;
}

/*---------------------------------------------------------------------
                       feed_forward function
----------------------------------------------------------------------*/
void PairANNA_ADP::dot_add_wxb(int nr, int nc, double** w, double* x, 
                               double** b, double* ans) {                        
    for (int i = 0; i < nr; i++) {
        for (int j = 0; j < nc; j++)
            ans[i] += w[i][j] * x[j];
        ans[i] += b[0][i];
    }
}

void PairANNA_ADP::anna_adp_actf(int flag_l, int flag_act, int ntl, int nr_w, 
                                 double* wxb, double* hidly) {                   
    double coeff_a = 1.7;
    double coeff_b = 0.3;
    double t_tanhx = 0.0;
    for (int i = 0; i < nr_w; i++) {
        if (flag_act == 0) {                            
            hidly[i] = wxb[i];
        }
        if (flag_act == 1) {                            
            hidly[i] = tanh(wxb[i]);
        }
        if (flag_act == 2) {                            
            hidly[i] = 1.0 / (1.0 + exp(wxb[i]));
        }
        if (flag_act == 3) {                           
            t_tanhx = tanh(coeff_b * wxb[i]);
            hidly[i] = coeff_a * t_tanhx;
        }
        if (flag_act == 4) {                           
            t_tanhx = tanh(coeff_b * wxb[i]);
            hidly[i] = coeff_a * t_tanhx;
        }
    }
}

void PairANNA_ADP::anna_adp_feed_forward(int itype, double* lparams,
                                         double* G, ANNAPARA* params) {     
    int ntl = params->ntl;
    int nsf = params->nsf;
    int nnod = params->nnod;
    double** hidly = new double* [ntl - 1];                                 
    c_2d_matrix(ntl - 1, nnod, hidly);

    double* wxb = new double[nnod];                                          
    for (int i = 0; i < ntl - 1; i++) {                                      
        memset(wxb, 0, sizeof(double) * nnod);
        double** weight = params->all_anna[itype].weight_all[i];
        double** bias = params->all_anna[itype].bias_all[i];
        int nr_w = nnod, nc_w = nnod;
        if (i == 0) {
            nc_w = nsf;
            dot_add_wxb(nr_w, nc_w, weight, G, bias, wxb);
        }
        else {
            if (i == ntl - 2)    nr_w = params->nout;
            dot_add_wxb(nr_w, nc_w, weight, hidly[i - 1], bias, wxb);
        }
        int flag_act = params->flagact[i];                                   
        anna_adp_actf(i, flag_act, ntl, nr_w, wxb, hidly[i]);
    }

    for (int i = 0; i < params->nout; i++)                                   
        lparams[i] = hidly[ntl - 2][i];

    delete[]wxb;
    d_2d_matrix(ntl - 1, hidly);
}
