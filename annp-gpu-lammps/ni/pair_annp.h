//* Host c++ *------------------------------------------
//      Artifical Neural Network Potential
//             CPU version
//______________________________________________________        
//  begin:  Mon Oct 23, 2022
//  email:  meng_zhang@metall.t.u-tokyo.ac.jp
//          junya_inoue@metall.t.u-tokyo.ac.jp   
//______________________________________________________
//------------------------------------------------------

#ifdef PAIR_CLASS
// clang-format off
PairStyle(annp, PairANNP);
// clang-format on
#else

#ifndef LMP_PAIR_ANNP_H
#define LMP_PAIR_ANNP_H

#include "pair.h"

namespace LAMMPS_NS {

   class PairANNP : public Pair {
   public:
      PairANNP(class LAMMPS *);
      virtual ~PairANNP();
      virtual void compute(int, int);                                          
      void settings(int, char**);
      void coeff(int, char**);
      double init_one(int, int);
      virtual void init_style();
      
      /*---------------------------------------------------------------------
          potentials as array data (pointer for gpu)
      ----------------------------------------------------------------------*/
   protected:
      int nelements_coeff;                                                     
      double cutmax;                                                           
      std::string* elements_coeff;                                           
      struct Element {
          int id_elem;                                                          
          double mass;
          std::string elements;                                                
          Element() {                                                           
              id_elem = 1;
              mass = 0.0;
              elements = "";
          }
      };
      struct Annparm {
          double*** weight_all, *** bias_all;                                  
      };
      struct Param_ANNP {
          Element* all_elem;
          Annparm* all_annp;
          int nelements;                                                        
          double cut;
          double e_scale, e_shift, e_atom;                                   
          int flagsym, *flagact;                                                
          int ntl, nhl, nnod, nsf, npsf, ntsf;                                       
          double* sf_min, * sf_max;                                             
          double** sym_coerad, ** sym_coeang;
      };

      Param_ANNP* params;
      virtual void read_file(char*);
      virtual void allocate();
      const double CFLENGTH = 1.889726;
      const double CFFORCE = 51.422515;

      /*---------------------------------------------------------------------
          sub_functions
      ----------------------------------------------------------------------*/
      // used for compute () function
      virtual void annp_symmetry_pair(int, double, double*, double*, double***, Param_ANNP*);
      virtual void annp_symmetry_trip(int, int, double, double, double, double, double*, double*, 
                                      double*, double*, double*, double***, Param_ANNP*);
      virtual void annp_feed_forward(int, double*, double*, int, double&, Param_ANNP*);
      virtual void copy_i(int, double, double*, double*);
      virtual void copy_i3(int, int, double, double**, double**);

      // used for above function
      virtual void annp_fc(double, double, double&, double&);
      virtual void annp_Tx(double, int, double*, double*);
      virtual void annp_dr_dij(int, double, double*, double*);
      virtual void annp_dct_djk(double, double, double*, double*, double, double*, double*);
      virtual void dot_add_wxb(int, int, double**, double*, double**, double*);
      virtual void annp_actf(int, int, int, int, double*, double*, double**);
      virtual void dot_mat_2d(int, int, int, double**, double**, double**);

      /*---------------------------------------------------------------------
        create/delete the 2-D and 3-D matrix nad initialization 0
      ----------------------------------------------------------------------*/
      // creating the 2D and 3D matrix
      inline void c_2d_matrix(int row, int col, double** v) {
          for (int i = 0; i < row; i++) {
              v[i] = new double[col];
              memset(v[i], 0, sizeof(double) * col);
          }
      }
      inline void c_3d_matrix(int n1d, int n2d, int n3d, double*** v) {
          for (int i = 0; i < n1d; i++) {
              v[i] = new double* [n2d];
              for (int j = 0; j < n2d; j++) {
                  v[i][j] = new double[n3d];
                  memset(v[i][j], 0, sizeof(double) * n3d);
              }
          }
      }

      // free the 2-D and 3-D matrix
      inline void d_2d_matrix(int row, double** v) {
          for (int i = 0; i < row; i++)
              delete[]v[i];
          delete[]v;
      }
      inline void d_3d_matrix(int n1d, int n2d, double*** v) {
          for (int i = 0; i < n1d; i++)
              for (int j = 0; j < n2d; j++)
                  delete[]v[i][j];
          for (int i = 0; i < n1d; i++)
              delete[]v[i];
          delete[]v;
      }

   };

}    // namespace LAMMPS_NS

#endif
#endif

