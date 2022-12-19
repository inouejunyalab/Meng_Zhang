//* Host c++ *------------------------------------------
//      Artifical Neural Network Potential
//             CPU version
//______________________________________________________        
//  begin:  Wed February 16, 2022 (1/16/2022)
//  email:  
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
      virtual void compute(int, int);                                           // virtual keywords allow derived can rewrite the function
      void settings(int, char**);
      void coeff(int, char**);
      double init_one(int, int);
      virtual void init_style();
      
      /*---------------------------------------------------------------------
          potentials as array data (pointer for gpu)
      ----------------------------------------------------------------------*/
   protected:
      int nelements_coeff;                                                      // number of elements from pair_coeff
      double cutmax;                                                            // max cutoff for all elements
      std::string* elements_coeff;                                              // name of element in pair_coeff
      struct Element {
          int id_elem;                                                          // id of the element
          double mass;
          std::string elements;                                                 // name of elements
          Element() {                                                           // constructor 
              id_elem = 1;
              mass = 0.0;
              elements = "";
          }
      };
      struct Annparm {
          double*** weight_all, *** bias_all;                                   // for all layers
      };
      struct Param_ANNP {
          Element* all_elem;
          Annparm* all_annp;
          int nelements;                                                        // element parameters    
          double cut;
          double e_scale, e_shift, e_atom;                                      // for scale and shift the energy
          int flagsym, *flagact;                                                // seting the type of symmetry and activation function
          int ntl, nhl, nnod, nsf, npsf, ntsf;                                  // neural network parameters       
          double* sfnor_cov, * sfnor_avg;                                       // for symmetry function normalizing
      };

      Param_ANNP* params;
      virtual void read_file(char*);
      virtual void allocate();

      /*---------------------------------------------------------------------
          sub_functions
      ----------------------------------------------------------------------*/
      // used for compute () function
      virtual void annp_symmetry_pair(int, double, double, double, double*, double*, double*, double***, Param_ANNP*);
      virtual void annp_symmetry_trip(int, int, double, double, double, double, double, double, double, 
                                      double*, double*, double*, double*, double*, double***, Param_ANNP*);
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
      // inlined functions for efficiency
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
/*---------------------------------------------------------------------
    symmetry and activation function
    symmetry:
    flagsym = 0; Chebyshev function
    flagsym = 1; Behler and Parrinello (BP) function
    flagsym = 2; Custom

    activation function:
    flagact = 0; linear function            f(x) = x
    flagact = 1; hyperbolic tangent         y in [-1:1]
    flagact = 2; sigmoid                    y in [0:1]
    flagact = 3; modified thanh             y in [-1.7159:1.7159] f(+/=1) = +/=1
    flagact = 4; tanh & linear twisting term
----------------------------------------------------------------------*/
