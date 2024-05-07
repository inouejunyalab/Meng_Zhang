//* Host c++ *------------------------------------------
//      Physics-informed Neural Network Potential
//             Accelerated by GPU
//______________________________________________________        
//  begin:  Sun July 9, 2023
//  email:  meng_zhang@metall.t.u-tokyo.ac.jp
//          junya_inoue@metall.t.u-tokyo.ac.jp  
//______________________________________________________
//------------------------------------------------------

#ifdef PAIR_CLASS
// clang-format off
PairStyle(anna_adp, PairANNA_ADP);
// clang-format on
#else

#ifndef LMP_PAIR_ANNA_ADP_H
#define LMP_PAIR_ANNA_ADP_H

#include "pair.h"

namespace LAMMPS_NS {

   class PairANNA_ADP : public Pair {
   public:
      PairANNA_ADP(class LAMMPS *);
      virtual ~PairANNA_ADP();
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
      struct ANNA {
          double*** weight_all, *** bias_all;                         
      };
      struct ANNAPARA {
          Element* all_elem;
          ANNA* all_anna;
          int nelements;                                                   
          double cut;
          double e_base, e_scal;                                           
          double* gparams;                                                   
          int flagsym, * flagact;                                            
          int ntl, nhl, nnod, nout, nsf, npsf, ntsf, ngp;                    
      };
      ANNAPARA* params;
     
/*---------------------------------------------------------------------
                                sub_functions
----------------------------------------------------------------------*/
      virtual void read_file(char*);
      virtual void allocate();

      // used for compute () function
      virtual void anna_adp_symmetry_pair(double, double, double*, ANNAPARA*);
      virtual void anna_adp_symmetry_trip(double, double, double, double*, ANNAPARA*);
      virtual void anna_adp_feed_forward(int, double *, double*, ANNAPARA*);

      // used for above function
      virtual void anna_adp_Tx(double, int, double*);
      virtual void dot_add_wxb(int, int, double**, double*, double**, double*);
      virtual void anna_adp_actf(int, int, int, int, double*, double*);

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

      // some math functions
      inline double dot3(const double* v1, const double* v2) {
          return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
      }
      inline void scale3(const double s, const double* v, double* ans) {
          ans[0] = s * v[0];
          ans[1] = s * v[1];
          ans[2] = s * v[2];
      }
   };

}    // namespace LAMMPS_NS

#endif
#endif

