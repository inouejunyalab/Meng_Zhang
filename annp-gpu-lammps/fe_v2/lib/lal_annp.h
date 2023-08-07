//* Host c++ *------------------------------------------
//      Artifical Neural Network Potential
//             Accelerated by GPU
//______________________________________________________        
//  begin:  Wed February 16, 2022
//  email:  meng_zhang@metall.t.u-tokyo.ac.jp
//          junya_inoue@metall.t.u-tokyo.ac.jp  
//______________________________________________________
//------------------------------------------------------

#ifndef LAL_ANNP_H
#define LAL_ANNP_H

#include "lal_base_annp.h"

namespace LAMMPS_AL {
	template <class numtyp, class acctyp> 
	class ANNP : public BaseAnnp<numtyp, acctyp> {
	public:
		ANNP();
		~ANNP();

        /// Clear any previous data and set up for a new LAMMPS run for generic systems
		/** \param max_nbors initial number of rows in the neighbor matrix
		  * \param cell_size cutoff + skin
		  * \param gpu_split fraction of particles handled by device
		  *
		  * Returns:
		  * -  0 if successful
		  * - -1 if fix gpu not found
		  * - -3 if there is an out of memory error
		  * - -4 if the GPU library was not compiled for GPU
		  * - -5 Double precision is not supported on card **/
		  // in the "lal_annp.cpp" file 
		int init(const int ntypes, const int nlocal, const int nall, 
				 const int max_nbors, const double cell_size, 
				 const double gpu_split, FILE* _screen, const int ntl, 
				 const int nhl, const int nnod, const int nsf, 
				 const int npsf, const int ntsf, const double e_scale, 
				  const double e_shift, const double e_atom, 
				 const int flagsym, int *flagact, double *sfnor_scal, 
				 double *sfnor_avg, double** host_cutsq, int* host_map, 
				 double*** host_weight_all, double*** host_bias_all);

		void clear();
												
		int bytes_per_atom(const int max_nbors) const;

		double host_memory_usage() const;													       

		int** compute(double* eatom, double& eng_vdwl, double** f, const int ago, 
					  const int inum_full, const int nall, const int nghost, 
				      double** host_x, int* host_type, double* sublo, double* subhi, 
					  tagint* tag, int** nspecial, tagint** special, 
					  const bool eflag, const bool vflag, const bool ea_flag, 
					  const bool va_flag, int& host_start, int** ilist, int** jnum, 
					  const double cpu_time, bool& success, double **vatom_annp);

		void compute(double* eatom, double& eng_vdwl, double** f, const int ago, 
					 const int inum_full, const int nall, const int nghost, 
					 double** host_x, int* host_type, int* ilist, int* numj,
					 int** firstneigh, const bool eflag, const bool vflag, 
					 const bool ea_flag, const bool va_flag, int& host_start, 
					 const double cpu_time, bool& success, double **vatom_annp);

		/*---------------------------------------------------------------------
			Device Kernels, Data, type Data
		----------------------------------------------------------------------*/
		UCL_Timer time_sh, time_ca, time_up;
		numtyp time_ca_all;
		numtyp time_up_all;
		numtyp time_sh_all;

		UCL_Texture weight_all_tex, bias_all_tex;
		UCL_Texture sfnor_scal_tex, sfnor_avg_tex;

		int _ntypes, _ntl, _nhl, _nnod, _nsf, _npsf, _ntsf, _flagsym;
		numtyp _cutMax;
		numtyp4 _out_mod;
		UCL_D_Vec<numtyp> _weight_all;
		UCL_D_Vec<numtyp> _bias_all;
		UCL_D_Vec<int> _flagact;																	// for saving active function
		UCL_D_Vec<numtyp> _sfnor_scal;																// for normalizing the symmetry function
		UCL_D_Vec<numtyp> _sfnor_avg;																// for normalizing the symmetry function
		int _max_size_w, _max_size_b;

		UCL_D_Vec<int> _acc_view;
		UCL_H_Vec<int> _host_acc;		
		bool shared_types;	
		
		int2 _gpup;
		int _max_force;
		UCL_Timer time_fp, time_ep;
		UCL_Vector<acctyp4, acctyp4> _force;
		
		UCL_Timer time_v2, time_v4;
		UCL_Vector<acctyp4, acctyp4> _virial4;
		UCL_Vector<acctyp2, acctyp2> _virial2;

		UCL_D_Vec<numtyp> _cutsq;																	// 1D for cuda, 
		UCL_D_Vec<int> _map;																		// method can be used from tersoff_gpu

		int _max_newj;
		UCL_D_Vec<acctyp4> _dGij;
		UCL_D_Vec<acctyp4> _Fj;			
		UCL_Vector<int, int> _newj;		
																																																
	protected:
		bool _allocated;
		int loop(const int eflag, const int vflag);													// is used for the "compute ()" in "lal_base_annp.cpp" file
		int loop_annp(const int eflag, const int vflag, const int nghost, const int nall);			// is used for the "compute ()" in "lal_annp.cpp" file to calculate the force and energy
	};
}
#endif


