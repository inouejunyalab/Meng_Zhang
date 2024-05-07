//* Host c++ *------------------------------------------
//      Physics-informed Neural Network Potential
//             Accelerated by GPU
//______________________________________________________
//  begin:  Monday August 07, 2023
//  email:  meng_zhang@metall.t.u-tokyo.ac.jp
//          junya_inoue@metall.t.u-tokyo.ac.jp  
//______________________________________________________
//------------------------------------------------------

#ifndef LAL_ANNA_ADP_H
#define LAL_ANNA_ADP_H

#include "lal_base_anna_adp.h"

namespace LAMMPS_AL {
	template <class numtyp, class acctyp>
	class ANNAADP : public BasePinnadp<numtyp, acctyp> {
	public:
		ANNAADP();
		~ANNAADP();

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
		  // in the "lal_anna_adp.cpp" file 
		int init(const int ntypes, const int nlocal, const int nall, 				
				 const int max_nbors, const double cell_size, 
				 const double gpu_split, FILE* _screen, 
				 const int maxspecial, const int ntl, const int nhl, 
				 const int nnod, const int nsf, const int npsf, 
				 const int ntsf, const int nout, const int ngp, 
				 const int n_mu, const int n_lamb, const double e_base, 
				 const int flagsym, int *flagact, double** host_cutsq, 
				 int* host_map, double*** host_weight_all,
				 double*** host_bias_all, double* host_gadp_params);

		// Clear all host and device data
		/** \note This is called at the beginning of the init() routine **/
		void clear();

		// Returns memory usage on device per atom									
		int bytes_per_atom(const int max_nbors) const;

		// Total host memory used by library for pair style
		double host_memory_usage() const;											

		int** compute(const int ago, const int inum_full, const int nall, 
					  const int nlocal, double** host_x, int* host_type, 
					  double* sublo, double* subhi, tagint* tag, 
					  int** nspecial, tagint** special, const bool eflag, 
					  const bool vflag, const bool ea_flag, const bool va_flag, 
					  void** adp_rho, void* adp_mu[], void* adp_lambda[], 
					  void *ladp_params[], int& host_start, int** ilist, 
					  int** jnum, bool& success, const double cpu_time);

		void compute(const int ago, const int inum_full, const int nall, 
					 const int nlocal, double** host_x, int* host_type, 
					 int* ilist, int* numj, int** firstneigh, 
					 const bool eflag, const bool vflag, 
					 const bool ea_flag, const bool va_flag, void** adp_rho, 
					 void* adp_mu[], void* adp_lambda[], void *ladp_params[],
					 int& host_start, bool& success, const double cpu_time);

		void compute_force(const int nall, int* ilist, const bool eflag, const bool vflag, const bool ea_flag, const bool va_flag);

		// used for copy data of _fp, _mu, and _lambda for ghost atoms from host to device 
		inline void add_fp_data() {										
			int nghost = this->atom->nall() - _nlocal;
			if (nghost > 0) {
				UCL_H_Vec<numtyp> host_view_fp;
				UCL_D_Vec<numtyp> dev_view_fp;
				host_view_fp.view_offset(_nlocal, _adp_rho.host);					
				dev_view_fp.view_offset(_nlocal, _adp_rho.device);
				ucl_copy(dev_view_fp, host_view_fp, nghost, true);
			}
		}
		inline void add_mu_data() {													
			int nall = this->atom->nall();
			int nghost = nall - _nlocal;
			if (nghost > 0) {
				UCL_H_Vec<numtyp> host_view_mu;
				UCL_D_Vec<numtyp> dev_view_mu;
				for(int k = 0; k < _nmu; k++){											
					host_view_mu.view_offset(_nlocal + k * nall, _adp_mu.host);
					dev_view_mu.view_offset(_nlocal + k * nall, _adp_mu.device);
					ucl_copy(dev_view_mu, host_view_mu, nghost, true);
				}
			}
		}
		inline void add_lambda_data() {												
			int nall = this->atom->nall();
			int nghost = nall - _nlocal;
			if (nghost > 0) {
				UCL_H_Vec<numtyp> host_view_lambda;
				UCL_D_Vec<numtyp> dev_view_lambda;									
				for(int k = 0; k < _nlamb; k++){									
					host_view_lambda.view_offset(_nlocal + k * nall, _adp_lambda.host);
					dev_view_lambda.view_offset(_nlocal + k * nall, _adp_lambda.device);
					ucl_copy(dev_view_lambda, host_view_lambda, nghost, true);
				}
			}
		}
		inline void add_ladp_params_data() {
			int nall = this->atom->nall();
			int nghost = nall - _nlocal;
			if (nghost > 0) {
				UCL_H_Vec<numtyp> host_view_ladp_params;
				UCL_D_Vec<numtyp> dev_view_ladp_params;								
				for(int k = 0; k < _nout; k++) {									
					host_view_ladp_params.view_offset(_nlocal + k * nall, _ladp_params.host);
					dev_view_ladp_params.view_offset(_nlocal + k * nall, _ladp_params.device);
					ucl_copy(dev_view_ladp_params, host_view_ladp_params, nghost, true);
				}
			}
		}

		/*-------------------------------------------------------------------
						Device Kernels, Data, type Data
		-------------------------------------------------------------------*/
		// if atom type constants fit in shared memory, use fast kernels
		bool shared_types;															
		
		UCL_Timer time_sh, time_eng, time_force, time_adp_comm1, time_adp_comm2;
		numtyp time_sh_all, time_eng_all, time_force_all;

		UCL_Texture weight_all_tex, bias_all_tex;
		UCL_Texture adp_rho_tex, adp_mu_tex, adp_lambda_tex, ladp_params_tex;

		int _nall, _ntypes, _ntl, _nhl, _nnod, _nout, _nmu, _nlamb; 
		int _nsf, _npsf, _ntsf, _ngp, _flagsym;
		int _max_size_w, _max_size_b;
		numtyp _cutMax;
		numtyp2 _adp_const;
		UCL_D_Vec<numtyp> _weight_all;
		UCL_D_Vec<numtyp> _bias_all;
		UCL_D_Vec<int> _flagact;													
		UCL_D_Vec<numtyp> _gadp_params;

		UCL_D_Vec<int> _acc_view;
		UCL_H_Vec<int> _host_acc;
		
		UCL_D_Vec<numtyp> _cutsq;													
		UCL_D_Vec<int> _map;														

		int _max_padp;
		UCL_Vector<numtyp, numtyp> _adp_rho;
		UCL_Vector<numtyp, numtyp> _adp_mu;
		UCL_Vector<numtyp, numtyp> _adp_lambda;
		UCL_Vector<numtyp, numtyp> _ladp_params;

	protected:
		bool _allocated;
		int _nlocal;
		int loop(const int eflag, const int vflag);									
		int loop_short_energy(const int nall, const int eflag, const int vflag);	
		void loop_calcu_force(const int nall, const bool eflag, const bool vflag);
	};
}
#endif
