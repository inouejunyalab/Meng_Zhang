//* Host c++ *------------------------------------------
//      Artifical Neural Network Potential
//             Accelerated by GPU
//______________________________________________________        
//  begin:  Monday August 07, 2023
//  email:  meng_zhang@metall.t.u-tokyo.ac.jp
//          junya_inoue@metall.t.u-tokyo.ac.jp  
//______________________________________________________
//------------------------------------------------------

#ifndef LAL_BASE_ANNA_ADP_H
#define LAL_BASE_ANNA_ADP_H

#include "lal_device.h"
#include "lal_balance.h"
#include "mpi.h"

#if defined(USE_OPENCL)
#include "geryon/ocl_texture.h"
#elif defined(USE_CUDART)
#include "geryon/nvc_texture.h"
#elif defined(USE_HIP)
#include "geryon/hip_texture.h"
#else
#include "geryon/nvd_texture.h"
#endif

namespace LAMMPS_AL {

template <class numtyp, class acctyp> 
class BasePinnadp {
public:
	BasePinnadp();
	virtual ~BasePinnadp();
	// Clear any previous data and set up for a new LAMMPS run
	/** \param max_nbors initial number of rows in the neighbor matrix
	  * \param cell_size cutoff + skin
	  * \param gpu_split fraction of particles handled by device
	  * \param k_name name for the kernel for force calculation
	  * \param k_short_nbor for the short neighbor list build
	  * Returns:
	  * -  0 if successful
	  * - -1 if fix gpu not found
	  * - -3 if there is an out of memory error
	  * - -4 if the GPU library was not compiled for GPU
	  * - -5 Double precision is not supported on card **/
	int init_anna_adp(const int nlocal, const int nall, const int max_nbors,
					  const int maxspecial, const double cell_size,
					  const double gpu_split, FILE* _screen, const void* pair_program,
					  const char* k_energy, const char* k_anna_adp=nullptr, 
					  const char* k_short_nbor=nullptr, const int onetype = 0);

	// Estimate the overhead for GPU context changes and CPU driver								
	void estimate_gpu_overhead(const int add_kernels = 0);

	// Check if there is enough storage for atom arrays and realloc if not
	/** \param success set to false if insufficient memory **/
	inline void resize_atom(const int inum, const int nall, bool& success) {
		if (atom->resize(nall, success))
			pos_tex.bind_float(atom->x, 4);
		ans->resize(inum, success);
	}

	// Check if there is enough storage for neighbors and realloc if not
	/** \param nlocal number of particles whose nbors must be stored on device
	  * \param host_inum number of particles whose nbors need to copied to host
	  * \param current maximum number of neighbors
	  * \note olist_size=total number of local particles **/
	inline void resize_local(const int inum, const int max_nbors, bool& success) {
		nbor->resize(inum, max_nbors, success);
	}

	// Check if there is enough storage for neighbors and realloc if not
	/** \param nlocal number of particles whose nbors must be stored on device
	  * \param host_inum number of particles whose nbors need to copied to host
	  * \param current maximum number of neighbors
	  * \note host_inum is 0 if the host is performing neighboring
	  * \note nlocal+host_inum=total number local particles
	  * \note olist_size=0 **/
	inline void resize_local(const int inum, const int host_inum,
							 const int max_nbors, bool& success) {
		nbor->resize(inum, host_inum, max_nbors, success);
	}

	// Clear all host and device data
	/** \note This is called at the beginning of the init() routine **/
	void clear_anna_adp();																		

	// Returns memory usage on device per atom
	int bytes_per_atom_anna_adp(const int max_nbors) const;										

	// Total host memory used by library for pair style
	double host_memory_usage_anna_adp() const;													

	// Accumulate timers
	inline void acc_timers() {
		if (device->time_device()) {
			nbor->acc_timers(screen);
			time_pair.add_to_total();
			atom->acc_timers();
			ans->acc_timers();
		}
	}

	// Zero timers
	inline void zero_timers() {
		time_pair.zero();
		atom->zero_timers();
		ans->zero_timers();
	}

	// copy neighbor list from host
	int* reset_nbors(const int nall, const int inum, int* ilist,
					 int* numj, int** firstneigh, bool& success);

	// build neighbor list on device
	void build_nbor_list(const int inum, const int host_inum, const int nall,
						 double** host_x, int* host_type, double* sublo,
						 double* subhi, tagint* tag, int** nspecial,
						 tagint** special, bool& success);

	// pair loop with host neighboring 
	void compute(const int ago, const int inum_full, const int nall,							
				 double** host_x, int* host_type, int* ilist, int* numj,
				 int** firstneigh, const bool eflag, const bool vflag,
				 const bool eatom, const bool vatom, int& host_start,
				 const double cpu_time, bool& success);

	// pair loop with device neighboring
	int** compute(const int ago, const int inum_full, const int nall,							
				  double** host_x, int* host_type, double* sublo,
				  double* subhi, tagint* tag, int** nspecial,
				  tagint** special, const bool eflag, 
				  const bool vflag, const bool eatom, 
				  const bool vatom, int& host_start, int** ilist, 
				  int** jnum, const double cpu_time, bool& success);

	/*---------------------------------------------------------------------
		                        Device Data
	----------------------------------------------------------------------*/
	// Device properties and atom and neighbor storage
	Device<numtyp, acctyp>* device;

	// Geryon device
	UCL_Device* ucl_device;

	// Device timers
	UCL_Timer time_pair;

	// host device load balancer
	Balance<numtyp, acctyp> hd_balancer;														

	// lammps pointer for screen output
	FILE* screen;

	// atom data
	Atom<numtyp, acctyp>* atom;

	// force and energy data
	Answer<numtyp, acctyp>* ans;

	// neighbor data 
	Neighbor* nbor;
	UCL_Kernel k_short_nbor, k_energy;

	/*---------------------------------------------------------------------
		                        Device kernels
	----------------------------------------------------------------------*/
	UCL_Program* pair_program,* pair_program_noev;
	UCL_Kernel k_pair, k_pair_noev;
	UCL_Kernel * k_pair_sel;
	inline int block_size() { return _block_size; }
	inline void set_kernel(const int eflag, const int vflag) {										
#if defined (LAL_OCL_EV_JIT)																		
		if (eflag || vflag)		
			k_pair_sel = &k_pair;																	
		else	
			k_pair_sel = &k_pair_noev;													
#endif
	}
	inline void set_kernel_anna_adp(const int eflag, const int vflag) {								
#if defined (LAL_OCL_EV_JIT)																		
		if (eflag || vflag)	
			k_pair_sel = &k_pair;																	
		else
			k_pair_sel = &k_pair_noev;
#endif
	}

	/*---------------------------------------------------------------------
		                          Textures
	----------------------------------------------------------------------*/
	UCL_Texture pos_tex;																			
protected:
	bool _compiled;
	int _block_size, _threads_per_atom, _onetype;
	double _max_bytes, _max_an_bytes;
	double _gpu_overhead, _driver_overhead;
	UCL_D_Vec<int>* _nbor_data;

	void compile_kernels(UCL_Device &dev, const void* pair_str, const char* kname, 
						 const char* updat, const char* short_nbor, const int onetype);

	virtual int loop(const int eflag, const int vflag) = 0;
	};
}
#endif
