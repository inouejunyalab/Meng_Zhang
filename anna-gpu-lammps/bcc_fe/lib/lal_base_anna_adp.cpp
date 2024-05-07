//* Host c++ *------------------------------------------
//      Physics-informed Neural Network Potential
//             Accelerated by GPU
//______________________________________________________
//  begin:  Monday August 07, 2023
//  email:  meng_zhang@metall.t.u-tokyo.ac.jp
//          junya_inoue@metall.t.u-tokyo.ac.jp   
//______________________________________________________
//------------------------------------------------------

#include "lal_base_anna_adp.h"
#include "iostream"

namespace LAMMPS_AL {
#define BasePinnadpT BasePinnadp<numtyp, acctyp>

extern Device<PRECISION, ACC_PRECISION> global_device;

template<class numtyp, class acctyp>
BasePinnadpT::BasePinnadp() : _compiled(false), _max_bytes(0), _onetype(0) {	
	device = &global_device;
	ans = new Answer<numtyp, acctyp>();
	nbor = new Neighbor();
	pair_program = nullptr;
	ucl_device = nullptr;
#if defined(LAL_OCL_EV_JIT)
	pair_program_noev = nullptr;
#endif
}

// free all array
template <class numtyp, class acctyp>
BasePinnadpT::~BasePinnadp() {
	delete ans;
	delete nbor;
	k_energy.clear();
	k_pair.clear();
	k_short_nbor.clear();
	if (pair_program) delete pair_program;
#if defined(LAL_OCL_EV_JIT)
	k_pair_noev.clear();
	if (pair_program_noev) delete pair_program_noev;
#endif
}

template <class numtyp, class acctyp>
int BasePinnadpT::bytes_per_atom_anna_adp(const int max_nbors) const {
	
	return device->atom.bytes_per_atom() + ans->bytes_per_atom() + 
		   nbor->bytes_per_atom(max_nbors);
}

// most of lines are copied from "lal_base_atomic.cpp" file
template <class numtyp, class acctyp>
int BasePinnadpT::init_anna_adp(const int nlocal, const int nall, const int max_nbors, 
								const int maxspecial, const double cell_size, 
								const double gpu_split, FILE* _screen, 
								const void* pair_program, const char* k_energy, 
								const char* k_anna_adp, const char* k_short_nbor, int onetype) {
	screen = _screen;

	int gpu_nbor = 0;
	if (device->gpu_mode() == Device<numtyp, acctyp>::GPU_NEIGH)
		gpu_nbor = 1;
	else if (device->gpu_mode() == Device<numtyp, acctyp>::GPU_HYB_NEIGH)
		gpu_nbor = 2;

	int _gpu_host = 0;
	int host_nlocal = hd_balancer.first_host_count(nlocal, gpu_split, gpu_nbor);
	if (host_nlocal > 0)
		_gpu_host = 1;

	_threads_per_atom = device->threads_per_atom();							
	//_threads_per_atom = 2;

	int success = device->init(*ans, false, false, nlocal, nall, maxspecial);
	if (success != 0)
		return success;

	if (ucl_device != device->gpu) _compiled = false;
	ucl_device = device->gpu;
	atom = &device->atom;

	_block_size = device->pair_block_size();
	compile_kernels(*ucl_device, pair_program, k_energy, 
					k_anna_adp, k_short_nbor, onetype);							

	if (_threads_per_atom > 1 && gpu_nbor == 0) {								
		nbor->packing(true);
		_nbor_data = &(nbor->dev_packed);
	}
	else
		_nbor_data = &(nbor->dev_nbor);

	success = device->init_nbor(nbor, nlocal, host_nlocal, nall, maxspecial, _gpu_host,
								max_nbors, cell_size, false, _threads_per_atom);
	if (success != 0)
		return success;

	hd_balancer.init(device, gpu_nbor, gpu_split);

	time_pair.init(*ucl_device);
	time_pair.zero();

	pos_tex.bind_float(atom->x, 4);

	_max_an_bytes = ans->gpu_bytes() + nbor->gpu_bytes();

	return 0;
}

template<class numtyp, class acctyp>
void BasePinnadpT::estimate_gpu_overhead(const int add_kernels) {
	device->estimate_gpu_overhead(1 + add_kernels, _gpu_overhead, _driver_overhead);
}

template <class numtyp, class acctyp>
void BasePinnadpT::clear_anna_adp() {
	// Output any timing information
	acc_timers();
	double avg_split = hd_balancer.all_avg_split();
	_gpu_overhead *= hd_balancer.timestep();
	_driver_overhead *= hd_balancer.timestep();
	device->output_times(time_pair, *ans, *nbor, avg_split, _max_bytes + _max_an_bytes,
						 _gpu_overhead, _driver_overhead, _threads_per_atom, screen);

	time_pair.clear();
	hd_balancer.clear();

	nbor->clear();
	ans->clear();
}

/*---------------------------------------------------------------------
	                    Copy neighbor list from host
----------------------------------------------------------------------*/
template <class numtyp, class acctyp>
int* BasePinnadpT::reset_nbors(const int nall, const int inum, int* ilist,
							  int* numj, int** firstneigh, bool& success) {
	success = true;

	int mn = nbor->max_nbor_loop(inum, numj, ilist);
	resize_atom(inum, nall, success);
	resize_local(inum, mn, success);
	if (!success)
		return nullptr;

	nbor->get_host(inum, ilist, numj, firstneigh, block_size());				

	double bytes = ans->gpu_bytes() + nbor->gpu_bytes();
	if (bytes > _max_an_bytes)
		_max_an_bytes = bytes;

	return ilist;
}

/*---------------------------------------------------------------------
	                Build neighbor list on device
----------------------------------------------------------------------*/
template <class numtyp, class acctyp>
inline void BasePinnadpT::build_nbor_list(const int inum, const int host_inum,
										 const int nall, double** host_x,
										 int* host_type, double* sublo,
										 double* subhi, tagint* tag,
										 int** nspecial, tagint** special,
										 bool& success) {
	success = true;
	resize_atom(inum, nall, success);
	resize_local(inum, host_inum, nbor->max_nbors(), success);
	if (!success)
		return;
	atom->cast_copy_x(host_x, host_type);

	int mn;
	nbor->build_nbor_list(host_x, inum, host_inum, nall, *atom, sublo, subhi,	
						  tag, nspecial, special, success, mn, ans->error_flag);

	double bytes = ans->gpu_bytes() + nbor->gpu_bytes();
	if (bytes > _max_an_bytes)
		_max_an_bytes = bytes;
}

/*---------------------------------------------------------------------
Copy nbor list from host if necessary and then calculate forces, virials
----------------------------------------------------------------------*/
template <class numtyp, class acctyp>
void BasePinnadpT::compute(const int f_ago, const int inum_full, 
						  const int nall, double** host_x, 
						  int* host_type, int* ilist, int* numj,
						  int** firstneigh, const bool eflag_in, 
						  const bool vflag_in, const bool eatom, 
						  const bool vatom, int& host_start,
						  const double cpu_time, bool& success) {

	acc_timers();
	int eflag, vflag;
	if (eatom) eflag = 2;														
	else if (eflag_in) eflag = 1;
	else eflag = 0;
	if (vatom) vflag = 2;														
	else if (vflag_in) vflag = 1;
	else vflag = 0;

#ifdef LAL_NO_BLOCK_REDUCE
	if (eflag) eflag = 2;
	if (vflag) vflag = 2;
#endif
	set_kernel(eflag, vflag);
	if (inum_full == 0) {
		host_start = 0;
		resize_atom(0, nall, success);
		zero_timers();
		return;
	}

	int ago = hd_balancer.ago_first(f_ago);
	int inum = hd_balancer.balance(ago, inum_full, cpu_time);
	ans->inum(inum);
	host_start = inum;

	if (ago == 0) {
		reset_nbors(nall, inum, ilist, numj, firstneigh, success);				
		if (!success)
			return;
	}

	atom->cast_x_data(host_x, host_type);
	hd_balancer.start_timer();
	atom->add_x_data(host_x, host_type);

	int evatom = 0;																
	if (eatom || vatom)	
		evatom = 1;																

	int _max_Fj_size = 0;
	const int red_blocks = loop(eflag, vflag);
	ans->copy_answers(eflag_in, vflag_in, eatom, vatom, ilist, red_blocks);
	device->add_ans_object(ans);
	hd_balancer.stop_timer();
}

/*---------------------------------------------------------------------
Reneighbor on GPU if necessary and then compute forces, virials, energies
----------------------------------------------------------------------*/
template <class numtyp, class acctyp>
int** BasePinnadpT::compute(const int ago, const int inum_full, const int nall,
						   double** host_x, int* host_type, double* sublo,
						   double* subhi, tagint* tag, int** nspecial,
						   tagint** special, const bool eflag_in, 
						   const bool vflag_in, const bool eatom, 
						   const bool vatom, int& host_start, int** ilist, 
						   int** jnum, const double cpu_time, bool& success) {
	acc_timers();
	int eflag, vflag;
	if (eatom) eflag = 2;														
	else if (eflag_in) eflag = 1;
	else eflag = 0;
	if (vatom) vflag = 2;														
	else if (vflag_in) vflag = 1;
	else vflag = 0;

#ifdef LAL_NO_BLOCK_REDUCE
	if (eflag) eflag = 2;
	if (vflag) vflag = 2;
#endif

	set_kernel(eflag, vflag);
	if (inum_full == 0) {
		host_start = 0;
		resize_atom(0, nall, success);
		zero_timers();
		return nullptr;
	}

	hd_balancer.balance(cpu_time);
	int inum = hd_balancer.get_gpu_count(ago, inum_full);
	ans->inum(inum);
	host_start = inum;

	if (ago == 0) {
		build_nbor_list(inum, inum_full - inum, nall, host_x, host_type,		
			sublo, subhi, tag, nspecial, special, success);
		if (!success)
			return nullptr;
		hd_balancer.start_timer();
	}
	else {
		atom->cast_x_data(host_x, host_type);
		hd_balancer.start_timer();
		atom->add_x_data(host_x, host_type);
	}

	*ilist = nbor->host_ilist.begin();
	*jnum = nbor->host_acc.begin();

	int evatom = 0;																
	if (eatom || vatom)
		evatom = 1;

	const int red_blocks = loop(eflag, vflag);
	ans->copy_answers(eflag_in, vflag_in, eatom, vatom, red_blocks);			
	device->add_ans_object(ans);
	hd_balancer.stop_timer();

	return nbor->host_jlist.begin() - host_start;
}

template <class numtyp, class acctyp>
double BasePinnadpT::host_memory_usage_anna_adp() const {
	return device->atom.host_memory_usage() + nbor->host_memory_usage() +
		   4 * sizeof(numtyp) + sizeof(BasePinnadp<numtyp, acctyp>);
}

/*---------------------------------------------------------------------
	                      compile the kernels
----------------------------------------------------------------------*/
template <class numtyp, class acctyp>
void BasePinnadpT::compile_kernels(UCL_Device& dev, const void* pair_str,		
								  const char* kenergy, const char* kname, 
							      const char* short_nbor, const int onetype) {
	if (_compiled && _onetype == onetype)
		return;
	_onetype = onetype;

	std::string s_anna = std::string(kname);
	if (pair_program) delete pair_program;
	pair_program = new UCL_Program(dev);
	std::string oclstring = device->compile_string() + " -DEVFLAG=1";
	if (_onetype) oclstring += " -DONETYPE=" + device->toa(_onetype);
	pair_program->load_string(pair_str, oclstring.c_str(), nullptr, screen);
	k_short_nbor.set_function(*pair_program, short_nbor);
	k_energy.set_function(*pair_program, kenergy);
	k_pair.set_function(*pair_program, kname);
	pos_tex.get_texture(*pair_program, "pos_tex");

#if defined(LAL_OCL_EV_JIT)
	oclstring = device->compile_string() + " -DEVFLAG=0";
	if (_onetype) oclstring += " -DONETYPE=" + device->toa(_onetype);
	if (pair_program_noev) delete pair_program_noev;
	pair_program_noev = new UCL_Program(dev);
	pair_program_noev->load_string(pair_str, oclstring.c_str(), nullptr, screen);
	k_pair_noev.set_function(*pair_program_noev, s_anna.c_str());
#else
	k_pair_sel = &k_pair;
#endif

	_compiled = true;

#if defined(USE_OPENCL) && (defined(CL_VERSION_2_1) || defined(CL_VERSION_3_0))
	if (dev.has_subgroup_support()) {
		size_t mx_subgroup_sz = k_pair.max_subgroup_size(_block_size);			
#if defined(LAL_OCL_EV_JIT)
		mx_subgroup_sz = std::min(mx_subgroup_sz, k_pair_noev.max_subgroup_size(_block_size));
#endif
		if (_threads_per_atom > mx_subgroup_sz)
			_threads_per_atom = mx_subgroup_sz;

		device->set_simd_size(mx_subgroup_sz);
	}
#endif
}

template class BasePinnadp<PRECISION, ACC_PRECISION>;
}
