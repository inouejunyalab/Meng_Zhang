//* Host c++ *------------------------------------------
//      Physics-informed Neural Network Potential
//             Accelerated by GPU
//______________________________________________________
//  begin:  Monday August 07, 2023
//  email:  meng_zhang@metall.t.u-tokyo.ac.jp
//          junya_inoue@metall.t.u-tokyo.ac.jp  
//______________________________________________________
//------------------------------------------------------

// create an global memory for saving the ghost atoms reuslts, it should be contained 
// the block id, thread id, global index of atom j, force[j], energy[j]
// each thread can be use the matrix by block id, thread id, to find the location
// it may waste some memory. so check the memory overflow is important

#if defined(USE_OPENCL)
#include "anna_adp_cl.h"
#elif defined(USE_CUDART)
const char* annaadp = 0;
#else
#include "anna_adp_cubin.h"
#endif

#include "lal_anna_adp.h"
#include "mpi.h"
#include <cassert>

namespace LAMMPS_AL {
#define ANNAADPMT ANNAADP<numtyp, acctyp>
	extern Device<PRECISION, ACC_PRECISION> device;

	template <class numtyp, class acctyp>
	ANNAADPMT::ANNAADP() : BasePinnadp<numtyp, acctyp>(), _allocated(false) {
	}
	template<class numtyp, class acctyp>
	ANNAADPMT::~ANNAADP() {
		clear();
	}

	template<class numtyp, class acctyp>
	int ANNAADPMT::bytes_per_atom(const int max_nbors) const {
		return this->bytes_per_atom_anna_adp(max_nbors);
	}

	template<class numtyp, class acctyp>
	int ANNAADPMT::init(const int ntypes, const int nlocal, const int nall, 
						const int max_nbors, const double cell_size, 
						const double gpu_split, FILE* _screen, 
						const int maxspecial, const int ntl, const int nhl, 
						const int nnod, const int nsf, const int npsf, 
						const int ntsf, const int nout, const int ngp, 
						const int n_mu, const int n_lamb, const double e_base, 
						const int flagsym, int *flagact, double** host_cutsq, 
						int* host_map, double*** host_weight_all,
						double*** host_bias_all, double* host_gadp_params) {

		int onetype = 0;
#ifdef USE_OPENCL
		for (int i = 1; i <= ntypes; i++) {
			for (int j = i; j <= ntypes; j++) {
				if (host_cutsq[i][j] > 0) {
					if (onetype > 0)	onetype = -1;
					else if (onetype == 0)	onetype = i * ntypes + j;			
				}
			}
		}
		if (onetype < 0)	onetype = 0;
#endif	
		int success;
		success = this->init_anna_adp(nlocal, nall, max_nbors, maxspecial, cell_size, 
									  gpu_split, _screen, anna_adp, "k_energy", 
									  "k_anna_adp", "k_anna_adp_short_nbor", onetype);
		if (success != 0)
			return success;
		
		_host_acc.alloc(nall, *(this->ucl_device), UCL_READ_WRITE);

		_max_padp = static_cast<int>(static_cast<double>(nall) * 1.10);		
		_adp_rho.alloc(_max_padp, *(this->ucl_device), UCL_READ_WRITE, UCL_READ_WRITE);
		adp_rho_tex.get_texture(*(this->pair_program), "rho_tex");
		adp_rho_tex.bind_float(_adp_rho, 1);

		_adp_mu.alloc(n_mu * _max_padp, *(this->ucl_device), UCL_READ_WRITE, UCL_READ_WRITE);
		adp_mu_tex.get_texture(*(this->pair_program), "imu_tex");
		adp_mu_tex.bind_float(_adp_rho, 1);

		_adp_lambda.alloc(n_lamb * _max_padp, *(this->ucl_device), UCL_READ_WRITE, UCL_READ_WRITE);
		adp_lambda_tex.get_texture(*(this->pair_program), "lambda_tex");
		adp_lambda_tex.bind_float(_adp_lambda, 1);

		_ladp_params.alloc(nout * _max_padp, *(this->ucl_device), UCL_READ_WRITE, UCL_READ_WRITE);
		ladp_params_tex.get_texture(*(this->pair_program), "ladp_tex");
		ladp_params_tex.bind_float(_ladp_params, 1);

		time_sh.init(*(this->ucl_device));
		time_eng.init(*(this->ucl_device));
		time_force.init(*(this->ucl_device));
		time_adp_comm1.init(*(this->ucl_device));
		time_adp_comm2.init(*(this->ucl_device));
		time_sh.zero();
		time_eng.zero();
		time_force.zero();
		time_adp_comm1.zero();
		time_adp_comm2.zero();

/*---------------------------------------------------------------------
				parameters used for "loop" function
----------------------------------------------------------------------*/
		_ntypes = ntypes;
		_ntl = ntl;
		_nhl = nhl;  
		_nnod = nnod;
		_nout = nout;
		_nmu = n_mu;
		_nlamb = n_lamb;
		_nsf = nsf;
		_npsf = npsf;
		_ntsf = ntsf;
		_flagsym = flagsym;
		_adp_const.x = e_base;

		//*----------------- for _cutsq, first way -------------------*/
		UCL_H_Vec<numtyp> dview_cutsq((ntypes + 1) * (ntypes + 1), *(this->ucl_device), UCL_WRITE_ONLY);
		dview_cutsq.zero();
		int index_cut = 0;
		_cutMax = 0.0;
		for (int i = 1; i <= ntypes; i++)
			for (int j = 1; j <= ntypes; j++) {
				index_cut = i * ntypes + j;
				dview_cutsq[index_cut] = host_cutsq[i][j];
				if(host_cutsq[i][j] > _cutMax)
					_cutMax = host_cutsq[i][j];
			}
		_adp_const.y = _cutMax;
		_cutsq.alloc((ntypes + 1) * (ntypes + 1), *(this->ucl_device), UCL_READ_ONLY);
		ucl_copy(_cutsq, dview_cutsq, false);
		
		//--------------------- for _map, second_way ------------------------
		UCL_H_Vec<int> dview_map(ntypes, *(this->ucl_device), UCL_WRITE_ONLY);
		dview_map.zero();
		for (int i = 0; i < ntypes; i++)
			dview_map[i] = host_map[i];
		_map.alloc(ntypes, *(this->ucl_device), UCL_READ_ONLY);
		ucl_copy(_map, dview_map, false);

		//----------------- for flag_activation function --------------------
		UCL_H_Vec<int> dview_flagact(ntl - 1, *(this->ucl_device), UCL_WRITE_ONLY);
		dview_flagact.zero();
		for (int i = 0; i < ntl - 1; i++)
			dview_flagact[i] = flagact[i];
		_flagact.alloc(ntl - 1, *(this->ucl_device), UCL_READ_ONLY);
		ucl_copy(_flagact, dview_flagact, false);

		/*------- for _weight_all, and _bias_all, second_way, we just have one 
				  potential file, thus no need "nparams" variable --------*/
		_max_size_w = 0;
		_max_size_b = 0;
		for (int i = 0; i < ntl - 1; i++) {
			int nrow_w = nnod, ncol_w = nnod, nrow_b = 1, ncol_b = nnod;
			if (i == 0) ncol_w = nsf;
			if (i == ntl - 2) {												
				nrow_w = nout;
				ncol_b = nout;
			}
			_max_size_w += nrow_w * ncol_w + 2;
			_max_size_b += nrow_b * ncol_b + 2;
		}
		UCL_H_Vec<numtyp> dview_weight(_max_size_w, *(this->ucl_device), UCL_WRITE_ONLY);
		UCL_H_Vec<numtyp> dview_bias(_max_size_b, *(this->ucl_device), UCL_WRITE_ONLY);
		dview_weight.zero();
		dview_bias.zero();
		int index_w = 0;
		int index_b = 0; 
		for (int i = 1; i <= ntypes; i++) {										
			int itype = host_map[i];
			if (i > 1 && itype == host_map[i - 1])	continue;
			for (int j = 0; j < ntl - 1; j++) {
				int nrow_w = nnod, ncol_w = nnod, nrow_b = 1, ncol_b = nnod;
				if (j == 0) ncol_w = nsf;
				if (j == ntl - 2) {												
					nrow_w = nout;
					ncol_b = nout;
				}
				for (int k = 0; k < nrow_w * ncol_w; k++) {
					dview_weight[index_w] = host_weight_all[itype][j][k];
					index_w++;
				}
				for (int k = 0; k < nrow_b * ncol_b; k++) {
					dview_bias[index_b] = host_bias_all[itype][j][k];
					index_b++;
				}
			}
		}
		_weight_all.alloc(_max_size_w, *(this->ucl_device), UCL_READ_ONLY);		
		ucl_copy(_weight_all, dview_weight, false);								
		_bias_all.alloc(_max_size_b, *(this->ucl_device), UCL_READ_ONLY);
		ucl_copy(_bias_all, dview_bias, false);

		weight_all_tex.get_texture(*(this->pair_program), "weight_tex");
		weight_all_tex.bind_float(_weight_all, 1);
		bias_all_tex.get_texture(*(this->pair_program), "bias_tex");
		bias_all_tex.bind_float(_bias_all, 1);

		UCL_H_Vec<numtyp> dview_adp_params(ngp, *(this->ucl_device), UCL_WRITE_ONLY);
		dview_adp_params.zero();
		for (int i = 0; i < ngp; i++) {
			dview_adp_params[i] = host_gadp_params[i];
		}
		_gadp_params.alloc(ngp, *(this->ucl_device), UCL_READ_ONLY);
		ucl_copy(_gadp_params, dview_adp_params, false);

		_allocated = true;
		this->_max_bytes = _weight_all.row_bytes() + _bias_all.row_bytes() + 
						   _map.row_bytes() + _cutsq.row_bytes() + 
						   _flagact.row_bytes() + _gadp_params.row_bytes() + 
						   _host_acc.row_bytes() + _adp_rho.device.row_bytes() + 
						   _adp_mu.device.row_bytes() + _adp_lambda.device.row_bytes() + _ladp_params.device.row_bytes();

		return 0;
	}

	// free all buffer
	template<class numtyp, class acctyp>
	void ANNAADPMT::clear() {

		if (!_allocated)
			return;
		time_sh.clear();
		time_eng.clear();
		time_force.clear();
		time_adp_comm1.clear();
		time_adp_comm2.clear();

		_allocated = false;
		_map.clear();
		_cutsq.clear();
		_adp_rho.clear();
		_adp_mu.clear();
		_adp_lambda.clear();
		_ladp_params.clear();
		_flagact.clear();
		_weight_all.clear();
		_bias_all.clear();
		_gadp_params.clear();
		_host_acc.clear();

		this->clear_anna_adp();
	}

	template <class numtyp, class acctyp>
	double ANNAADPMT::host_memory_usage() const {
		return this->host_memory_usage_anna_adp() + sizeof(ANNAADP<numtyp, acctyp>);
	}

	/*---------------------------------------------------------------------
	  copy nbor list from host if necessary and then compute atom energies
	----------------------------------------------------------------------*/
	template <class numtyp, class acctyp>
	void ANNAADPMT::compute(const int f_ago, const int inum_full, const int nall, const int nlocal, 
							double** host_x, int* host_type, int* ilist, int* numj, int** firstneigh, 
							const bool eflag_in, const bool vflag_in, const bool ea_flag, 
							const bool va_flag, void** adp_rho, void* adp_mu[], void* adp_lambda[], 
							void *ladp_params[], int& host_start, bool& success, const double cpu_time) {
		
		this->acc_timers();														
		int eflag, vflag;
		if (ea_flag)	eflag = 2;												
		else if (eflag_in) eflag = 1;
		else eflag = 0;
		if (va_flag)	vflag = 2;
		else if (vflag_in)	vflag = 1;
		else vflag = 0;

#ifdef LAL_NO_BLOCK_REDUCE
		if (eflag) eflag = 2;
		if (vflag) vflag = 2;
#endif
		this->set_kernel_anna_adp(eflag, vflag);

		if (this->device->time_device()) {
			this->time_pair.add_time_to_total(time_force.time());				
			this->atom->add_transfer_time(time_adp_comm1.time());				
			this->atom->add_transfer_time(time_adp_comm2.time());				
		}

		// ------------Resize _fp, _mu, and _lambda array for ANNAADP-----------
		if (nall > _max_padp) {
			_max_padp = nall; 												
			_adp_rho.resize(_max_padp);
			adp_rho_tex.bind_float(_adp_rho, 1);							

			_adp_mu.resize(_nmu * _max_padp);
			adp_mu_tex.bind_float(_adp_mu, 1);

			_adp_lambda.resize(_nlamb * _max_padp);
			adp_lambda_tex.bind_float(_adp_lambda, 1);

			_ladp_params.resize(_nout * _max_padp);
			ladp_params_tex.bind_float(_ladp_params, 1);
		}
		
		*adp_rho = _adp_rho.host.begin();
		for(int k = 0; k < _nmu; k ++) {
			adp_mu[k] = _adp_mu.host.begin() + k * nall;					
		}
		for(int k = 0; k < _nlamb; k ++) {
			adp_lambda[k] = _adp_lambda.host.begin() + k * nall;
		}
		for(int k = 0; k < _nout; k ++) {
			ladp_params[k] = _ladp_params.host.begin() + k * nall;
		}

		if (inum_full == 0) {
			host_start = 0;
			this->resize_atom(0, nall, success);							
			this->zero_timers();
			return;
		}

		int ago = this->hd_balancer.ago_first(f_ago);
		int inum = this->hd_balancer.balance(ago, inum_full, cpu_time);
		this->ans->inum(inum);
		host_start = inum;

		if (ago == 0) {
			this->reset_nbors(nall, inum, ilist, numj, firstneigh, success);	
			if (!success)
				return;
		}
		else {																
			_acc_view.view_offset(inum, this->nbor->dev_nbor, inum);
			ucl_copy(_acc_view, this->nbor->host_acc, inum, false);			
		}

		this->atom->cast_x_data(host_x, host_type);							
		this->atom->add_x_data(host_x, host_type);

		// -------------------------calculation------------------------------
		time_sh_all = 0.0;
		time_eng_all = 0.0;
		time_force_all = 0.0;
		loop_short_energy(nall, eflag, vflag);									

		_nlocal = nlocal;
		time_adp_comm1.start();
		_adp_rho.update_host(nlocal, false);
		_adp_mu.update_host(_nmu * nall, false);								
		_adp_lambda.update_host(_nlamb * nall, false);
		_ladp_params.update_host(_nout * nall, false);

		time_adp_comm1.stop();
		time_adp_comm1.sync_stop();
	}

	/*---------------------------------------------------------------------
		build nbor list from host if necessary and then compute atom energies
	----------------------------------------------------------------------*/
	template <class numtyp, class acctyp>
	int** ANNAADPMT::compute(const int ago, const int inum_full, const int nall, const int nlocal, double** host_x, 
							 int* host_type, double* sublo, double* subhi, tagint* tag, 
							 int** nspecial, tagint** special, const bool eflag_in, 
							 const bool vflag_in, const bool ea_flag, const bool va_flag, 
							 void** adp_rho, void* adp_mu[], void* adp_lambda[], void *ladp_params[], 
							 int& host_start, int** ilist, int** jnum, bool& success, const double cpu_time) {

		this->acc_timers();
		int eflag, vflag;
		if (ea_flag)	eflag = 2;												
		else if (eflag_in) eflag = 1;
		else eflag = 0;
		if (va_flag)	vflag = 2;
		else if (vflag_in)	vflag = 1;
		else vflag = 0;

#ifdef LAL_NO_BLOCK_REDUCE
		if (eflag) eflag = 2;
		if (vflag) vflag = 2;
#endif

		this->set_kernel_anna_adp(eflag, vflag);

		if (this->device->time_device()) {
			this->time_pair.add_time_to_total(time_force.time());				
			this->atom->add_transfer_time(time_adp_comm1.time());				
			this->atom->add_transfer_time(time_adp_comm2.time());				
		}

		// -----------------Resize FP and A array for ANNAADP-----------------
		if (nall > _max_padp) {
			_max_padp = nall; 												
			_adp_rho.resize(_max_padp);
			adp_rho_tex.bind_float(_adp_rho, 1);							

			_adp_mu.resize(_nmu * _max_padp);
			adp_mu_tex.bind_float(_adp_mu, 1);

			_adp_lambda.resize(_nlamb * _max_padp);
			adp_lambda_tex.bind_float(_adp_lambda, 1);

			_ladp_params.resize(_nout * _max_padp);
			ladp_params_tex.bind_float(_ladp_params, 1);
		}	
		*adp_rho = _adp_rho.host.begin();
		for(int k = 0; k < _nmu; k ++) {
			adp_mu[k] = _adp_mu.host.begin() + k * nall;						
		}
		for(int k = 0; k < _nlamb; k ++) {
			adp_lambda[k] = _adp_lambda.host.begin() + k * nall;
		}
		for(int k = 0; k < _nout; k ++) {
			ladp_params[k] = _ladp_params.host.begin() + k * nall;
		}

		if (inum_full == 0) {
			host_start = 0;	
			this->resize_atom(0, nall, success);								
			this->zero_timers();
			return nullptr;
		}

		this->hd_balancer.balance(cpu_time);
		int inum = this->hd_balancer.get_gpu_count(ago, inum_full);
		this->ans->inum(inum);
		host_start = inum;
	
		if (ago == 0) {
			this->build_nbor_list(inum, inum_full - inum, nall, host_x, host_type,
								  sublo, subhi, tag, nspecial, special, success);
			if (!success)
				return nullptr;
			_acc_view.view_offset(inum, this->nbor->dev_nbor, inum);

			if ((int)_host_acc.row_bytes() / 4 < inum) {						
				_host_acc.clear();
				_host_acc.alloc(inum + 2, *(this->ucl_device), UCL_READ_WRITE);
			}
			ucl_copy(_host_acc, _acc_view, inum, false);
		}
		else {
			_acc_view.view_offset(inum, this->nbor->dev_nbor, inum);
			ucl_copy(_acc_view, _host_acc, inum, false);
			this->atom->data_unavail();											
			this->atom->cast_x_data(host_x, host_type);
			this->atom->add_x_data(host_x, host_type);
		}
		*ilist = this->nbor->host_ilist.begin();
		*jnum = this->nbor->host_acc.begin();

		time_sh_all = 0.0;
		time_eng_all = 0.0;
		time_force_all = 0.0;
		loop_short_energy(nall, eflag, vflag);

		_nlocal = nlocal;														
		time_adp_comm1.start();
		_adp_rho.update_host(nlocal, false);
		_adp_mu.update_host(_nmu * nall, false);
		_adp_lambda.update_host(_nlamb * nall, false);
		_ladp_params.update_host(_nout * nall, false);

		time_adp_comm1.stop();
		time_adp_comm1.sync_stop();
		return this->nbor->host_jlist.begin() - host_start;
	}

	/*---------------------------------------------------------------------
					calculate energies, forces, and torques
	----------------------------------------------------------------------*/
	template<class numtyp, class acctyp>
	void ANNAADPMT::compute_force(const int nall, int *ilist, const bool eflag, const bool vflag, 
								  const bool ea_flag, const bool va_flag) {

		if (this->ans->inum() == 0)
			return;
		this->hd_balancer.start_timer();
		time_adp_comm2.start();
		this->add_fp_data();
		this->add_mu_data();
		this->add_lambda_data();
		this->add_ladp_params_data();
		time_adp_comm2.stop();

		loop_calcu_force(nall, eflag, vflag);								
		if (ilist == nullptr) {												
			this->ans->copy_answers(eflag, vflag, ea_flag, va_flag, this->ans->inum());
		}
		else {																
			this->ans->copy_answers(eflag, vflag, ea_flag, va_flag, ilist, this->ans->inum());
		}
		this->device->add_ans_object(this->ans);
		this->hd_balancer.stop_timer();
	}

	/*---------------------------------------------------------------------
				calculate energies, forces, and torques
	----------------------------------------------------------------------*/
	template <class numtyp, class acctyp>									
	int ANNAADPMT::loop(const int eflag, const int vflag) {

		this->time_pair.start();
		int ainum = this->ans->inum();
		const int BX = this->block_size();
		int GX = static_cast<int>(ceil(static_cast<double>(this->ans->inum()) /
										(BX / this->_threads_per_atom)));
		this->time_pair.stop();
		return GX;
	}
	
	/*---------------------------------------------------------------------
				short neighbor list and calculate energy 
	----------------------------------------------------------------------*/
	template <class numtyp, class acctyp>
	int ANNAADPMT::loop_short_energy(const int nall, const int eflag, const int vflag) {

		const int BX = this->block_size();
		int GX = static_cast<int>(ceil(static_cast<double>(this->ans->inum()) /
								  (BX / this->_threads_per_atom)));

		int ainum = this->ans->inum();										
		int nbor_pitch = this->nbor->nbor_pitch();

		this->time_pair.start();											
		time_sh.start();													
		this->k_short_nbor.set_size(GX, BX);
		this->k_short_nbor.run(&this->atom->x, &_cutMax, &_ntypes, 
							   &this->nbor->dev_nbor, &nbor_pitch,
							   &this->_nbor_data->begin(), &ainum, 
							   &this->_threads_per_atom);
		time_sh.stop();
		time_sh_all += time_sh.time();

		time_eng.start();
		this->k_energy.set_size(GX, BX);
		this->k_energy.run(&this->atom->x, &_map, &ainum, &this->_threads_per_atom, &this->nbor->dev_nbor, 
						   &this->_nbor_data->begin(), &nbor_pitch, &nall, &_ntl, &_nhl, &_nnod, &_npsf, 
						   &_ntsf, &_nsf,  &_nout, &_flagact, &eflag, &vflag, &_adp_const, &_weight_all, 
						   &_bias_all, &_gadp_params, &_adp_rho, &_adp_mu, &_adp_lambda, &_ladp_params, &this->ans->engv);

		time_eng.stop();
		time_eng_all += time_eng.time();									

		this->time_pair.stop();
		return ainum;														
	}

	/*---------------------------------------------------------------------
								calculate force
	----------------------------------------------------------------------*/
	template <class numtyp, class acctyp>
	void ANNAADPMT::loop_calcu_force(const int nall, const bool _eflag, const bool _vflag) {
		int eflag, vflag;														
		if (_eflag)
			eflag = 1;
		else
			eflag = 0;

		if (_vflag)
			vflag = 1;
		else
			vflag = 0;

		const int BX = this->block_size();
		int GX = static_cast<int>(ceil(static_cast<double>(this->ans->inum()) /
								  (BX / this->_threads_per_atom)));
		time_force.start();
		int ainum = this->ans->inum();
		int nbor_pitch = this->nbor->nbor_pitch();

		this->k_pair_sel->set_size(GX, BX);
		this->k_pair_sel->run(&this->atom->x, &_ntypes, &this->_threads_per_atom,
							  &this->nbor->dev_nbor, &this->_nbor_data->begin(),
							  &nbor_pitch, &this->ans->force, &this->ans->engv, 
							  &eflag, &vflag, &ainum, &_gadp_params, &_adp_const, 
							  &nall, &_adp_rho, &_adp_mu, &_adp_lambda, &_ladp_params);

		time_force.stop();
		time_force_all += time_force.time();
	}
	template class ANNAADP<PRECISION, ACC_PRECISION>;
}
