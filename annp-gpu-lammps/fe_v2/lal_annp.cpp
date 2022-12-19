//* Host c++ *------------------------------------------
//      Artifical Neural Network Potential
//             Accelerated by GPU
//______________________________________________________        
//  begin:  Wed February 16, 2022
//  email:  
//______________________________________________________
//------------------------------------------------------

// create an global memory for saving the ghost atoms reuslts, it should be contained the block id, thread id, global index of atom j, force[j], energy[j]
// each thread can be use the matrix by block id, thread id, to find the location
// it may waste some memory. so check the memory overflow is important

#if defined(USE_OPENCL)
#include "annp_cl.h"
#elif defined(USE_CUDART)
const char* annp = 0;
#else
#include "annp_cubin.h"
#endif

#include "lal_annp.h"
#include "mpi.h"
#include <cassert>

namespace LAMMPS_AL {
#define ANNPMT ANNP<numtyp, acctyp>
	extern Device<PRECISION, ACC_PRECISION> device;

	template <class numtyp, class acctyp>
	ANNPMT::ANNP() : BaseAnnp<numtyp, acctyp>(), _allocated(false) {
	}
	template<class numtyp, class acctyp>
	ANNPMT::~ANNP() {
		clear();
	}

	template<class numtyp, class acctyp>
	int ANNPMT::bytes_per_atom(const int max_nbors) const {
		return this->bytes_per_atom_annp(max_nbors);
	}

	template<class numtyp, class acctyp>
	int ANNPMT::init(const int ntypes, const int nlocal, const int nall, const int max_nbors, 
					 const double cell_size, const double gpu_split, FILE* _screen, 
					 const int ntl, const int nhl, const int nnod, const int nsf, const int npsf,
					 const int ntsf, const double e_scale, const double e_shift, 
					 const double e_atom, const int flagsym, int* flagact, 
					 double* sfnor_scal, double* sfnor_avg, double** host_cutsq, 
					 int* host_map, double*** host_weight_all, double*** host_bias_all) {
		int onetype = 0;
#ifdef USE_OPENCL
		for (int i = 1; i <= ntypes; i++) {
			for (int j = i; j <= ntypes; j++) {
				if (host_cutsq[i][j] > 0) {
					if (onetype > 0)	onetype = -1;
					else if (onetype == 0)	onetype = i * ntypes + j;								// _cutsq is one dimension, so we must change the index of _cutsq matrix, style from "eam" and "tersoff" potentials
				}
			}
		}
		if (onetype < 0)	onetype = 0;
#endif	
		int success;
		success = this->init_annp(nlocal, nall, max_nbors, 0, cell_size, gpu_split, _screen, 
							      annp, "k_annp", "k_annp_updat", "k_annp_short_nbor", onetype);
		if (success != 0)
			return success;
		
		// initialization allocate the matrix of _Fj
		_Fj.alloc(nall, *(this->ucl_device), UCL_READ_WRITE);
		_dGij.alloc(nall, *(this->ucl_device), UCL_READ_WRITE);
		_max_newj = static_cast<int>(static_cast<double>(nlocal) * 1.10);
		_newj.alloc(_max_newj, *(this->ucl_device), UCL_READ_WRITE, UCL_READ_WRITE);
		_host_acc.alloc(nall, *(this->ucl_device), UCL_READ_WRITE);

		// initialization allocate the matrix of virial
		_virial2.alloc(_max_newj, *(this->ucl_device), UCL_READ_WRITE, UCL_READ_WRITE);
		_virial4.alloc(_max_newj, *(this->ucl_device), UCL_READ_WRITE, UCL_READ_WRITE);

		//_force.alloc(nall, *(this->ucl_device), UCL_READ_WRITE, UCL_READ_WRITE);
		_max_force = static_cast<int>(static_cast<double>(nall) * 1.10);
		_force.alloc(_max_force, *(this->ucl_device), UCL_READ_WRITE, UCL_READ_WRITE);

		time_fp.init(*(this->ucl_device));
		time_fp.zero();
		time_ep.init(*(this->ucl_device));
		time_ep.zero();
		time_v2.init(*(this->ucl_device));
		time_v2.zero();
		time_v4.init(*(this->ucl_device));
		time_v4.zero();

		time_sh.init(*(this->ucl_device));
		time_ca.init(*(this->ucl_device));
		time_up.init(*(this->ucl_device));
		time_sh.zero();
		time_ca.zero();
		time_up.zero();

/*---------------------------------------------------------------------
		parameters used for "loop" function
----------------------------------------------------------------------*/
		_ntypes = ntypes;
		_ntl = ntl;
		_nhl = nhl;  
		_nnod = nnod;
		_nsf = nsf;
		_npsf = npsf;
		_ntsf = ntsf;
		_flagsym = flagsym;
		_out_mod.x = e_scale;
		_out_mod.y = e_shift;
		_out_mod.z = e_atom;

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
		_cutsq.alloc((ntypes + 1) * (ntypes + 1), *(this->ucl_device), UCL_READ_ONLY);
		ucl_copy(_cutsq, dview_cutsq, false);
		//this->atom->type_pack1(ntypes, ntypes, _cutsq, host_write, host_cutsq);					// pack1 function is in "lal_atom.h" file
		
		//*----------------- for _map, second_way -------------------*/
		UCL_H_Vec<int> dview_map(ntypes, *(this->ucl_device), UCL_WRITE_ONLY);
		dview_map.zero();
		for (int i = 0; i < ntypes; i++)
			dview_map[i] = host_map[i];
		_map.alloc(ntypes, *(this->ucl_device), UCL_READ_ONLY);
		ucl_copy(_map, dview_map, false);

		//*----------------- for flag_activation function -------------------*/
		UCL_H_Vec<int> dview_flagact(ntl - 1, *(this->ucl_device), UCL_WRITE_ONLY);
		dview_flagact.zero();
		for (int i = 0; i < ntl - 1; i++)
			dview_flagact[i] = flagact[i];
		_flagact.alloc(ntl - 1, *(this->ucl_device), UCL_READ_ONLY);
		ucl_copy(_flagact, dview_flagact, false);

		//*----------------- for sfnor_cov and sfnor_avg -------------------*/
		UCL_H_Vec<numtyp> dview_sfn(nsf, *(this->ucl_device), UCL_WRITE_ONLY);
		_sfnor_scal.alloc(nsf, *(this->ucl_device), UCL_READ_ONLY);
		dview_sfn.zero();
		for (int i = 0; i < nsf; i++)
			dview_sfn[i] = sfnor_scal[i];
		ucl_copy(_sfnor_scal, dview_sfn, false);
		
		_sfnor_avg.alloc(nsf, *(this->ucl_device), UCL_READ_ONLY);
		dview_sfn.zero();
		for (int i = 0; i < nsf; i++)
			dview_sfn[i] = sfnor_avg[i];
		ucl_copy(_sfnor_avg, dview_sfn, false);

		//.........used for the texture memory
		sfnor_scal_tex.get_texture(*(this->pair_program), "sfsc_tex");
		sfnor_scal_tex.bind_float(_sfnor_scal, 1);
		sfnor_avg_tex.get_texture(*(this->pair_program), "sfav_tex");
		sfnor_avg_tex.bind_float(_sfnor_avg, 1);

		//*----------------- for _weight_all, and _bias_all, second_way, we just have one potential file, thus no need "nparams" variable -------------------*/
		_max_size_w = 0;
		_max_size_b = 0;
		for (int i = 0; i < ntl - 1; i++) {
			int nrow_w = nnod, ncol_w = nnod, nrow_b = 1, ncol_b = nnod;
			if (i == 0) ncol_w = nsf;
			if (i == ntl - 2) {																		// for the last layer
				nrow_w = 1;
				ncol_b = 1;
			}
			_max_size_w += nrow_w * ncol_w + 2;
			_max_size_b += nrow_b * ncol_b + 2;
		}
		UCL_H_Vec<numtyp> dview_weight(_max_size_w, *(this->ucl_device), UCL_WRITE_ONLY);			// 3D, x: 10*28(nnod*nsf), y: nlayer-1, z: nelements
		UCL_H_Vec<numtyp> dview_bias(_max_size_b, *(this->ucl_device), UCL_WRITE_ONLY);				// 3D, x: nnod, y: nlayer-1, z: nelements
		dview_weight.zero();
		dview_bias.zero();
		int index_w = 0;
		int index_b = 0; 
		for (int i = 1; i <= ntypes; i++) {															// copy value from host memeory
			int itype = host_map[i];
			if (i > 1 && itype == host_map[i - 1])	continue;
			for (int j = 0; j < ntl - 1; j++) {
				int nrow_w = nnod, ncol_w = nnod, nrow_b = 1, ncol_b = nnod;
				if (j == 0) ncol_w = nsf;
				if (j == ntl - 2) {																	// for the last layer
					nrow_w = 1;
					ncol_b = 1;
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
		_weight_all.alloc(_max_size_w, *(this->ucl_device), UCL_READ_ONLY);							// allocate memeory on GPU, and copy data
		ucl_copy(_weight_all, dview_weight, false);													// copy data from buffer to GPU
		_bias_all.alloc(_max_size_b, *(this->ucl_device), UCL_READ_ONLY);
		ucl_copy(_bias_all, dview_bias, false);

		//.........used for the texture memory
		weight_all_tex.get_texture(*(this->pair_program), "weight_tex");
		weight_all_tex.bind_float(_weight_all, 1);
		bias_all_tex.get_texture(*(this->pair_program), "bias_tex");
		bias_all_tex.bind_float(_bias_all, 1);

		_allocated = true;
		this->_max_bytes = _weight_all.row_bytes() + _bias_all.row_bytes() + _map.row_bytes() +
						   _cutsq.row_bytes() + _flagact.row_bytes() + _Fj.row_bytes() +
						   _dGij.row_bytes() + _sfnor_scal.row_bytes() + _sfnor_avg.row_bytes() +
						   _force.row_bytes() + _newj.row_bytes() + _host_acc.row_bytes() + 
						   _virial2.row_bytes() + _virial4.row_bytes();

		return 0;
	}

	// free all buffer
	template<class numtyp, class acctyp>
	void ANNPMT::clear()
	{
		if (!_allocated)
			return;
		time_fp.clear();
		time_ep.clear();
		time_sh.clear();
		time_ca.clear();
		time_up.clear();
		time_v2.clear();
		time_v4.clear();

		_allocated = false;
		_map.clear();
		_cutsq.clear();
		_flagact.clear();
		_sfnor_scal.clear();
		_sfnor_avg.clear();
		_weight_all.clear();
		_bias_all.clear();
		_newj.clear();
		_host_acc.clear();
		_Fj.clear();
		_force.clear();
		_dGij.clear();

		_virial2.clear();
		_virial4.clear();
		this->clear_annp();
	}

	template <class numtyp, class acctyp>
	double ANNPMT::host_memory_usage() const {
		return this->host_memory_usage_annp() + sizeof(ANNP<numtyp, acctyp>);
	}

	/*---------------------------------------------------------------------
		copy nbor list from host if necessary and then compute atom energies/forces
	----------------------------------------------------------------------*/
	template <class numtyp, class acctyp>
	void ANNPMT::compute(double* eatom, double& eng_vdwl, double** f, const int f_ago, 
						 const int inum_full, const int nall, const int nghost, 
					     double** host_x, int* host_type, int* ilist, int* numj, 
						 int** firstneigh, const bool eflag_in, const bool vflag_in, 
						 const bool ea_flag, const bool va_flag, int& host_start, 
						 const double cpu_time, bool& success, double **vatom_annp) {
		 
		this->acc_timers();																			// va_flag is for atomic virial, ea_flag is for atomic energy
		int eflag, vflag;
		if (ea_flag)	eflag = 2;																	// energy of atom i
		else if (eflag_in) eflag = 1;
		else eflag = 0;
		if (va_flag)	vflag = 2;
		else if (vflag_in)	vflag = 1;
		else vflag = 0;

#ifdef LAL_NO_BLOCK_REDUCE
		if (eflag) eflag = 2;
		if (vflag) vflag = 2;
#endif
		this->set_kernel_annp(eflag);
		if (this->device->time_device()) {		
			this->atom->add_transfer_time(time_fp.time());											// Add transfer time from device -> host after part 1
			this->atom->add_transfer_time(time_ep.time());
			this->atom->add_transfer_time(time_v2.time());
			this->atom->add_transfer_time(time_v4.time());
		}

		if (inum_full == 0) {
			host_start = 0;
			this->resize_atom(0, nall, success);													// Make sure textures are correct if realloc by a different hybrid style
			this->zero_timers();
			return;
		}

		int ago = this->hd_balancer.ago_first(f_ago);
		int inum = this->hd_balancer.balance(ago, inum_full, cpu_time);
		this->ans->inum(inum);
		host_start = inum;

		//printf("^^^^^^^^^^^^^^^^^^ago^^^^^^^^^^^^^^ %d\n", ago);
		if (ago == 0) {
			this->reset_nbors(nall, inum, ilist, numj, firstneigh, success);						// copy neighbor list from host
			if (!success)
				return;
		}
		else {																						// update the numj into "dev_nbor" matrix
			_acc_view.view_offset(inum, this->nbor->dev_nbor, inum);
			ucl_copy(_acc_view, this->nbor->host_acc, inum, false);									// dev_nbor = dev_packed, or dev_nbor != dev_packed, we just copy this
		}

		//this->atom->data_unavail();																// this one is so important, for updatint the coordinate for next step
		this->atom->cast_x_data(host_x, host_type);													// .simulation, this funciton in "lal_atom.h" file, are used in "lal_device.h" file, "fix_gpu" function
		this->hd_balancer.start_timer();															// .this one must set before "cast_x_data" and "add_x_data" functions 
		this->atom->add_x_data(host_x, host_type);

		time_sh_all = 0.0;																			// short_nbor time
		time_ca_all = 0.0;																			// calculation time		
		time_up_all = 0.0;																			// updating the force time
		const int red_blocks = loop_annp(eflag, vflag, nghost, nall);

		// updating the energy for each atom
		time_ep.start();
		this->ans->engv.update_host(red_blocks, true);
		time_ep.stop();
		time_ep.sync_stop();
		if (eflag_in) {
			for (int ii = 0; ii < red_blocks; ii++) {
				eng_vdwl += this->ans->engv[ii];
			}
			if(ea_flag)																				// normally, no need to update the energy for atoms each steps
			for (int ii = 0; ii < red_blocks; ii++) {
				eatom[ii] += this->ans->engv[ii];
			}
		}
	
		// updating the force for all atoms
		time_fp.start();
		_force.update_host(nall, true);
		time_fp.stop();
		time_fp.sync_stop();
		int nlocal = nall - nghost;
		for (int ii = 0; ii < nall; ii++) {
			int idj = _force[ii].w;
			if (nghost > 0 && (ii >= nlocal && idj == 0))	continue;
			if (idj < red_blocks)
				idj = ilist[idj];
			f[idj][0] = _force[ii].x;
			f[idj][1] = _force[ii].y;
			f[idj][2] = _force[ii].z;
		}

		// updating the virial for each atom
		if(va_flag) {
			time_v2.start();
			_virial2.update_host(nall, true);
			time_v2.stop();
			time_v2.sync_stop();
			time_v4.start();
			_virial4.update_host(nall, true);
			time_v4.stop();
			time_v4.sync_stop();
			for(int ii = 0; ii < nall; ii++) {
				vatom_annp[ii][0] = _virial4[ii].x;
				vatom_annp[ii][1] = _virial4[ii].y;
				vatom_annp[ii][2] = _virial4[ii].z;
				vatom_annp[ii][3] = _virial4[ii].w;
				vatom_annp[ii][4] = _virial2[ii].x;
				vatom_annp[ii][5] = _virial2[ii].y;
			}			
		}

		_force.clear();
		_virial2.clear();
		_virial4.clear();
		this->hd_balancer.stop_timer();
	}

	/*---------------------------------------------------------------------
		build nbor list from host if necessary and then compute atom energies/forces
	----------------------------------------------------------------------*/
	template <class numtyp, class acctyp>
	int** ANNPMT::compute(double* eatom, double& eng_vdwl, double** f, const int ago, 
						  const int inum_full, const int nall, const int nghost, 
						  double** host_x, int* host_type, double* sublo, 
						  double* subhi, tagint* tag, int** nspecial, tagint** special, 
						  const bool eflag_in, const bool vflag_in, const bool ea_flag, 
						  const bool va_flag, int& host_start, int** ilist, int** jnum, 
						  const double cpu_time, bool& success, double **vatom_annp) {
		this->acc_timers();
		int eflag, vflag;
		if (ea_flag)	eflag = 2;																	// flag for used to calculate the energy of atoms
		else if (eflag_in) eflag = 1;
		else eflag = 0;
		if (va_flag)	vflag = 2;
		else if (vflag_in)	vflag = 1;
		else vflag = 0;

#ifdef LAL_NO_BLOCK_REDUCE
		if (eflag) eflag = 2;
		if (vflag) vflag = 2;
#endif

		this->set_kernel_annp(eflag);
		if (this->device->time_device()) {
			this->atom->add_transfer_time(time_fp.time());											// Add transfer time from device -> host after part 1
			this->atom->add_transfer_time(time_ep.time());
			this->atom->add_transfer_time(time_v2.time());
			this->atom->add_transfer_time(time_v4.time());
		}
		if (inum_full == 0) {
			host_start = 0;	
			this->resize_atom(0, nall, success);													// Make sure textures are correct if realloc by a different hybrid style
			this->zero_timers();
			return nullptr;
		}

		this->hd_balancer.balance(cpu_time);
		int inum = this->hd_balancer.get_gpu_count(ago, inum_full);
		this->ans->inum(inum);
		host_start = inum;

		//printf("^^^^^^^^^^^^^^^^^^ago^^^^^^^^^^^^^^ %d\n", ago);		
		// Build neighbor list on GPU if necessary
		if (ago == 0) {
			this->build_nbor_list(inum, inum_full - inum, nall, host_x, host_type,					// build neighbor list on GPU
								  sublo, subhi, tag, nspecial, special, success);
			if (!success)
				return nullptr;
			this->hd_balancer.start_timer();
			_acc_view.view_offset(inum, this->nbor->dev_nbor, inum);

			if ((int)_host_acc.row_bytes() / 4 < inum) {											// in the case of neighbor list changed, but the host_acc don't have enough size
				_host_acc.clear();
				_host_acc.alloc(inum + 2, *(this->ucl_device), UCL_READ_WRITE);
			}
			ucl_copy(_host_acc, _acc_view, inum, false);
		}
		else {
			_acc_view.view_offset(inum, this->nbor->dev_nbor, inum);
			ucl_copy(_acc_view, _host_acc, inum, false);
			this->atom->data_unavail();																// this one is so important, for updatint the coordinate for next step
			this->atom->cast_x_data(host_x, host_type);
			this->hd_balancer.start_timer();
			this->atom->add_x_data(host_x, host_type);
		}
		*ilist = this->nbor->host_ilist.begin();
		*jnum = this->nbor->host_acc.begin();

		time_sh_all = 0.0;
		time_ca_all = 0.0;
		time_up_all = 0.0;
		const int red_blocks = loop_annp(eflag, vflag, nghost, nall);

		// updating the energy for each atom
		time_ep.start();
		this->ans->engv.update_host(red_blocks, true);
		time_ep.stop();
		time_ep.sync_stop();
		if (eflag_in) {
			for (int ii = 0; ii < red_blocks; ii++) {
				eng_vdwl += this->ans->engv[ii];
			}
			if(ea_flag)
			for (int ii = 0; ii < red_blocks; ii++) {
				eatom[ii] += this->ans->engv[ii];
			}
		}

		// updating the force for neighbors
		time_fp.start();
		_force.update_host(nall, true);
		time_fp.stop();
		time_fp.sync_stop();
		int nlocal = nall - nghost;
		for (int ii = 0; ii < nall; ii++) {
			int idj = _force[ii].w;
			if (nghost > 0 && (ii >= nlocal && idj == 0))	continue;
			f[idj][0] = _force[ii].x;
			f[idj][1] = _force[ii].y;
			f[idj][2] = _force[ii].z;
		}

		// updating the virial for each atom
		if(va_flag) {
			time_v2.start();
			_virial2.update_host(nall, true);
			time_v2.stop();
			time_v2.sync_stop();
			time_v4.start();
			_virial4.update_host(nall, true);
			time_v4.stop();
			time_v4.sync_stop();
			for(int ii = 0; ii < nall; ii++) {
				vatom_annp[ii][0] = _virial4[ii].x;
				vatom_annp[ii][1] = _virial4[ii].y;
				vatom_annp[ii][2] = _virial4[ii].z;
				vatom_annp[ii][3] = _virial4[ii].w;
				vatom_annp[ii][4] = _virial2[ii].x;
				vatom_annp[ii][5] = _virial2[ii].y;
			}			
		}
		
		_force.clear();
		_virial2.clear();
		_virial4.clear();
		this->hd_balancer.stop_timer();
		return this->nbor->host_jlist.begin() - host_start;
	}

	/*---------------------------------------------------------------------
		calculate energies, forces, and torques
	----------------------------------------------------------------------*/
	template <class numtyp, class acctyp>															// used for the "loop" in "lal_base_annp.cpp" file
	int ANNPMT::loop(const int eflag, const int vflag) {
		int ainum = this->ans->inum();
		const int BX = this->block_size();
		int GX = static_cast<int>(ceil(static_cast<double>(this->ans->inum()) /
										(BX / this->_threads_per_atom)));
		this->time_pair.stop();
		return GX;
	}
	
	/*---------------------------------------------------------------------
		calculate energies, forces, and torques
	----------------------------------------------------------------------*/
	template <class numtyp, class acctyp>
	int ANNPMT::loop_annp(const int eflag, const int vflag, const int nghost, const int nall) {

		// compute the block and grid size to keep all cores busy
		const int BX = this->block_size();		
		int GX = static_cast<int>(ceil(static_cast<double>(this->ans->inum()) /
										(BX / this->_threads_per_atom)));
		int n_Block = 1000, nloop_GX = 0;
		if (n_Block >= GX) {
			n_Block = GX;
			nloop_GX = 1;
		}
		else nloop_GX = GX / n_Block + 1;
		int _num_atoms = n_Block * BX / this->_threads_per_atom;

		int ainum = this->ans->inum();																// for tersoff potential, it is the this->_ainum
		int nbor_pitch = this->nbor->nbor_pitch();

		//*----------------- obtaining the short neighbor list ------------------*/
		if (_max_newj != ainum) {																	// adjust the size of _newj matrix
			_newj.alloc(ainum, *(this->ucl_device), UCL_READ_WRITE, UCL_READ_WRITE);
		}
		this->time_pair.start();
		time_sh.start();
		this->k_short_nbor.set_size(GX, BX);
		this->k_short_nbor.run(&this->atom->x, &_cutMax, &_ntypes, &this->nbor->dev_nbor,			// get the shor neighbor list
								&this->_nbor_data->begin(), &_newj,
								&ainum, &nbor_pitch, &this->_threads_per_atom);

		time_sh.stop();
		time_sh_all += time_sh.time();
		printf("calculate_short_time(ms).... %f, %d, %d\n", time_sh_all, nloop_GX, GX);

		_newj.update_host(ainum, false);															// get the maximum neighbor of matrix
		int _max_nbor_size = 0;
		for (int ii = 0; ii < ainum; ii++) {
			if (_max_nbor_size < _newj[ii])
				_max_nbor_size = _newj[ii];
		}

		int _max_size = static_cast<int>(static_cast<double>(_max_nbor_size) * n_Block * BX / this->_threads_per_atom * 1.10);	
		if (_max_size > nall) {
			_Fj.clear();
			_Fj.alloc(_max_size, *(this->ucl_device), UCL_READ_WRITE);
		}
		int max_dG_size = static_cast<int>(static_cast<double>(_max_size) * _nsf * 1.10);			// for saving the dG_dj value
		if (max_dG_size > nall) {
			_dGij.clear();
			_dGij.alloc(max_dG_size, *(this->ucl_device), UCL_READ_WRITE);
		}
		_gpup.x = _max_nbor_size;
		_gpup.y = _num_atoms;
		_max_force = static_cast<int>(static_cast<double>(nall) * 1.10);
		_force.alloc(_max_force, *(this->ucl_device), UCL_READ_WRITE, UCL_READ_WRITE);
		_force.zero();
		_virial2.alloc(_max_force, *(this->ucl_device), UCL_READ_WRITE, UCL_READ_WRITE);			// for saving the virial
		_virial4.alloc(_max_force, *(this->ucl_device), UCL_READ_WRITE, UCL_READ_WRITE);			// for saving the virial
		_virial2.zero();
		_virial4.zero();
		
		//*----------------- calculating the energy and force -------------------*/
		for (int i = 0; i < nloop_GX; i++) {	
			_dGij.zero();																			// but for initialization this, the n_Block = 20000 can not using for OpenCL
			_Fj.zero();
			int begin_i = i * _num_atoms;
			if (i == nloop_GX - 1)
				_gpup.y = ainum - begin_i;

			time_ca.start();
			this->k_pair_sel->set_size(n_Block, BX);
			this->k_pair_sel->run(&this->atom->x, &_ntypes, &_ntl, &_nhl, &_nnod, &_nsf, 
						          &_npsf, &_ntsf, &_map, &this->_threads_per_atom, 
								  &_cutMax, &_sfnor_scal, &_sfnor_avg, &_out_mod, &eflag, 
								  &_weight_all, &_bias_all, &ainum, &_flagact, 
								  &this->nbor->dev_nbor, &this->_nbor_data->begin(), 
								  &nbor_pitch, &_Fj, &this->ans->engv, 
								  &_force, &_dGij, &_gpup, &begin_i, &_newj);
			time_ca.stop();
			time_ca_all += time_ca.time();															// the time_ca.time() contains the cuEventSynchronize function

			time_up.start();
			this->k_updat.set_size(1, BX);																		
			this->k_updat.run(&this->atom->x, &_newj, &_Fj, &_force, &_virial2, 
							  &_virial4, &eflag, &vflag, &begin_i, &_gpup);
			time_up.stop();
			time_up_all += time_up.time();															// must be use for the update function finish
		}

		printf("update_time(ms)... %f\n", time_up_all);
		printf("calculat_time(ms).. %f\n", time_ca_all);
		this->time_pair.stop();
		return ainum;																				// if return GX, thus, the GX=inum. if not, updating the energy error, because the loop just equal the return value
	}
	template class ANNP<PRECISION, ACC_PRECISION>;
}
