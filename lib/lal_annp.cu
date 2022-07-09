//* Device code *---------------------------------------
//      Artifical Neural Network Potential
//             Accelerated by GPU
//______________________________________________________        
//  begin:  Wed February 16, 2022
//  email:  meng_zhang@metall.t.u-tokyo.ac.jp
//          junya_inoue@metall.t.u-tokyo.ac.jp 
//______________________________________________________
//------------------------------------------------------

#if defined(NV_KERNEL) || defined(USE_HIP)
#include "lal_aux_fun1.h"

#ifndef _DOUBLE_DOUBLE
_texture( pos_tex,float4);
_texture( weight_tex,float);
_texture( bias_tex,float);
_texture( sfsc_tex,float);
_texture( sfav_tex,float);
#else
_texture_2d( pos_tex,int4);
_texture( weight_tex,int2);
_texture( bias_tex,int2);
_texture( sfsc_tex,int2);
_texture( sfav_tex,int2);
#endif

#if (__CUDACC_VER_MAJOR__ >= 11)
#define weight_tex weight_all;
#define bias_tex bias_all
#define sfsc_tex sfnor_scal
#define sfav_tex sfnor_avg
#endif

#else
#define pos_tex x_
#define weight_tex weight_all;
#define bias_tex bias_all
#define sfsc_tex sfnor_scal
#define sfav_tex sfnor_avg
#endif

#define MY_PI (numtyp)3.14159265358979323846
#define coeff_a (numtyp)1.7159
#define coeff_b (numtyp)0.666666666666667
#define coeff_c (numtyp)0.1

#if (SHUFFLE_AVAIL == 0)

#define local_allocate_acc_numj()                                           \
    __local int red_accj_in[BLOCK_PAIR];                                    \
    __local int red_accj_ou[BLOCK_PAIR];

#define acc_numj(newj, in_out, ii, num_in, num_ou, tid, t_per_atom, offset) \
    if (t_per_atom > 1) {                                                   \
        red_accj_in[tid] = num_in;                                          \
        red_accj_ou[tid] = num_ou;                                          \
        for (int s = 0; s < t_per_atom; s++) {                              \
            in_out[s] = red_accj_in[tid - offset + s];                      \
            in_out[s + 10] = red_accj_ou[tid - offset + s];                 \
        }                                                                   \
        for (unsigned int s = t_per_atom / 2; s > 0; s >>= 1) {             \
            simdsync();                                                     \
            if (offset < s) {                                               \
                red_accj_in[tid] += red_accj_in[tid + s];                   \
            }                                                               \
        }                                                                   \
        num_in = red_accj_in[tid];                                          \
    }                                                                       \
    else {                                                                  \
        in_out[offset] = num_in;                                            \
        in_out[offset + 10] = num_ou;                                       \
    }                                                                       \
    if(offset ==0) {                                                        \
        newj[ii] = num_in;                                                  \
    }


#define local_allocate_acc_dGij()                                           \
    __local numtyp red_accj[19][BLOCK_PAIR];

#define acc_dGij(dGij, dG_dkx, dG_dky, dG_dkz, begin_k,                     \
                 ntsf, tid, offset, t_per_atom)                             \
    for (int m = 0; m < ntsf; m++) {                                        \
        red_acc[m][tid] = dG_dk[m].z;                                       \
        for (unsigned int s = t_per_atom / 2; s > 0; s >>= 1) {             \
            simdsync();                                                     \
            if(offset < s) {                                                \
                dG_dkx[m][tid] += dG_dkx[m][tid + s];                       \
                dG_dky[m][tid] += dG_dky[m][tid + s];                       \
                red_acc[m][tid] += red_acc[m][tid + s];                     \
            }                                                               \
        }                                                                   \
        if(offset == 0) {                                                   \
            int index_bm = begin_k + m;                                     \
            dGij[index_bm].x += dG_dkx[m][tid];                             \
            dGij[index_bm].y += dG_dky[m][tid];                             \
            dGij[index_bm].z += red_acc[m][tid];                            \
        }                                                                   \
    }

#define local_allocate_acc_Gi()                                             \
    __local numtyp red_accG[BLOCK_PAIR];

#define acc_Gi(dG_dj, sf_scal, sf_avg, nsf, tid, t_per_atom, offset)        \
    if (t_per_atom > 1) {                                                   \
        for (int i = 0; i < nsf; i++) {                                     \
            red_accG[tid] = dG_dj[i].w;                                     \
            for (unsigned int s = t_per_atom / 2; s > 0; s >>= 1) {         \
                simdsync();                                                 \
                if (offset < s) {                                           \
                    red_accG[tid] += red_accG[tid + s];                     \
                }                                                           \
            }                                                               \
            dG_dj[i].w = red_accG[tid];                                     \
        }                                                                   \
    }                                                                       \
    numtyp sfsc, sfav;                                                      \
    for (int i = 0; i < nsf; i++) {                                         \
        fetch(sfsc, i, sfsc_tex);                                           \
        fetch(sfav, i, sfav_tex);                                           \
        dG_dj[i].w = dG_dj[i].w - sfsc * sfav;                              \
    }                                                                       \
    if (t_per_atom > 1) {                                                   \
        for (int i = 0; i < nsf; i++) {                                     \
            red_accG[tid] = dG_dj[i].w;                                     \
            red_accG[tid] = red_accG[tid - offset];                         \
            dG_dj[i].w = red_accG[tid];                                     \
        }                                                                   \
    }

#define local_allocate_store_answers_annp()                                 \
    __local acctyp red_acc[3][BLOCK_PAIR];

#define store_answers_annp(fi, energy, ii, inum, tid, t_per_atom,           \
                           offset, eflag, engv, force)                      \
    if (t_per_atom > 1) {                                                   \
        red_acc[0][tid] = fi.x;                                             \
        red_acc[1][tid] = fi.y;                                             \
        red_acc[2][tid] = fi.z;                                             \
        for (unsigned int s = t_per_atom / 2; s > 0; s >>= 1) {             \
                simdsync();                                                 \
                if (offset < s) {                                           \
                    for (int i = 0; i < 4; i++)                             \
                        red_acc[i][tid] += red_acc[i][tid + s];             \
                }                                                           \
        }                                                                   \
        fi.x = red_acc[0][tid];                                             \
        fi.y = red_acc[1][tid];                                             \
        fi.z = red_acc[2][tid];                                             \
    }                                                                       \
    if (offset == 0 && ii < inum) {                                         \
        int ei = ii;                                                        \
        if (EVFLAG && eflag) {                                              \
            engv[ei] = energy;                                              \
        }                                                                   \
        acctyp4 old_f = force[ii];                                          \
        old_f.x += fi.x;                                                    \
        old_f.y += fi.y;                                                    \
        old_f.z += fi.z;                                                    \
        force[ii] = old_f;                                                  \
    }

#else

#define local_allocate_acc_numj()                                           \
    __local int red_accj_in[BLOCK_PAIR];                                    \
    __local int red_accj_ou[BLOCK_PAIR];

#define acc_numj(newj, in_out, ii, num_in, num_ou, tid, t_per_atom, offset) \
    if (t_per_atom > 1) {                                                   \
        red_accj_in[tid] = num_in;                                          \
        red_accj_ou[tid] = num_ou;                                          \
        for (int s = 0; s < t_per_atom; s++) {                              \
            in_out[s] = red_accj_in[tid - offset + s];                      \
            in_out[s + 10] = red_accj_ou[tid - offset + s];                 \
        }                                                                   \
        for (unsigned int s = t_per_atom / 2; s > 0; s >>= 1) {             \
            num_in += shfl_down(num_in, s, t_per_atom);                     \
        }                                                                   \
    }                                                                       \
    else {                                                                  \
        in_out[offset] = num_in;                                            \
        in_out[offset + 10] = num_ou;                                       \
    }                                                                       \
    if (offset == 0) {                                                      \
        newj[ii] = num_in;                                                  \
    }

#define local_allocate_acc_dGij()

#define acc_dGij(dGij, dG_dkx, dG_dky, dG_dkz, begin_k,                     \
                 ntsf, tid, offset, t_per_atom)                             \
    for (int m = 0; m < ntsf; m++) {                                        \
        for (unsigned int s = t_per_atom / 2; s > 0; s >>= 1) {             \
            dG_dkz[m] += shfl_down(dG_dkz[m], s, t_per_atom);               \
            simdsync();                                                     \
            if(offset < s) {                                                \
                dG_dkx[m][tid] += dG_dkx[m][tid + s];                       \
                dG_dky[m][tid] += dG_dky[m][tid + s];                       \
            }                                                               \
        }                                                                   \
        if(offset == 0) {                                                   \
            int index_bm = begin_k + m;                                     \
            dGij[index_bm].x += dG_dkx[m][tid];                             \
            dGij[index_bm].y += dG_dky[m][tid];                             \
            dGij[index_bm].z += dG_dkz[m];                                  \
        }                                                                   \
    }

#define local_allocate_acc_Gi()                                             \
    __local numtyp red_accG[BLOCK_PAIR];

#define acc_Gi(dG_dj, sf_scal, sf_avg, nsf, tid, t_per_atom, offset)        \
    if (t_per_atom > 1) {                                                   \
       for (int m = 0; m < nsf; m++) {	                                    \
            for (unsigned int s = t_per_atom / 2; s > 0; s >>= 1) {         \
                dG_dj[m].w += shfl_down(dG_dj[m].w, s, t_per_atom);         \
                simdsync();                                                 \
            }                                                               \
        }                                                                   \
    }                                                                       \
    numtyp sfsc, sfav;                                                      \
    for(int i = 0; i < nsf; i++) {                                          \
        fetch(sfsc, i, sfsc_tex);                                           \
        fetch(sfav, i, sfav_tex);                                           \
        dG_dj[i].w = dG_dj[i].w - sfsc * sfav;                              \
    }                                                                       \
    if (t_per_atom > 1) {                                                   \
        for (int i = 0; i < nsf; i++) {                                     \
            red_accG[tid] = dG_dj[i].w;                                     \
            red_accG[tid] = red_accG[tid - offset];                         \
            dG_dj[i].w = red_accG[tid];                                     \
        }                                                                   \
    }

#define local_allocate_store_answers_annp()

#define store_answers_annp(fi, energy, ii, inum, tid, t_per_atom,           \
                           offset, eflag, engv, force)                      \
    if (t_per_atom > 1) {                                                   \
        for (unsigned int s = t_per_atom / 2; s > 0; s >>= 1) {             \
            fi.x += shfl_down(fi.x, s, t_per_atom);                         \
            fi.y += shfl_down(fi.y, s, t_per_atom);                         \
            fi.z += shfl_down(fi.z, s, t_per_atom);                         \
        }                                                                   \
    }                                                                       \
    if (offset == 0 && ii < inum) {                                         \
        int ei = ii;                                                        \
        if (EVFLAG && eflag) {                                              \
            engv[ei] = energy;                                              \
        }                                                                   \
        acctyp4 old_f = force[ii];                                          \
        old_f.x += fi.x;                                                    \
        old_f.y += fi.y;                                                    \
        old_f.z += fi.z;                                                    \
        force[ii] = old_f;                                                  \
    }

#endif

//---------------------------------------------------------------------
	// get the short neighbor list
//----------------------------------------------------------------------
__kernel void k_annp_short_nbor(const __global numtyp4* restrict x_,
								const __global numtyp* restrict cutsq,
								const int ntypes, 
								__global int* dev_nbor,
								__global int* dev_packed,
								__global int* newj, const int inum, 
								const int nbor_pitch, const int t_per_atom) {
	int tid, ii, offset, n_stride;
	atom_info(t_per_atom, ii, tid, offset);
	local_allocate_acc_numj();

	if (ii < inum) {
		int i, nbor_j, nbor_end, jnum;																
		nbor_info(dev_nbor, dev_packed, nbor_pitch, t_per_atom, ii, 
				  offset, i, jnum, n_stride, nbor_end, nbor_j);

		numtyp4 ix; fetch4(ix, i, pos_tex);
		int nbor_begin = nbor_j;
		int itype = ix.w;																			
		int index_in = 0;																			
		int index_ou = 0;
		int j_out[300], j_in[300];																	
		
		for (; nbor_j < nbor_end; nbor_j += n_stride) {
			int sj = dev_packed[nbor_j];
			int sj_nomask = sj;
			sj &= NEIGHMASK;
			numtyp4 jx; fetch4(jx, sj, pos_tex);
			int jtype = jx.w;

			numtyp deltx = ix.x - jx.x;
			numtyp delty = ix.y - jx.y;
			numtyp deltz = ix.z - jx.z;
			numtyp rsqij = deltx * deltx + delty * delty + deltz * deltz;

			int ijtype = itype * ntypes + jtype;
			if (rsqij <= cutsq[ijtype]) {
				j_in[index_in] = sj_nomask;
				index_in++;
			}
			else {
				j_out[index_ou] = sj_nomask;
				index_ou++;
			}
		}																										
		int in_out[20];																				
		acc_numj(newj, in_out, ii, index_in, index_ou, tid, t_per_atom, offset);					
		int numj = newj[ii];
		dev_nbor[ii + nbor_pitch] = numj;															

		int sum_in = 0, sum_ou = numj;
		nbor_begin -= offset;																		
		for (int j = 0; j < offset; j++) {
			sum_in += in_out[j];
			sum_ou += in_out[j + 10];
		}
		for (int j = 0; j < in_out[offset]; j++) {
			int index = sum_in + j;
			int begin_in = nbor_begin + (index / t_per_atom) * n_stride + index % t_per_atom;
			dev_packed[begin_in] = j_in[j];
		}
		for (int j = 0; j < in_out[offset + 10]; j++) {
			int index = sum_ou + j;
			int begin_ou = nbor_begin + (index / t_per_atom) * n_stride + index % t_per_atom;
			dev_packed[begin_ou] = j_out[j];
		}

		nbor_info(dev_nbor, dev_packed, nbor_pitch, t_per_atom, ii,
				  offset, i, jnum, n_stride, nbor_end, nbor_j);
		for (; nbor_j < nbor_end; nbor_j += n_stride) {
			int sj = dev_packed[nbor_j];
			int sj_nomask = sj;
			sj &= NEIGHMASK;
			numtyp4 jx; fetch4(jx, sj, pos_tex);
		}
	}
}

//---------------------------------------------------------------------
	// __kernel extern "C" __global__, in "ucl_nv_kernel.h" file
//----------------------------------------------------------------------
__kernel void k_annp(const __global numtyp4* restrict x_, const int ntypes,
				 	 const int ntl, const int nhl, const int nnod,
					 const int nsf, const int npsf, const int ntsf,
					 const __global int* restrict map, const int t_per_atom,
					 const __global numtyp* restrict cutsq,
					 const __global numtyp* restrict sfnor_scal,
					 const __global numtyp* restrict sfnor_avg,
					 const numtyp4 out_mod, const int eflag,
					 const __global numtyp* restrict weight_all,
					 const __global numtyp* restrict bias_all, const int inum,
					 const __global int* flagact, const __global int* dev_nbor,
					 const __global int* dev_packed, const int nbor_pitch,
					 __global acctyp4* Fj, 
					 __global acctyp* restrict engv, 
					 __global acctyp4* restrict force, 
					 __global numtyp4* dGij,
					 const int2 gpup, const int begin_i, const __global int* restrict newj) {

	int max_nbor_size = gpup.x;
	numtyp e_scale = out_mod.x;																		
	numtyp e_shift = out_mod.y;
	numtyp e_atom = out_mod.z;

	local_allocate_acc_dGij();
	local_allocate_acc_Gi();
	local_allocate_store_answers_annp();
	acctyp4 f;
	f.x = (acctyp)0; f.y = (acctyp)0; f.z = (acctyp).0;
	acctyp energy;
	energy = (acctyp)0;

	int tid, ii, offset, n_stride;
	atom_info(t_per_atom, ii, tid, offset);																

	int begin_jk = ii * max_nbor_size;																
	int index_bm;
	ii += begin_i;																					
	if (ii < inum) {
		int i, nbor_j, nbor_end, jnum, n_jnum;
		nbor_info(dev_nbor, dev_packed, nbor_pitch, t_per_atom, ii, 
			      offset, i, jnum, n_stride, nbor_end, nbor_j);										

		numtyp4 ix; fetch4(ix, i, pos_tex);															
		int itype = ix.w;
		int idj = offset;																			
		numtyp4 dG_dj[28];

		numtyp dG_dkz[19];																			
		__shared__ numtyp dG_dkx[19][BLOCK_PAIR];
		__shared__ numtyp dG_dky[19][BLOCK_PAIR];
		for (int k = 0; k < 28; k++) {
			dG_dj[k].w = (numtyp)0.0;																
		}

		for (; nbor_j < nbor_end; nbor_j += n_stride, idj += t_per_atom) {
			for (int k = 0; k < 28; k++) {															
				dG_dj[k].x = (numtyp)0.0;
				dG_dj[k].y = (numtyp)0.0;
				dG_dj[k].z = (numtyp)0.0;
			}
			int j = dev_packed[nbor_j];																
			j &= NEIGHMASK;
			numtyp4 jx; fetch4(jx, j, pos_tex);														
			int jtype = jx.w;
			int ijtype = itype * ntypes + jtype;
			numtyp deltx = ix.x - jx.x;
			numtyp delty = ix.y - jx.y;
			numtyp deltz = ix.z - jx.z;
			numtyp rsqij = deltx * deltx + delty * delty + deltz * deltz;

			numtyp Rc_ij = ucl_sqrt(cutsq[ijtype]);
			numtyp rij = ucl_sqrt(rsqij);
			numtyp xij = 2.0 * rij / Rc_ij - 1.0;
			numtyp term1 = MY_PI * rij / Rc_ij;
			numtyp fcij = 0.5 * (cos(term1) + 1.0);
			numtyp dfcij = -0.5 * MY_PI * sin(term1) / Rc_ij;
			numtyp4 tx;
			numtyp4 dtx;
			numtyp sf_scal;
			tx.x = (numtyp)1.0;						tx.y = xij;
			dtx.x = (numtyp)0.0;					dtx.y = (numtyp)1.0;
			numtyp4 term_fc;
			term_fc.x = 2.0 * fcij / Rc_ij;

			fetch(sf_scal, 0, sfsc_tex);															
			dG_dj[0].w += sf_scal * fcij;
			term1 = -dfcij * sf_scal / rij;
			dG_dj[0].x += term1 * deltx;			dG_dj[0].y += term1 * delty;
			dG_dj[0].z += term1 * deltz;

			fetch(sf_scal, 1, sfsc_tex);															
			dG_dj[1].w += sf_scal * fcij * xij;
			term1 = -sf_scal * (term_fc.x + xij * dfcij) / rij;
			dG_dj[1].x += term1 * deltx;			dG_dj[1].y += term1 * delty;
			dG_dj[1].z += term1 * deltz;

			for (int m = 2; m < npsf; m++) {
				fetch(sf_scal, m, sfsc_tex);
				tx.z = 2.0 * xij * tx.y - tx.x;
				dtx.z = 2.0 * tx.y + 2.0 * xij * dtx.y - dtx.x;
				tx.x = tx.y;
				dtx.x = dtx.y;
				tx.y = tx.z;
				dtx.y = dtx.z;

				dG_dj[m].w += sf_scal * fcij * tx.z;												
				term1 = -sf_scal * (dtx.z * term_fc.x + tx.z * dfcij) / rij;
				dG_dj[m].x += term1 * deltx;		dG_dj[m].y += term1 * delty;
				dG_dj[m].z += term1 * deltz;
			}

			int idk = idj, nbor_k, nbor_kend, k_loop;
			k_loop = 1 + jnum / t_per_atom - (int)idj / t_per_atom;
			nbor_k = nbor_j;
			nbor_kend = nbor_j + t_per_atom - offset;
			for (int n = 0; n < k_loop; n++) {
				if (n != 0) {
					nbor_k = nbor_j;
					nbor_k += (n * n_stride - offset);
					nbor_kend = nbor_k + t_per_atom;
				}
				if (nbor_kend > nbor_end)	nbor_kend = nbor_end;
				for (; nbor_k < nbor_kend; nbor_k++, idk++) {
					if (nbor_k == nbor_j)	continue;
					for (int m = 0; m < 19; m++) {													
						dG_dkx[m][tid] = (numtyp)0.0;
						dG_dky[m][tid] = (numtyp)0.0;
						dG_dkz[m] = (numtyp)0.0;
					}

					int k = dev_packed[nbor_k];
					k &= NEIGHMASK;
					numtyp4 kx; fetch4(kx, k, pos_tex);
					int ktype = kx.w;
					int iktype = itype * ntypes + ktype;
					numtyp delt2x = ix.x - kx.x;
					numtyp delt2y = ix.y - kx.y;
					numtyp delt2z = ix.z - kx.z;
					numtyp rsqik = delt2x * delt2x + delt2y * delt2y + delt2z * delt2z;

					numtyp Rc_ik = ucl_sqrt(cutsq[iktype]);
					numtyp rik = ucl_sqrt(rsqik);
					numtyp rinv12 = ucl_recip(rij * rik);
					numtyp cos_theta = (deltx * delt2x + delty * delt2y + deltz * delt2z) * rinv12;
					numtyp xik = 0.5 * (cos_theta + 1.0);
					term1 = MY_PI * rik / Rc_ik;
					numtyp fcik = 0.5 * (cos(term1) + 1.0);
					numtyp dfcik = -0.5 * MY_PI * sin(term1) / Rc_ik;

					numtyp4 dct_dj;
					numtyp4 dct_dk;
					numtyp4 tdGt_dj;
					numtyp4 tdGt_dk;
					term1 = cos_theta / rsqij;
					numtyp term2 = cos_theta / rsqik;
					dct_dj.x = -delt2x * rinv12 + term1 * deltx;
					dct_dj.y = -delt2y * rinv12 + term1 * delty;
					dct_dj.z = -delt2z * rinv12 + term1 * deltz;
					dct_dk.x = -deltx * rinv12 + term2 * delt2x;
					dct_dk.y = -delty * rinv12 + term2 * delt2y;
					dct_dk.z = -deltz * rinv12 + term2 * delt2z;
					tx.x = (numtyp)1.0;							tx.y = xik;
					dtx.x = (numtyp)0.0;						dtx.y = (numtyp)1.0;
					term_fc.x = fcij * fcik;
					term_fc.y = dfcij * fcik / rij;
					term_fc.z = dfcik * fcij / rik;

					fetch(sf_scal, npsf, sfsc_tex);
					dG_dj[npsf].w += sf_scal * term_fc.x;
					term2 = sf_scal * term_fc.y;													
					numtyp term3 = sf_scal * term_fc.z;
					tdGt_dj.x = -term2 * deltx;
					tdGt_dj.y = -term2 * delty;
					tdGt_dj.z = -term2 * deltz;
					tdGt_dk.x = -term3 * delt2x;
					tdGt_dk.y = -term3 * delt2y;
					tdGt_dk.z = -term3 * delt2z;
					dG_dj[npsf].x += tdGt_dj.x;					dG_dj[npsf].y += tdGt_dj.y;
					dG_dj[npsf].z += tdGt_dj.z;
					dG_dkx[0][tid] += tdGt_dk.x;				dG_dky[0][tid] += tdGt_dk.y;
					dG_dkz[0] += tdGt_dk.z;	

					int index_t = npsf + 1;
					fetch(sf_scal, index_t, sfsc_tex);
					dG_dj[index_t].w += sf_scal * tx.y * term_fc.x;
					term1 = 0.5 * sf_scal * term_fc.x;
					term2 = sf_scal * tx.y * term_fc.y;
					term3 = sf_scal * tx.y * term_fc.z;
					tdGt_dj.x = term1 * dct_dj.x - term2 * deltx;
					tdGt_dj.y = term1 * dct_dj.y - term2 * delty;
					tdGt_dj.z = term1 * dct_dj.z - term2 * deltz;
					tdGt_dk.x = term1 * dct_dk.x - term3 * delt2x;
					tdGt_dk.y = term1 * dct_dk.y - term3 * delt2y;
					tdGt_dk.z = term1 * dct_dk.z - term3 * delt2z;

					dG_dj[index_t].x += tdGt_dj.x;				dG_dj[index_t].y += tdGt_dj.y;
					dG_dj[index_t].z += tdGt_dj.z;
					dG_dkx[1][tid] += tdGt_dk.x;				dG_dky[1][tid] += tdGt_dk.y;
					dG_dkz[1] += tdGt_dk.z;

					for (int m = 2; m < ntsf; m++) {
						index_t = m + npsf;
						tx.z = 2.0 * xik * tx.y - tx.x;
						dtx.z = 2.0 * tx.y + 2.0 * xik * dtx.y - dtx.x;
						tx.x = tx.y;
						dtx.x = dtx.y;
						tx.y = tx.z;
						dtx.y = dtx.z;

						fetch(sf_scal, index_t, sfsc_tex);
						dG_dj[index_t].w += sf_scal * tx.z * term_fc.x;
						term1 = 0.5 * sf_scal * dtx.z * term_fc.x;
						term2 = sf_scal * tx.z * term_fc.y;
						term3 = sf_scal * tx.z * term_fc.z;
						tdGt_dj.x = term1 * dct_dj.x - term2 * deltx;
						tdGt_dj.y = term1 * dct_dj.y - term2 * delty;
						tdGt_dj.z = term1 * dct_dj.z - term2 * deltz;
						tdGt_dk.x = term1 * dct_dk.x - term3 * delt2x;
						tdGt_dk.y = term1 * dct_dk.y - term3 * delt2y;
						tdGt_dk.z = term1 * dct_dk.z - term3 * delt2z;

						dG_dj[index_t].x += tdGt_dj.x;			dG_dj[index_t].y += tdGt_dj.y;
						dG_dj[index_t].z += tdGt_dj.z;
						dG_dkx[m][tid] += tdGt_dk.x;			dG_dky[m][tid] += tdGt_dk.y;
						dG_dkz[m] += tdGt_dk.z;
					}																				

					int begin_k = (begin_jk + idk) * nsf + npsf;
					if (n == 0 || t_per_atom == 1)													
						for (int m = 0; m < ntsf; m++) {
							index_bm = begin_k + m;
							dGij[index_bm].x += dG_dkx[m][tid];
							dGij[index_bm].y += dG_dky[m][tid];
							dGij[index_bm].z += dG_dkz[m];
						}
					else {
						acc_dGij(dGij, dG_dkx, dG_dky, dG_dkz, begin_k, ntsf, tid, offset, t_per_atom);				
					}
				}
			}
			int begin_j = (begin_jk + idj) * nsf;
			dGij[begin_j].w = (numtyp)j;
			for (int m = 0; m < nsf; m++) {
				index_bm = begin_j + m;
				dGij[index_bm].x += dG_dj[m].x;
				dGij[index_bm].y += dG_dj[m].y;
				dGij[index_bm].z += dG_dj[m].z;
			}
		}																							
		acc_Gi(dG_dj, sfnor_scal, sfnor_avg, nsf, tid, t_per_atom, offset);

		numtyp hidly[10];																			
		numtyp t_hidly[10];
		numtyp hidly_d[10];																			
		numtyp lays_dw[10 * 28];																	
		numtyp temp_dw[10 * 28];
		numtyp hidly_dw[10 * 28];																												
		numtyp weight, bias;
		int index_w, index_w2;
		for (int m = 0; m < nnod; m++) {
			hidly[m] = 0.0;
			hidly_d[m] = 0.0;
			index_w = m * nsf;
			for (int n = 0; n < nsf; n++) {
				index_w2 = index_w + n;
				hidly_dw[index_w2] = 0.0;															
				temp_dw[index_w2] = 0.0;
			}
		}
		index_w = 0;
		index_w2 = 0;
		int2 nrc_w[3];
		int index_t;
		nrc_w[0].x = nnod; nrc_w[0].y = nsf;
		nrc_w[1].x = nnod; nrc_w[1].y = nnod;
		nrc_w[2].x = 1; nrc_w[2].y = nnod;
		for (int n = 0; n < ntl - 1; n++) {
			for (int k = 0; k < nnod; k++) {
				t_hidly[k] = 0.0;
				index_t = k * nsf;
				for (int m = 0; m < nsf; m++) {														
					lays_dw[index_t + m] = 0.0;
				}
			}
			int actflag = flagact[n];
			for (int k = 0; k < nrc_w[n].x; k++) {
				fetch(bias, k + n * nnod, bias_tex);
				for (int m = 0; m < nrc_w[n].y; m++) {
					weight = weight_all[index_w];
					if (n == 0) {
						t_hidly[k] += weight * dG_dj[m].w;
					}
					else {
						t_hidly[k] += weight * hidly[m];
					}
					index_w++;
				}
				t_hidly[k] += bias;
			}
			for (int k = 0; k < nrc_w[n].x; k++) {													 
				if (actflag == 0) {
					t_hidly[k] = t_hidly[k];
					hidly_d[k] = 1;
				}
				if (actflag == 4) {
					numtyp t_exp = coeff_b * t_hidly[k];
					numtyp t_tanhx = (ucl_exp(t_exp) - ucl_exp(-t_exp)) / (ucl_exp(t_exp) + ucl_exp(-t_exp));
					t_hidly[k] = coeff_a * t_tanhx + coeff_c * t_hidly[k];
					hidly_d[k] = coeff_a * (1.0 - t_tanhx * t_tanhx) * coeff_b + coeff_c;
				}
				hidly[k] = t_hidly[k];
			}
			int index_dw = 0;
			for (int k = 0; k < nrc_w[n].x; k++) {													
				for (int m = 0; m < nrc_w[n].y; m++) {										
					weight = weight_all[index_w2];
					hidly_dw[index_dw] = hidly_d[k] * weight;
					index_w2++;
					index_dw++;
				}
			}																						
			int index_tdw = 0;
			for (int k = 0; k < nrc_w[n].x; k++) {
				for (int m = 0; m < nsf; m++) {
					if (n == 0) {
						temp_dw[k * nsf + m] = hidly_dw[k * nsf + m];
					}
					else {
						for (int j = 0; j < nrc_w[n].y; j++) {
							lays_dw[index_tdw] += hidly_dw[k * nnod + j] * temp_dw[j * nsf + m];
						}
						index_tdw++;
					}
				}
			}
			if (n != 0 && n != ntl - 1)																
				for (int k = 0; k < nrc_w[n].x; k++) {
					for (int m = 0; m < nsf; m++) {
						temp_dw[k * nsf + m] = lays_dw[k * nsf + m];
					}
				}																					
		}
		if (offset == 0) {
			engv[ii] = e_scale * hidly[0] + e_shift + e_atom;
		}

		acctyp4 F;
		for (int jj = offset; jj < jnum; jj += t_per_atom) {
			F.x = 0.0; F.y = 0.0; F.z = 0.0;
			int begin_j = (begin_jk + jj) * nsf;
			for (int k = 0; k < nsf; k++) {
				F.x -= lays_dw[k] * dGij[begin_j + k].x * e_scale;
				F.y -= lays_dw[k] * dGij[begin_j + k].y * e_scale;
				F.z -= lays_dw[k] * dGij[begin_j + k].z * e_scale;
			}
			Fj[begin_jk + jj].x += F.x;
			Fj[begin_jk + jj].y += F.y;
			Fj[begin_jk + jj].z += F.z;
			Fj[begin_jk + jj].w = dGij[begin_j].w;
		}
	}																								
}

//----------------------------------------------------------------------
	// updating the force for neighbor
//----------------------------------------------------------------------
__kernel void k_annp_updat(const __global int* restrict newj, 
						   const __global acctyp4* restrict Fj,
						   __global acctyp4* force,
						   const int begin_i, const int2 gpup) {

	int max_nbor_size = gpup.x;
	int num_atoms = gpup.y;
	int tid = THREAD_ID_X;
	__shared__ int ii;

	for (ii = 0; ii < num_atoms; ) {
		__shared__ acctyp4 tFj[BLOCK_PAIR];
		int n_jnum = newj[ii + begin_i];
		int begin_jk = ii * max_nbor_size;
		if (tid < n_jnum) {
			int idj = begin_jk + tid;
			int index = (int)Fj[idj].w;
			tFj[tid].x = Fj[idj].x;
			tFj[tid].y = Fj[idj].y;
			tFj[tid].z = Fj[idj].z;

			acctyp4 old_f = force[index];															
			old_f.w = index;
			old_f.x += tFj[tid].x;
			old_f.y += tFj[tid].y;
			old_f.z += tFj[tid].z;
			force[index] = old_f;
		}
		for (unsigned int s = n_jnum / 2; s > 0; s >>= 1) {
			__syncthreads();
			if (tid < s) {
				tFj[tid].x += tFj[tid + s].x;
				tFj[tid].y += tFj[tid + s].y;
				tFj[tid].z += tFj[tid + s].z;
			}
			__syncthreads();
			if (s % 2 == 1 && s != 1 && tid == 0) {													
				tFj[tid].x += tFj[tid + s - 1].x;
				tFj[tid].y += tFj[tid + s - 1].y;
				tFj[tid].z += tFj[tid + s - 1].z;
			}
		}

		if (n_jnum % 2 == 1 && tid == 0) {															
			tFj[tid].x += tFj[tid + n_jnum - 1].x;
			tFj[tid].y += tFj[tid + n_jnum - 1].y;
			tFj[tid].z += tFj[tid + n_jnum - 1].z;
		}
		if (tid == 0) {
			acctyp4 old_f = force[ii + begin_i];													
			old_f.w = (numtyp)(ii + begin_i);
			old_f.x -= tFj[tid].x;
			old_f.y -= tFj[tid].y;
			old_f.z -= tFj[tid].z;
			force[ii + begin_i] = old_f;
		}
		__syncthreads();		
		ii++;
	}
}
