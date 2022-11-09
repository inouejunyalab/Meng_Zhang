//* Device code *---------------------------------------
//      Artifical Neural Network Potential
//             Accelerated by GPU
//______________________________________________________        
//  begin:  Mon Oct 23, 2022
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
_texture( sfmi_tex,float);
_texture( cofsym_tex, float4);
#else
_texture_2d( pos_tex,int4);
_texture( weight_tex,int2);
_texture( bias_tex,int2);
_texture( sfsc_tex,int2);
_texture( sfmi_tex,int2);
_texture( cofsym_tex, int4);
#endif

#if (__CUDACC_VER_MAJOR__ >= 11)
#define weight_tex weight_all;
#define bias_tex bias_all
#define sfsc_tex sf_scal
#define sfmi_tex sf_min
#define cofsym_tex coeff_sym
#endif

#else
#define pos_tex x_
#define weight_tex weight_all;
#define bias_tex bias_all
#define sfsc_tex sf_scal
#define sfmi_tex sf_min
#define cofsym_tex coeff_sym
#endif

#define MY_PI (numtyp)3.14159265358979323846
#define coeff_a (numtyp)1.7159
#define coeff_b (numtyp)0.666666666666667
#define coeff_c (numtyp)0.1
#define CFLENGTH (numtyp)1.889726
#define CFFORCE (numtyp)51.422515

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
            in_out[s + 35] = red_accj_ou[tid - offset + s];                 \
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
        in_out[offset + 35] = num_ou;                                       \
    }                                                                       \
    if(offset ==0) {                                                        \
        newj[ii] = num_in;                                                  \
    }                                                                       \
	simdsync();


#define local_allocate_acc_dGij()                                           \
    __local numtyp red_accj[24][BLOCK_PAIR];								\
	__shared__ numtyp red_acck2[BLOCK_PAIR];

#define acc_dGij(dGij, dG_dkx, dG_dky, dG_dkyL, dG_dkz, begin_k,            \
                 ntsf, tid, offset, t_per_atom, myMAX)                      \
    red_acck2[tid] = dG_dkyL;                                               \
    for (unsigned int s = t_per_atom / 2; s > 0; s >>= 1) {                 \
        simdsync();                                                         \
        if(offset < s)                                                      \
            red_acck2[tid] += red_acck2[tid + s];                           \
    }                                                                       \
    for (int m = 0; m < ntsf; m++) {                                        \
        red_accj[m][tid] = dG_dkz[m];                                       \
        for (unsigned int s = t_per_atom / 2; s > 0; s >>= 1) {             \
            simdsync();                                                     \
            if(offset < s) {                                                \
                dG_dkx[m][tid] += dG_dkx[m][tid + s];                       \
                if(m != myMAX)                                              \
                     dG_dky[m][tid] += dG_dky[m][tid + s];                  \
                red_accj[m][tid] += red_accj[m][tid + s];                   \
            }                                                               \
        }                                                                   \
        if(offset == 0) {                                                   \
            int index_bm = begin_k + m;                                     \
            dGij[index_bm].x += dG_dkx[m][tid];                             \
            if(m != myMAX)                                                  \
                dGij[index_bm].y += dG_dky[m][tid];                         \
            dGij[index_bm].z += red_accj[m][tid];                           \
        }                                                                   \
    }                                                                       \
	if (offset == 0)                                                        \
		dGij[begin_k + myMAX].y += red_acck2[tid];

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
    numtyp sfsc, sfmi;                                                      \
    for (int i = 0; i < nsf; i++) {                                         \
        fetch(sfsc, i, sfsc_tex);                                           \
        fetch(sfmi, i, sfmi_tex);                                           \
        dG_dj[i].w = (dG_dj[i].w - sfmi) / sfsc;                            \
    }                                                                       \
    if (t_per_atom > 1) {                                                   \
        for (int i = 0; i < nsf; i++) {                                     \
            red_accG[tid] = dG_dj[i].w;                                     \
            red_accG[tid] = red_accG[tid - offset];                         \
            dG_dj[i].w = red_accG[tid];                                     \
        }                                                                   \
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
            in_out[s + 35] = red_accj_ou[tid - offset + s];                 \
        }                                                                   \
        for (unsigned int s = t_per_atom / 2; s > 0; s >>= 1) {             \
            num_in += shfl_down(num_in, s, t_per_atom);                     \
        }                                                                   \
    }                                                                       \
    else {                                                                  \
        in_out[offset] = num_in;                                            \
        in_out[offset + 35] = num_ou;                                       \
    }                                                                       \
    if (offset == 0) {                                                      \
        newj[ii] = num_in;                                                  \
    }                                                                       \
	simdsync();

#define local_allocate_acc_dGij()

#define acc_dGij(dGij, dG_dkx, dG_dky, dG_dkyL, dG_dkz, begin_k,            \
                 ntsf, tid, offset, t_per_atom, myMAX)                      \
    for (unsigned int s = t_per_atom / 2; s > 0; s >>= 1)                   \
        dG_dkyL += shfl_down(dG_dkyL, s, t_per_atom);                       \
    for (int m = 0; m < ntsf; m++) {                                        \
        for (unsigned int s = t_per_atom / 2; s > 0; s >>= 1) {             \
            dG_dkz[m] += shfl_down(dG_dkz[m], s, t_per_atom);               \
            simdsync();                                                     \
            if(offset < s) {                                                \
                dG_dkx[m][tid] += dG_dkx[m][tid + s];                       \
                if(m != myMAX)                                              \
                    dG_dky[m][tid] += dG_dky[m][tid + s];                   \
            }                                                               \
        }                                                                   \
        if(offset == 0) {                                                   \
            int index_bm = begin_k + m;                                     \
            dGij[index_bm].x += dG_dkx[m][tid];                             \
            if(m != myMAX)                                                  \
                dGij[index_bm].y += dG_dky[m][tid];                         \
            dGij[index_bm].z += dG_dkz[m];                                  \
        }                                                                   \
    }                                                                       \
    if(offset == 0)                                                         \
       dGij[begin_k + myMAX].y += dG_dkyL;

#define local_allocate_acc_Gi()                                             \
    __local numtyp red_accG[BLOCK_PAIR];

#define acc_Gi(dG_dj, sf_scal, sf_avg, nsf, tid, t_per_atom, offset)        \
    numtyp sfsc, sfmi;                                                      \
    for (int i = 0; i < nsf; i++) {                                         \
        fetch(sfsc, i, sfsc_tex);                                           \
        fetch(sfmi, i, sfmi_tex);                                           \
        dG_dj[i].w = (dG_dj[i].w - sfmi) / sfsc;                            \
    }

#endif

//---------------------------------------------------------------------
	// get the short neighbor list
//----------------------------------------------------------------------
__kernel void k_annp_short_nbor(const __global numtyp4* restrict x_,
								const numtyp cutoff,
								__global int* dev_nbor,
								__global int* dev_packed,
								__global int* newj, const int inum, 
								const int nbor_pitch, const int t_per_atom) {
	int tid, ii, offset, n_stride;
	atom_info(t_per_atom, ii, tid, offset);
	local_allocate_acc_numj();

	if (ii < inum) {
		int i, nbor_j, nbor_end, jnum;																// for getting the information of the neighbor-list
		nbor_info(dev_nbor, dev_packed, nbor_pitch, t_per_atom, ii, 
				  offset, i, jnum, n_stride, nbor_end, nbor_j);

		numtyp4 ix; fetch4(ix, i, pos_tex);
		int nbor_begin = nbor_j;																			
		int index_in = 0;																		
		int index_ou = 0;
		int j_out[300], j_in[300];																
		
		numtyp cutoff2 = cutoff * cutoff;
		for (; nbor_j < nbor_end; nbor_j += n_stride) {
			int sj = dev_packed[nbor_j];
			int sj_nomask = sj;
			sj &= NEIGHMASK;
			numtyp4 jx; fetch4(jx, sj, pos_tex);

			numtyp deltx = ix.x - jx.x;
			numtyp delty = ix.y - jx.y;
			numtyp deltz = ix.z - jx.z;
			numtyp r2ij = CFLENGTH * CFLENGTH * (deltx * deltx + delty * delty + deltz * deltz);
			if (r2ij < cutoff2) {
				j_in[index_in] = sj_nomask;
				index_in++;
			}
			else {
				j_out[index_ou] = sj_nomask;
				index_ou++;
			}
		}																						
		int in_out[70];																				
		acc_numj(newj, in_out, ii, index_in, index_ou, tid, t_per_atom, offset);					
		int numj = newj[ii];
		dev_nbor[ii + nbor_pitch] = numj;													

		int sum_in = 0, sum_ou = numj;
		nbor_begin -= offset;																	
		for (int j = 0; j < offset; j++) {
			sum_in += in_out[j];
			sum_ou += in_out[j + 35];
		}
		for (int j = 0; j < in_out[offset]; j++) {
			int index = sum_in + j;
			int begin_in = nbor_begin + (index / t_per_atom) * n_stride + index % t_per_atom;
			dev_packed[begin_in] = j_in[j];
		}
		for (int j = 0; j < in_out[offset + 35]; j++) {
			int index = sum_ou + j;
			int begin_ou = nbor_begin + (index / t_per_atom) * n_stride + index % t_per_atom;
			dev_packed[begin_ou] = j_out[j];
		}
	}
}

//---------------------------------------------------------------------
	// __kernel extern "C" __global__, in "ucl_nv_kernel.h" file
//----------------------------------------------------------------------
__kernel void k_annp(const __global numtyp4* restrict x_, const int ntypes,
				 	 const int ntl, const int nhl, const int nnod,
					 const int nsf, const int npsf, const int ntsf,
					 const int t_per_atom, const __global numtyp* restrict sf_scal,
					 const __global numtyp* restrict sf_min,
					 const numtyp4 out_mod, const int eflag,
					 const __global numtyp* restrict weight_all,
					 const __global numtyp* restrict bias_all, const int inum,
					 const __global int* flagact, const __global int* dev_nbor,
					 const __global int* dev_packed, const int nbor_pitch,
					 __global acctyp4* Fj, __global numtyp4* dGij,
					 __global acctyp* restrict engv, 
					 __global acctyp4* restrict force, 
					 const __global numtyp4* restrict coeff_sym,
					 const int2 gpup, const int begin_i, const __global int* restrict newj) {

	int max_nbor_size = gpup.x;

	local_allocate_acc_dGij();
	int tid, ii, offset, n_stride;
	atom_info(t_per_atom, ii, tid, offset);														

	//---------------- starting calculation
	int begin_jk = ii * max_nbor_size;														
	int index_bm;
	ii += begin_i;																				
	if (ii < inum) {
		int i, nbor_j, nbor_end, jnum;
		nbor_info(dev_nbor, dev_packed, nbor_pitch, t_per_atom, ii, 
			      offset, i, jnum, n_stride, nbor_end, nbor_j);									

		numtyp4 ix; fetch4(ix, i, pos_tex);														
		int idj = offset;																		
		numtyp4 dG_dj[27] = { 0.0,0.0,0.0,0.0 };

		//-------------------- starting -------------
		for (; nbor_j < nbor_end; nbor_j += n_stride, idj += t_per_atom) {
			for (int k = 0; k < nsf; k++) {														
				dG_dj[k].x = (numtyp)0.0;
				dG_dj[k].y = (numtyp)0.0;
				dG_dj[k].z = (numtyp)0.0;
			}
			int j = dev_packed[nbor_j];													
			j &= NEIGHMASK;
			numtyp4 jx; fetch4(jx, j, pos_tex);								
			numtyp deltx = ix.x - jx.x;
			numtyp delty = ix.y - jx.y;
			numtyp deltz = ix.z - jx.z;
			numtyp r2ij = deltx * deltx + delty * delty + deltz * deltz;
			numtyp rij = ucl_sqrt(r2ij);
			numtyp rij_m = rij * CFLENGTH;
			numtyp r2ij_m = rij_m * rij_m;

			numtyp4 dr_dj, coeff;														
			dr_dj.x = -deltx / rij;
			dr_dj.y = -delty / rij;
			dr_dj.z = -deltz / rij;
			numtyp Rc, coe_0, coe_fc, fcij, dfcij, term1, term2, term3;
			fetch4(coeff, 0, cofsym_tex);
			Rc = coeff.w;

			coe_0 = MY_PI / Rc;
			coe_fc = coe_0 * rij_m;
			fcij = 0.5 * (cos(coe_fc) + 1.0);
			dfcij = -0.5 * coe_0 * sin(coe_fc);
			for (int m = 0; m < npsf; m++) {
				fetch4(coeff, m, cofsym_tex);
				term1 = ucl_exp(-coeff.x * r2ij_m);
				term2 = term1 * (-2.0 * fcij * coeff.x * rij_m + dfcij);				
				dG_dj[m].w += term1 * fcij;													
				dG_dj[m].x += term2 * dr_dj.x;		
				dG_dj[m].y += term2 * dr_dj.y;												
				dG_dj[m].z += term2 * dr_dj.z;
			}

			//-------------- triple symmetry function
			int idk, nbor_k; 
			numtyp4 dr_dk, dr_djk, dct_dj, dct_dk, term2_drj;
			numtyp4 term2_drk, term2_t, term3_drj, term3_drk;

			nbor_k = nbor_j + n_stride;
			idk = idj + 1;
			for (; nbor_k < nbor_end; nbor_k += n_stride, idk++) {
				//if (nbor_k == nbor_j)	continue;			
				numtyp dG_dkx[24] = { 0.0 };
				numtyp dG_dky[24] = { 0.0 };
				numtyp dG_dkz[24] = { 0.0 };

				int k = dev_packed[nbor_k];
				k &= NEIGHMASK;
				numtyp4 kx; fetch4(kx, k, pos_tex);

				numtyp delt2x = ix.x - kx.x;								numtyp delt2y = ix.y - kx.y;
				numtyp delt2z = ix.z - kx.z;
				numtyp delt3x = jx.x - kx.x;								numtyp delt3y = jx.y - kx.y;
				numtyp delt3z = jx.z - kx.z;
				numtyp r2ik = delt2x * delt2x + delt2y * delt2y + delt2z * delt2z;
				numtyp r2jk = delt3x * delt3x + delt3y * delt3y + delt3z * delt3z;
				numtyp rik = ucl_sqrt(r2ik);
				numtyp rjk = ucl_sqrt(r2jk);
				numtyp rik_m = rik * CFLENGTH;
				numtyp rjk_m = rjk * CFLENGTH;

				if (rjk_m < Rc) {
					numtyp rinv12 = ucl_recip(rij * rik);
					numtyp cos_theta = (deltx * delt2x + delty * delt2y + deltz * delt2z) * rinv12;
					dr_dk.x = -delt2x / rik;								dr_dk.y = -delt2y / rik;
					dr_dk.z = -delt2z / rik;
					dr_djk.x = delt3x / rjk;								dr_djk.y = delt3y / rjk;
					dr_djk.z = delt3z / rjk;

					term1 = cos_theta / r2ij;
					term2 = cos_theta / r2ik;
					dct_dj.x = -delt2x * rinv12 + term1 * deltx;
					dct_dj.y = -delt2y * rinv12 + term1 * delty;
					dct_dj.z = -delt2z * rinv12 + term1 * deltz;
					dct_dk.x = -deltx * rinv12 + term2 * delt2x;
					dct_dk.y = -delty * rinv12 + term2 * delt2y;
					dct_dk.z = -deltz * rinv12 + term2 * delt2z;					

					numtyp r2sum = CFLENGTH * CFLENGTH * (r2ij + r2ik + r2jk);
					term2_t.x = rjk_m * dr_djk.x;							term2_t.y = rjk_m * dr_djk.y;
					term2_t.z = rjk_m * dr_djk.z;
					term2_drj.x = 2.0 * (rij_m * dr_dj.x + term2_t.x);		term2_drj.y = 2.0 * (rij_m * dr_dj.y + term2_t.y);
					term2_drj.z = 2.0 * (rij_m * dr_dj.z + term2_t.z);
					term2_drk.x = 2.0 * (rik_m * dr_dk.x - term2_t.x);		term2_drk.y = 2.0 * (rik_m * dr_dk.y - term2_t.y);
					term2_drk.z = 2.0 * (rik_m * dr_dk.z - term2_t.z);

					numtyp coe_fcik = coe_0 * rik_m;
					numtyp fcik = 0.5 * (cos(coe_fcik) + 1.0);
					numtyp dfcik = -0.5 * coe_0 * sin(coe_fcik);

					numtyp coe_fcjk = coe_0 * rjk_m;
					numtyp fcjk = 0.5 * (cos(coe_fcjk) + 1.0);
					numtyp dfcjk = -0.5 * coe_0 * sin(coe_fcjk);

					numtyp term_fc = fcij * fcik * fcjk;
					numtyp4 term3_t;
					term3_t.x = fcjk * dfcij;								term3_t.y = fcij * dfcjk;
					term3_t.z = fcjk * dfcik;								term3_t.w = fcik * dfcjk;
					term3_drj.x = fcik * (term3_t.x * dr_dj.x + term3_t.y * dr_djk.x);
					term3_drj.y = fcik * (term3_t.x * dr_dj.y + term3_t.y * dr_djk.y);
					term3_drj.z = fcik * (term3_t.x * dr_dj.z + term3_t.y * dr_djk.z);
					term3_drk.x = fcij * (term3_t.z * dr_dk.x - term3_t.w * dr_djk.x);
					term3_drk.y = fcij * (term3_t.z * dr_dk.y - term3_t.w * dr_djk.y);
					term3_drk.z = fcij * (term3_t.z * dr_dk.z - term3_t.w * dr_djk.z);

					for (int m = 0; m < ntsf; m++) {
						int index_t = npsf + m;
						fetch4(coeff, index_t, cofsym_tex);											// eta, lambda, zeta, Rc (x, y, z, w)
						numtyp flag = 1 + coeff.y * cos_theta;
						if (flag <= 0)	continue;

						numtyp term_coe = pow(2, 1 - coeff.z);
						numtyp term_cot = term_coe * pow(flag, coeff.z);
						numtyp term_exp = ucl_exp(-coeff.x * r2sum);
						dG_dj[index_t].w += term_cot * term_exp * term_fc;

						term1 = coeff.y * term_cot * term_exp * term_fc * coeff.z / flag / CFLENGTH;
						term3 = term_cot * term_exp;
						term2 = term3 * term_fc * coeff.x;		
						dG_dj[index_t].x += term1 * dct_dj.x - term2 * term2_drj.x + term3 * term3_drj.x;
						dG_dj[index_t].y += term1 * dct_dj.y - term2 * term2_drj.y + term3 * term3_drj.y;
						dG_dj[index_t].z += term1 * dct_dj.z - term2 * term2_drj.z + term3 * term3_drj.z;
						dG_dkx[m] = term1 * dct_dk.x - term2 * term2_drk.x + term3 * term3_drk.x;
						dG_dky[m] = term1 * dct_dk.y - term2 * term2_drk.y + term3 * term3_drk.y;
						dG_dkz[m] = term1 * dct_dk.z - term2 * term2_drk.z + term3 * term3_drk.z;
					}
				}
				// update the value of dG_dk
				int begin_k = (begin_jk + idk) * nsf + npsf;
				for (int m = 0; m < ntsf; m++) {
					index_bm = begin_k + m;
					dGij[index_bm].x += dG_dkx[m];
					dGij[index_bm].y += dG_dky[m];
					dGij[index_bm].z += dG_dkz[m];
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
		}	// j loop
		acc_Gi(dG_dj, sf_scal, sf_min, nsf, tid, t_per_atom, offset);

		numtyp weight, bias, tsf_scal;
		numtyp hidly[24] = { 0.0 };															
		numtyp t_hidly[24] = { 0.0 };
		numtyp hidly_d[24] = { 0.0 };														
		numtyp lays_dw[24 * 27] = { 0.0 };													
		numtyp temp_dw[24 * 27] = { 0.0 };
		numtyp hidly_dw[24 * 27] = { 0.0 };																								
		int index_w = 0, index_w2 = 0, index_t;
		int2 nrc_w[3];
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
					//fetch(weight, index_w, weight_tex);
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
					hidly[k] = t_hidly[k];
					hidly_d[k] = 1;
				}
				if (actflag == 4) {
					numtyp t_exp = t_hidly[k];
					hidly[k] = (ucl_exp(t_exp) - ucl_exp(-t_exp)) / (ucl_exp(t_exp) + ucl_exp(-t_exp));
					hidly_d[k] = (1.0 - hidly[k] * hidly[k]);
				}
			}
			// for the dE_dG
			int index_dw = 0;
			for (int k = 0; k < nrc_w[n].x; k++) {											
				for (int m = 0; m < nrc_w[n].y; m++) {
					//fetch(weight, index_w2, weight_tex);											// cannot be used for RTX A5000
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
		engv[ii] = hidly[0];																		// updating the erergy

		// force for atom i and j
		acctyp4 F;
		int begin_i = begin_jk + ii;
		for (int jj = offset; jj < jnum; jj += t_per_atom) {
			F.x = 0.0; F.y = 0.0; F.z = 0.0;
			int begin_j = (begin_jk + jj) * nsf;
			for (int k = 0; k < nsf; k++) {
				fetch(tsf_scal, k, sfsc_tex);
				F.x -= lays_dw[k] * dGij[begin_j + k].x / tsf_scal * CFFORCE;
				F.y -= lays_dw[k] * dGij[begin_j + k].y / tsf_scal * CFFORCE;
				F.z -= lays_dw[k] * dGij[begin_j + k].z / tsf_scal * CFFORCE;
			}
			int index_jk = begin_i + jj + 1;
			Fj[index_jk].x += F.x;
			Fj[index_jk].y += F.y;
			Fj[index_jk].z += F.z;
			Fj[index_jk].w = dGij[begin_j].w;

			Fj[begin_i].x -= F.x;
			Fj[begin_i].y -= F.y;
			Fj[begin_i].z -= F.z;
		}
	}	// if ii
}

//----------------------------------------------------------------------
	// updating the force for neighbor
//----------------------------------------------------------------------
__kernel void k_annp_updat(const __global numtyp4* restrict x_,
						   const __global int* restrict newj, 
						   const __global acctyp4* restrict Fj,
						   __global acctyp4* force,
						   __global acctyp2* virial2,
						   __global acctyp4* virial4,
						   const int eflag, const int vflag, 
						   const int begin_i, const int2 gpup) {

	int max_nbor_size = gpup.x;
	int num_atoms = gpup.y;
	int tid = THREAD_ID_X;
	__shared__ int ii;
	__shared__ numtyp4 ix;

	acctyp2 old_v2;
	acctyp4 old_v4;
	for (ii = 0; ii < num_atoms; ) {
		__shared__ acctyp4 tFj[BLOCK_PAIR];
		__shared__ acctyp tvirial[BLOCK_PAIR][6];

		int indexi = ii + begin_i;
		int n_jnum = newj[indexi];
		int begin_jk = ii * max_nbor_size;
		int begin_i = begin_jk + ii;
		fetch4(ix, indexi, pos_tex);															
		
		if (tid == 0) {																		
			acctyp4 old_f = force[indexi];
			old_f.x += Fj[begin_i].x;
			old_f.y += Fj[begin_i].y;
			old_f.z += Fj[begin_i].z;
			force[indexi] = old_f;
		}
		if (tid < n_jnum) {
			int idj = begin_i + tid + 1;
			int indexj = (int)Fj[idj].w;
			tFj[tid].x = Fj[idj].x;
			tFj[tid].y = Fj[idj].y;
			tFj[tid].z = Fj[idj].z;

			acctyp4 old_f = force[indexj];												
			old_f.w = indexj;
			old_f.x += tFj[tid].x;
			old_f.y += tFj[tid].y;
			old_f.z += tFj[tid].z;
			force[indexj] = old_f;

			if(EVFLAG && vflag) {
				numtyp4 jx; fetch4(jx, indexj, pos_tex);
				numtyp delx = ix.x - jx.x;
				numtyp dely = ix.y - jx.y;
				numtyp delz = ix.z - jx.z;

				tvirial[tid][0] = delx*-tFj[tid].x;
				tvirial[tid][1] = dely*-tFj[tid].y;
				tvirial[tid][2] = delz*-tFj[tid].z;
				tvirial[tid][3] = delx*-tFj[tid].y;
				tvirial[tid][4] = delx*-tFj[tid].z;
			
				old_v2 = virial2[indexj];
				old_v4 = virial4[indexj];	
				old_v4.x += 0.5 * tvirial[tid][0];
				old_v4.y += 0.5 * tvirial[tid][1];
				old_v4.z += 0.5 * tvirial[tid][2];
				old_v4.w += 0.5 * tvirial[tid][3];
				old_v2.x += 0.5 * tvirial[tid][4];
				old_v2.y += 0.5 * tvirial[tid][5];
				virial2[indexj] = old_v2;
				virial4[indexj] = old_v4;
			}
		}
		if (EVFLAG && vflag) {
			for (unsigned int s = n_jnum / 2; s > 0; s >>= 1) {
				int idtid = tid + s;
				__syncthreads();
				if (tid < s) {
					tvirial[tid][0] += tvirial[idtid][0];
					tvirial[tid][1] += tvirial[idtid][1];
					tvirial[tid][2] += tvirial[idtid][2];
					tvirial[tid][3] += tvirial[idtid][3];
					tvirial[tid][4] += tvirial[idtid][4];
					tvirial[tid][5] += tvirial[idtid][5];
				}
				__syncthreads();
				if (s % 2 == 1 && s != 1 && tid == 0) {											
					idtid -= 1;
					tvirial[tid][0] += tvirial[idtid][0];
					tvirial[tid][1] += tvirial[idtid][1];
					tvirial[tid][2] += tvirial[idtid][2];
					tvirial[tid][3] += tvirial[idtid][3];
					tvirial[tid][4] += tvirial[idtid][4];
					tvirial[tid][5] += tvirial[idtid][5];
				}
			}
			if (n_jnum % 2 == 1 && tid == 0) {												
				int idtid = tid + n_jnum - 1;
				tvirial[tid][0] += tvirial[idtid][0];
				tvirial[tid][1] += tvirial[idtid][1];
				tvirial[tid][2] += tvirial[idtid][2];
				tvirial[tid][3] += tvirial[idtid][3];
				tvirial[tid][4] += tvirial[idtid][4];
				tvirial[tid][5] += tvirial[idtid][5];
			}
			if (tid == 0) {
				old_v2 = virial2[indexi];
				old_v4 = virial4[indexi];						
				old_v4.x += 0.5 * tvirial[tid][0];
				old_v4.y += 0.5 * tvirial[tid][1];
				old_v4.z += 0.5 * tvirial[tid][2];
				old_v4.w += 0.5 * tvirial[tid][3];
				old_v2.x += 0.5 * tvirial[tid][4];
				old_v2.y += 0.5 * tvirial[tid][5];
				virial2[indexi] = old_v2;
				virial4[indexi] = old_v4;
			}
		}
		__syncthreads();
		ii++;
	}
}
