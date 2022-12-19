//* Device code *---------------------------------------
//      Artifical Neural Network Potential
//             Accelerated by GPU
//______________________________________________________        
//  begin:  Wed February 16, 2022
//  email:  
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
            in_out[s + 20] = red_accj_ou[tid - offset + s];                 \
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
        in_out[offset + 20] = num_ou;                                       \
    }                                                                       \
    if(offset == 0) {                                                       \
        newj[ii] = num_in;                                                  \
    }                                                                       \
	simdsync();


#define local_allocate_acc_dGij()                                           \
    __local numtyp red_accj[19][BLOCK_PAIR];

#define acc_dGij(dGij, dG_dkx, dG_dky, dG_dkz, begin_k,                     \
                 ntsf, tid, offset, t_per_atom)                             \
    for (int m = 0; m < ntsf; m++) {                                        \
        red_accj[m][tid] = dG_dk[m].z;                                      \
        for (unsigned int s = t_per_atom / 2; s > 0; s >>= 1) {             \
            simdsync();                                                     \
            if(offset < s) {                                                \
                dG_dkx[m][tid] += dG_dkx[m][tid + s];                       \
                dG_dky[m][tid] += dG_dky[m][tid + s];                       \
                red_accj[m][tid] += red_accj[m][tid + s];                   \
            }                                                               \
        }                                                                   \
        if(offset == 0) {                                                   \
            int index_bm = begin_k + m;                                     \
            dGij[index_bm].x += dG_dkx[m][tid];                             \
            dGij[index_bm].y += dG_dky[m][tid];                             \
            dGij[index_bm].z += red_accj[m][tid];                           \
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
        dG_dj[i].w = sfsc * dG_dj[i].w - sfsc * sfav;                       \
    }                                                                       \
    if (t_per_atom > 1) {                                                   \
        for (int i = 0; i < nsf; i++) {                                     \
            red_accG[tid] = dG_dj[i].w;                                     \
            red_accG[tid] = red_accG[tid - offset];                         \
            dG_dj[i].w = red_accG[tid];                                     \
        }                                                                   \
    }

#define local_allocate_acc_Fi()                                             \
    __local acctyp red_accfi[3][BLOCK_PAIR];

#define acc_Fi(Fi, force, i, tid, t_per_atom, offset)                       \
    if (t_per_atom > 1) {                                                   \
        red_accfi[0][tid] = Fi.x;                                           \
        red_accfi[1][tid] = Fi.y;                                           \
        red_accfi[2][tid] = Fi.z;                                           \
        for (unsigned int s = t_per_atom / 2; s > 0; s >>= 1) {             \
                simdsync();                                                 \
                if (offset < s) {                                           \
                    for (int i = 0; i < 3; i++)                             \
                        red_accfi[i][tid] += red_accfi[i][tid + s];         \
                }                                                           \
        }                                                                   \
    }                                                                       \
    if (offset == 0 && ii) {                                                \
        acctyp4 old_f = force[ii];                                          \
        old_f.x += red_accfi[0][tid];                                       \
        old_f.y += red_accfi[1][tid];                                       \
        old_f.z += red_accfi[2][tid];                                       \
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
            in_out[s + 20] = red_accj_ou[tid - offset + s];                 \
        }                                                                   \
        for (unsigned int s = t_per_atom / 2; s > 0; s >>= 1) {             \
            num_in += shfl_down(num_in, s, t_per_atom);                     \
        }                                                                   \
    }                                                                       \
    else {                                                                  \
        in_out[offset] = num_in;                                            \
        in_out[offset + 20] = num_ou;                                       \
    }                                                                       \
    if (offset == 0) {                                                      \
        newj[ii] = num_in;                                                  \
    }                                                                       \
	simdsync();

#define local_allocate_acc_dGij()

#define acc_dGij(dGij, dG_dkx, dG_dky, dG_dkz, begin_k,                     \
                 ntsf, tid, offset, t_per_atom)                             \
    for (int m = 0; m < ntsf; m++) {                                        \
        for (unsigned int s = t_per_atom / 2; s > 0; s >>= 1) {             \
            dG_dkz[m] += shfl_down(dG_dkz[m], s, t_per_atom);               \
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
            }                                                               \
        }                                                                   \
    }                                                                       \
    numtyp sfsc, sfav;                                                      \
    for(int i = 0; i < nsf; i++) {                                          \
        fetch(sfsc, i, sfsc_tex);                                           \
        fetch(sfav, i, sfav_tex);                                           \
        dG_dj[i].w = sfsc * dG_dj[i].w - sfsc * sfav;                       \
    }                                                                       \
    if (t_per_atom > 1) {                                                   \
        for (int i = 0; i < nsf; i++) {                                     \
            red_accG[tid] = dG_dj[i].w;                                     \
            red_accG[tid] = red_accG[tid - offset];                         \
            dG_dj[i].w = red_accG[tid];                                     \
        }                                                                   \
    }

#define local_allocate_acc_Fi()

#define acc_Fi(Fi, force, ii, tid, t_per_atom, offset)                      \
    if (t_per_atom > 1) {                                                   \
        for (unsigned int s = t_per_atom / 2; s > 0; s >>= 1) {             \
            Fi.x += shfl_down(Fi.x, s, t_per_atom);                         \
            Fi.y += shfl_down(Fi.y, s, t_per_atom);                         \
            Fi.z += shfl_down(Fi.z, s, t_per_atom);                         \
        }                                                                   \
    }                                                                       \
    if (offset == 0) {                                                      \
        acctyp4 old_f = force[ii];                                          \
        old_f.x += Fi.x;                                                    \
        old_f.y += Fi.y;                                                    \
        old_f.z += Fi.z;                                                    \
        force[ii] = old_f;                                                  \
    }

#endif

//---------------------------------------------------------------------
	// get the short neighbor list
//----------------------------------------------------------------------
__kernel void k_annp_short_nbor(const __global numtyp4* restrict x_,
								const numtyp cutMax,
								const int ntypes, 
								__global int* dev_nbor,
								__global int* dev_packed,
								__global int* newj, const int inum, 
								const int nbor_pitch, const int t_per_atom) {
	int tid, ii, offset, n_stride;
	atom_info(t_per_atom, ii, tid, offset);

	local_allocate_acc_numj();
	if (ii < inum) {
		int i, nbor_j, nbor_end, jnum;																// get the information of the neighbor-list
		nbor_info(dev_nbor, dev_packed, nbor_pitch, t_per_atom, ii, 
				  offset, i, jnum, n_stride, nbor_end, nbor_j);

		numtyp4 ix; fetch4(ix, i, pos_tex);
		int nbor_begin = nbor_j;																			
		int index_in = 0;																			// for accumulating the number of j
		int index_ou = 0;
		int j_out[300], j_in[300];																	// j_in save for neighbor j within cutoff, j_out save for j without  cutoff
		
		for (; nbor_j < nbor_end; nbor_j += n_stride) {
			int sj = dev_packed[nbor_j];
			int sj_nomask = sj;
			sj &= NEIGHMASK;
			numtyp4 jx; fetch4(jx, sj, pos_tex);

			numtyp deltx = ix.x - jx.x;
			numtyp delty = ix.y - jx.y;
			numtyp deltz = ix.z - jx.z;
			numtyp r2ij = deltx * deltx + delty * delty + deltz * deltz;

			if (r2ij <= cutMax && r2ij > 1.0e-10) {
				j_in[index_in] = sj_nomask;
				index_in++;
			}
			else {
				j_out[index_ou] = sj_nomask;
				index_ou++;
			}
		}																							// cutsq is modified in "lal_atom.h" file, change it to one array			
		int in_out[40];																				// for saving the number of "in" vlaue (0-9), and "out" value(10-19)�� maximum value is 10 > t_per_atoms
		acc_numj(newj, in_out, ii, index_in, index_ou, tid, t_per_atom, offset);					// accumulate the number of "J" atoms, the value saved into "newj" matirx
		int numj = newj[ii];
		dev_nbor[ii + nbor_pitch] = numj;															// update the number of neighbors

		int sum_in = 0, sum_ou = numj;
		nbor_begin -= offset;																		// to initial position
		for (int j = 0; j < offset; j++) {
			sum_in += in_out[j];
			sum_ou += in_out[j + 20];
		}
		for (int j = 0; j < in_out[offset]; j++) {
			int index = sum_in + j;
			int begin_in = nbor_begin + (index / t_per_atom) * n_stride + index % t_per_atom;
			dev_packed[begin_in] = j_in[j];
		}
		for (int j = 0; j < in_out[offset + 20]; j++) {
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
					 const __global int* restrict map, const int t_per_atom,
					 const numtyp cutMax,
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
	numtyp e_scale = out_mod.x;																		// will be changed into numtyp format
	numtyp e_shift = out_mod.y;
	numtyp e_atom = out_mod.z;
	numtyp Rc = ucl_sqrt(cutMax);

	local_allocate_acc_dGij();
	local_allocate_acc_Gi();
	local_allocate_acc_Fi();
	int tid, ii, offset, n_stride;
	atom_info(t_per_atom, ii, tid, offset);															// in "lal_aux_fun1.h" file	

	//---------------- starting calculation
	int begin_jk = ii * max_nbor_size;																// Fj, dGij, matrix always start from 0
	int index_bm;
	ii += begin_i;																					// ii should update

	//if(ii == 2)
	if (ii < inum) {
		int i, nbor_j, nbor_end, jnum;
		nbor_info(dev_nbor, dev_packed, nbor_pitch, t_per_atom, ii, 
			      offset, i, jnum, n_stride, nbor_end, nbor_j);										// get the nbor information of atom i

		numtyp4 ix; fetch4(ix, i, pos_tex);															// define in "lal_preporcessor.h" file
		int idj = offset;																			// index for dGij, because the 0-28 is for atom i
		numtyp4 dG_dj[28] = { 0.0,0.0,0.0,0.0 };

		numtyp dG_dkz[19];																			// for saving the value of dG_dk, just can two shared memory fo dG_dkx, dG_dky
		__shared__ numtyp dG_dkx[19][BLOCK_PAIR];
		__shared__ numtyp dG_dky[19][BLOCK_PAIR];

		for (; nbor_j < nbor_end; nbor_j += n_stride, idj += t_per_atom) {
			for (int k = 0; k < nsf; k++) {															// set zero for atom j
				dG_dj[k].x = (numtyp)0.0;
				dG_dj[k].y = (numtyp)0.0;
				dG_dj[k].z = (numtyp)0.0;
			}
			int j = dev_packed[nbor_j];																// the 3rd row is the starting location in "packed nobrs"
			j &= NEIGHMASK;
			numtyp4 jx; fetch4(jx, j, pos_tex);														// get the coordinates of j
			numtyp deltx = ix.x - jx.x;
			numtyp delty = ix.y - jx.y;
			numtyp deltz = ix.z - jx.z;
			numtyp r2ij = deltx * deltx + delty * delty + deltz * deltz;

			//---------------- pair symmetry function	
			numtyp x_fc, xij, term1, term2, term1_coeff, term2_coeff;
			numtyp rij = ucl_sqrt(r2ij);
			numtyp coe_0 = MY_PI / Rc;
			x_fc = coe_0 * rij;
			numtyp fcij = 0.5 * (cos(x_fc) + 1.0);
			numtyp dfcij = -0.5 * coe_0 * sin(x_fc);

			numtyp4 dr_dj, tx, dtx;
			dr_dj.x = -deltx / rij;
			dr_dj.y = -delty / rij;
			dr_dj.z = -deltz / rij;

			xij = 2.0 * rij / Rc - 1.0;
			tx.x = (numtyp)1.0;						tx.y = xij;
			dtx.x = (numtyp)0.0;					dtx.y = (numtyp)1.0;
			term1_coeff = 2.0 * fcij / Rc;

			dG_dj[0].w += fcij;
			dG_dj[0].x += dfcij * dr_dj.x;			dG_dj[0].y += dfcij* dr_dj.y;
			dG_dj[0].z += dfcij * dr_dj.z;

			dG_dj[1].w += fcij * xij;
			term1 = term1_coeff + xij * dfcij ;
			dG_dj[1].x += term1 * dr_dj.x;			dG_dj[1].y += term1 * dr_dj.y;
			dG_dj[1].z += term1 * dr_dj.z;

			for (int m = 2; m < npsf; m++) {
				tx.z = 2.0 * xij * tx.y - tx.x;
				dtx.z = 2.0 * tx.y + 2.0 * xij * dtx.y - dtx.x;
				tx.x = tx.y;						dtx.x = dtx.y;
				tx.y = tx.z;						dtx.y = dtx.z;

				dG_dj[m].w += fcij * tx.z;												// This is G value 
				term1 = dtx.z * term1_coeff + tx.z * dfcij;
				dG_dj[m].x += term1 * dr_dj.x;		dG_dj[m].y += term1 * dr_dj.y;
				dG_dj[m].z += term1 * dr_dj.z;
				//printf("G pair.... %d %d %d %d %f %f %f %f\n", m, ii, offset, j, rij, Rc, fcij, tx.z);
			}
			//-------------- triple symmetry function
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
					for (int m = 0; m < ntsf; m++) {													// set zero for atom k
						dG_dkx[m][tid] = (numtyp)0.0;
						dG_dky[m][tid] = (numtyp)0.0;
						dG_dkz[m] = (numtyp)0.0;
					}

					int k = dev_packed[nbor_k];
					k &= NEIGHMASK;
					numtyp4 kx; fetch4(kx, k, pos_tex);
					numtyp delt2x = ix.x - kx.x;
					numtyp delt2y = ix.y - kx.y;
					numtyp delt2z = ix.z - kx.z;
					numtyp r2ik = delt2x * delt2x + delt2y * delt2y + delt2z * delt2z;
					numtyp rik = ucl_sqrt(r2ik);
					numtyp rinv12 = ucl_recip(rij * rik);
					numtyp cos_theta = (deltx * delt2x + delty * delt2y + deltz * delt2z) * rinv12;
					// if(ii == 2)
					// 	printf("check11... %d %d %d %d %d %f\n", ii, offset, j, k, n, dG_dj[9].w);
						
					x_fc = coe_0 * rik;
					numtyp fcik = 0.5 * (cos(x_fc) + 1.0);
					numtyp dfcik = -0.5 * coe_0 * sin(x_fc);

					numtyp4 dr_dk, dct_dj, dct_dk, tdGt_dj, tdGt_dk;
					numtyp term_cos1, term_cos2;
					dr_dk.x = - delt2x / rik;
					dr_dk.y = - delt2y / rik;
					dr_dk.z = - delt2z / rik;
					
					term_cos1 = cos_theta / r2ij;
					term_cos2 = cos_theta / r2ik;
					dct_dj.x = -delt2x * rinv12 + term_cos1 * deltx;
					dct_dj.y = -delt2y * rinv12 + term_cos1 * delty;
					dct_dj.z = -delt2z * rinv12 + term_cos1 * deltz;
					dct_dk.x = -deltx * rinv12 + term_cos2 * delt2x;
					dct_dk.y = -delty * rinv12 + term_cos2 * delt2y;
					dct_dk.z = -deltz * rinv12 + term_cos2 * delt2z;

					numtyp xik = 0.5 * (cos_theta + 1.0);
					tx.x = (numtyp)1.0;							tx.y = xik;
					dtx.x = (numtyp)0.0;						dtx.y = (numtyp)1.0;

					numtyp4 term2_dj, term2_dk;
					term2 = fcij * fcik;
					term2_coeff = dfcij * fcik;
					term2_dj.x = term2_coeff * dr_dj.x;			term2_dj.y = term2_coeff * dr_dj.y;
					term2_dj.z = term2_coeff * dr_dj.z;
					term2_coeff = dfcik * fcij;
					term2_dk.x = term2_coeff * dr_dk.x;			term2_dk.y = term2_coeff * dr_dk.y;
					term2_dk.z = term2_coeff * dr_dk.z;

					// for G0 value
					dG_dj[npsf].w += term2;
					dG_dj[npsf].x += term2_dj.x;				dG_dj[npsf].y += term2_dj.y;
					dG_dj[npsf].z += term2_dj.z;
					dG_dkx[0][tid] += term2_dk.x;				dG_dky[0][tid] += term2_dk.y;
					dG_dkz[0] += term2_dk.z;	

					// for G1 value
					int index_t = npsf + 1;
					dG_dj[index_t].w += tx.y * term2;
					term1_coeff = 0.5  * term2;
					tdGt_dj.x = term1_coeff * dct_dj.x + tx.y * term2_dj.x;
					tdGt_dj.y = term1_coeff * dct_dj.y + tx.y * term2_dj.y;
					tdGt_dj.z = term1_coeff * dct_dj.z + tx.y * term2_dj.z;
					tdGt_dk.x = term1_coeff * dct_dk.x + tx.y * term2_dk.x;
					tdGt_dk.y = term1_coeff * dct_dk.y + tx.y * term2_dk.y;
					tdGt_dk.z = term1_coeff * dct_dk.z + tx.y * term2_dk.z;

					dG_dj[index_t].x += tdGt_dj.x;				dG_dj[index_t].y += tdGt_dj.y;
					dG_dj[index_t].z += tdGt_dj.z;
					dG_dkx[1][tid] += tdGt_dk.x;				dG_dky[1][tid] += tdGt_dk.y;
					dG_dkz[1] += tdGt_dk.z;
					
					// for G2-ntsf value
					for (int m = 2; m < ntsf; m++) {
						index_t = npsf + m;
						tx.z = 2.0 * xik * tx.y - tx.x;
						dtx.z = 2.0 * tx.y + 2.0 * xik * dtx.y - dtx.x;
						tx.x = tx.y;
						dtx.x = dtx.y;
						tx.y = tx.z;
						dtx.y = dtx.z;
						dG_dj[index_t].w += tx.z * term2;

						numtyp t_term1_coeff = term1_coeff * dtx.z;
						tdGt_dj.x = t_term1_coeff * dct_dj.x + tx.z * term2_dj.x;
						tdGt_dj.y = t_term1_coeff * dct_dj.y + tx.z * term2_dj.y;
						tdGt_dj.z = t_term1_coeff * dct_dj.z + tx.z * term2_dj.z;
						tdGt_dk.x = t_term1_coeff * dct_dk.x + tx.z * term2_dk.x;
						tdGt_dk.y = t_term1_coeff * dct_dk.y + tx.z * term2_dk.y;
						tdGt_dk.z = t_term1_coeff * dct_dk.z + tx.z * term2_dk.z;

						dG_dj[index_t].x += tdGt_dj.x;
						dG_dj[index_t].y += tdGt_dj.y;
						dG_dj[index_t].z += tdGt_dj.z;

						dG_dkx[m][tid] += tdGt_dk.x;
						dG_dky[m][tid] += tdGt_dk.y;
						dG_dkz[m] += tdGt_dk.z;
					}	// ntsf loop

					int begin_k = (begin_jk + idk) * nsf + npsf;

					if (n == 0 || t_per_atom == 1)																	// updating the dG_dk, values
						for (int m = 0; m < ntsf; m++) {
							index_bm = begin_k + m;
							dGij[index_bm].x += dG_dkx[m][tid];
							dGij[index_bm].y += dG_dky[m][tid];
							dGij[index_bm].z += dG_dkz[m];
						}
					else {
					// if(ii == 2)
					// 	printf("check00... %d %d %d %d %d %d %d %d %f\n", ii, offset, j, k, n, begin_jk, max_nbor_size, begin_k, dG_dj[9].w);
						acc_dGij(dGij, dG_dkx, dG_dky, dG_dkz, begin_k, ntsf, tid, offset, t_per_atom);				// in this case, all threads are processed the same "k"
					// if(ii == 2)
					// 	printf("check22... %d %d %d %d %d %d %d %d %f\n", ii, offset, j, k, n, begin_jk, max_nbor_size, begin_k, dG_dj[9].w);
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
				//printf("%d, %d, %d, %f�� %f, %f\n", i, m, index_bm, dG_dj[m].x, dG_dj[m].y, dG_dj[m].z);
			}
		}																									// j loop
		acc_Gi(dG_dj, sfnor_scal, sfnor_avg, nsf, tid, t_per_atom, offset);
		//  if(offset == 0)
		//  	for(int n = 0; n < nsf; n++)
		//  		printf("checking....000... %d %d %d %f %f %f %f\n", n, ii, offset, ix.x, ix.y, ix.z, dG_dj[n].w);


		numtyp hidly[10] = { 0.0 };																			// saving the value of hidly layer
		numtyp t_hidly[10] = { 0.0 };
		numtyp hidly_d[10] = { 0.0 };																		// derivate of hidly layer
		numtyp lays_dw[10 * 28] = { 0.0 };																	// for 2 layers multipulity
		numtyp temp_dw[10 * 28] = { 0.0 };
		numtyp hidly_dw[10 * 28] = { 0.0 };																	// using for update		

		//if(offset == 0)
		//	for(int n = 0; n < nsf; n++)
		//		printf("checking... %d %d %f\n", ii, offset, dG_dj[n].w);

		numtyp weight, bias;
		int index_w = 0, index_w2 = 0, index_t;
		int2 nrc_w[3];
		nrc_w[0].x = nnod; nrc_w[0].y = nsf;
		nrc_w[1].x = nnod; nrc_w[1].y = nnod;
		nrc_w[2].x = 1; nrc_w[2].y = nnod;
		for (int n = 0; n < ntl - 1; n++) {
			for (int k = 0; k < nnod; k++) {
				t_hidly[k] = 0.0;
				index_t = k * nsf;
				for (int m = 0; m < nsf; m++) {														// set 0
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
			for (int k = 0; k < nrc_w[n].x; k++) {													// for activation 
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
			for (int k = 0; k < nrc_w[n].x; k++) {													// hidly_d multiply the weight
				for (int m = 0; m < nrc_w[n].y; m++) {
					//fetch(weight, index_w2, weight_tex);											// cannot be used for RTX A5000
					weight = weight_all[index_w2];
					hidly_dw[index_dw] = hidly_d[k] * weight;
					index_w2++;
					index_dw++;
				}
			}																						// for geting dE_dG
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
			if (n != 0 && n != ntl - 1)																// updating the temp matrix
				for (int k = 0; k < nrc_w[n].x; k++) {
					for (int m = 0; m < nsf; m++) {
						temp_dw[k * nsf + m] = lays_dw[k * nsf + m];
					}
				}																					// update the tt_dw
		}
		// get energy and force for atom i: wq
		if (offset == 0) {
			engv[ii] = e_scale * hidly[0] + e_shift + e_atom;
			//printf("energy.... %d %f\n", ii, engv[ii]);
		}

		// force for atom j
		acctyp4 temp_f, Fi;
		numtyp scaling;
		Fi.x = 0.0; Fi.y = 0.0; Fi.z = 0.0;
		for (int jj = offset; jj < jnum; jj += t_per_atom) {
			temp_f.x = 0.0; temp_f.y = 0.0; temp_f.z = 0.0;
			int begin_j = (begin_jk + jj) * nsf;
			for (int k = 0; k < nsf; k++) {
				fetch(scaling, k, sfsc_tex);
				temp_f.x -= scaling * lays_dw[k] * dGij[begin_j + k].x * e_scale;
				temp_f.y -= scaling * lays_dw[k] * dGij[begin_j + k].y * e_scale;
				temp_f.z -= scaling * lays_dw[k] * dGij[begin_j + k].z * e_scale;
			}
			Fj[begin_jk + jj].x += temp_f.x;
			Fj[begin_jk + jj].y += temp_f.y;
			Fj[begin_jk + jj].z += temp_f.z;
			Fj[begin_jk + jj].w = dGij[begin_j].w;

			Fi.x -= temp_f.x;
			Fi.y -= temp_f.y;
			Fi.z -= temp_f.z;
		}
		acc_Fi(Fi, force, ii, tid, t_per_atom, offset);
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

		if (tid < n_jnum) {
			int idj = begin_jk + tid;
			int indexj = (int)Fj[idj].w;
			tFj[tid].x = Fj[idj].x;
			tFj[tid].y = Fj[idj].y;
			tFj[tid].z = Fj[idj].z;

			acctyp4 old_f = force[indexj];															// old values
			old_f.w = indexj;
			old_f.x += tFj[tid].x;
			old_f.y += tFj[tid].y;
			old_f.z += tFj[tid].z;
			force[indexj] = old_f;

			if(EVFLAG && vflag) {
				fetch4(ix, indexi, pos_tex);
				numtyp4 jx; fetch4(jx, indexj, pos_tex);
				numtyp delx = ix.x - jx.x;
				numtyp dely = ix.y - jx.y;
				numtyp delz = ix.z - jx.z;	

				tvirial[tid][0] = delx*-tFj[tid].x;
				tvirial[tid][1] = dely*-tFj[tid].y;
				tvirial[tid][2] = delz*-tFj[tid].z;
				tvirial[tid][3] = delx*-tFj[tid].y;
				tvirial[tid][4] = delx*-tFj[tid].z;
				tvirial[tid][5] = dely*-tFj[tid].z;
						
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
		for (unsigned int s = n_jnum / 2; s > 0; s >>= 1) {
			int idtid = tid + s;
			__syncthreads();
			if (tid < s) {
				if(EVFLAG && vflag) {
					tvirial[tid][0] += tvirial[idtid][0];
					tvirial[tid][1] += tvirial[idtid][1];
					tvirial[tid][2] += tvirial[idtid][2];
					tvirial[tid][3] += tvirial[idtid][3];
					tvirial[tid][4] += tvirial[idtid][4];
					tvirial[tid][5] += tvirial[idtid][5];
				}
			}
			__syncthreads();
			if (s % 2 == 1 && s != 1 && tid == 0) {													// in case of the odd number of value occuring at sub-layers
				idtid -= 1;
				if(EVFLAG && vflag) {
					tvirial[tid][0] += tvirial[idtid][0];
					tvirial[tid][1] += tvirial[idtid][1];
					tvirial[tid][2] += tvirial[idtid][2];
					tvirial[tid][3] += tvirial[idtid][3];
					tvirial[tid][4] += tvirial[idtid][4];
					tvirial[tid][5] += tvirial[idtid][5];
				}
			}
		}
		if (n_jnum % 2 == 1 && tid == 0) {															// for the n_jnum is equal to odd
			int idtid = tid + n_jnum - 1;
			if(EVFLAG && vflag) {
				tvirial[tid][0] += tvirial[idtid][0];
				tvirial[tid][1] += tvirial[idtid][1];
				tvirial[tid][2] += tvirial[idtid][2];
				tvirial[tid][3] += tvirial[idtid][3];
				tvirial[tid][4] += tvirial[idtid][4];
				tvirial[tid][5] += tvirial[idtid][5];
			}
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
		__syncthreads();		
		ii++;
	}
}
