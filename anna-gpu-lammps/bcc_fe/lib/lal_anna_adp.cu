//* Device code *---------------------------------------
//      Physics-informed Neural Network Potential
//             Accelerated by GPU
//______________________________________________________
//  begin:  Monday August 07, 2023
//  email:  meng_zhang@metall.t.u-tokyo.ac.jp
//          junya_inoue@metall.t.u-tokyo.ac.jp
//______________________________________________________
//------------------------------------------------------

#if defined(NV_KERNEL) || defined(USE_HIP)
#include "lal_aux_fun1.h"

#ifndef _DOUBLE_DOUBLE
_texture( pos_tex,float4);
_texture( rho_tex,float);
_texture( imu_tex,float);
_texture( lambda_tex,float);
_texture( ladp_tex,float);
_texture( weight_tex,float);
_texture( bias_tex,float);
#else
_texture_2d( pos_tex,int4);
_texture( rho_tex, int2);
_texture( imu_tex, int2);
_texture( lambda_tex,int2);
_texture( ladp_tex,int2);
_texture( weight_tex,int2);
_texture( bias_tex,int2);
#endif

#if (__CUDACC_VER_MAJOR__ >= 11)
#define weight_tex weight_all;
#define bias_tex bias_all
#define rho_tex adp_rho
#define imu_tex adp_mu
#define lambda_tex adp_lambda
#define ladp_tex ladp_params
#endif

#else
#define pos_tex x_
#define rho_tex adp_rho
#define imu_tex adp_mu
#define lambda_tex adp_lambda
#define ladp_tex ladp_params
#define weight_tex weight_all;
#define bias_tex bias_all
#endif

#define MY_PI (numtyp)3.14159265358979323846
#define coeff_a (numtyp)1.7
#define coeff_b (numtyp)0.6

#if (SHUFFLE_AVAIL == 0)

#define local_allocate_acc_numj()                                           \
    __local int red_accj_in[BLOCK_PAIR];                                    \
    __local int red_accj_ou[BLOCK_PAIR];

#define acc_numj(dev_nbor, in_out, ii, nbor_pitch, num_in,                  \
                 num_ou, tid, t_per_atom, offset)                           \
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
    } else {                                                                \
        in_out[offset] = num_in;                                            \
        in_out[offset + 20] = num_ou;                                       \
    }                                                                       \
    if(offset == 0) {                                                       \
        dev_nbor[ii + nbor_pitch] = num_in;                                 \
    }

#define local_allocate_acc_Gi()                                             \
    __local numtyp red_accGi[BLOCK_PAIR];

#define acc_Gi(G_i, nsf, tid, t_per_atom, offset)                           \
    if (t_per_atom > 1) {                                                   \
        for (int i = 0; i < nsf; i++) {                                     \
            red_accGi[tid] = G_i[i]                                         \
            for (unsigned int s = t_per_atom / 2; s > 0; s >>= 1) {         \
                simdsync();                                                 \
                if (offset < s)                                             \
                    red_accGi[tid] += red_accGi[tid + s];                   \
            }                                                               \
            G_i[i] = red_accG[tid - offset];                                \
        }                                                                   \
    }

#define local_allocate_acc_hide()                                           \
    __local numtyp red_acchid[BLOCK_PAIR];

#define acc_hide(hid, t_hid, nnod, tid, t_per_atom, offset)                 \
    if(t_per_atom > 1) {                                                    \
        for(int i = 0; i < nnod; i++) {                                     \
            red_acchid[tid] = t_hid[i];                                     \
            for (unsigned int s = t_per_atom / 2; s > 0; s >>= 1) {         \
                simdsync();                                                 \
                if (offset < s)                                             \
                    red_acchid[tid] += red_acchid[tid + s];                 \
            }                                                               \
            hid[i] = red_acchid[tid - offset];                              \
        }                                                                   \
    }

#define local_allocate_store_energy_padp()                                  \
    __local acctyp red_accrho[BLOCK_PAIR];                                  \
    __local acctyp red_accmu[BLOCK_PAIR][3];                                \
    __local acctyp red_acclamb[BLOCK_PAIR][6];

#define store_energy_padp(rho_i, mu_i, lambda_i, c1F, c2F, adp_rho,         \
                          adp_mu, adp_lambda, energy, ii, i, nall, inum,    \
                          tid, t_per_atom, offset, eflag, vflag, engv)      \
    if (t_per_atom > 1) {                                                   \
        red_accrho[tid] = rho_i;                                            \
         for (unsigned int s = t_per_atom / 2; s > 0; s >>= 1) {            \
            simdsync();                                                     \
            if (offset < s)                                                 \
                red_accrho[tid] += red_accrho[tid + s];                     \
        }                                                                   \
        rho_i = red_accrho[tid];                                            \
        for (int k = 0; k < 3; k++) {                                       \
            red_accmu[tid][k] = mu_i[k];                                    \
            for (unsigned int s = t_per_atom / 2; s > 0; s >>= 1) {         \
                simdsync();                                                 \
                if (offset < s)                                             \
                    red_accmu[tid][k] += red_accmu[tid + s][k];             \
           }                                                                \
           mu_i[k] = red_accmu[tid][k];                                     \
        }                                                                   \
        for (int k = 0; k < 6; k++) {                                       \
            red_acclamb[tid][k] = mu_i[k];                                  \
            for (unsigned int s = t_per_atom / 2; s > 0; s >>= 1) {         \
                simdsync();                                                 \
                if (offset < s)                                             \
                    red_acclamb[tid][k] += red_acclamb[tid + s][k];         \
           }                                                                \
           lambda_i[k] = red_acclamb[tid][k];                               \
        }                                                                   \
    }                                                                       \
    if (offset == 0 && ii < inum) {                                         \
        acctyp v_i = lambda_i[0] + lambda_i[1] + lambda_i[2];               \
        acctyp sum_mu = 0.0, sum_lamb = 0.0;                                \
        adp_rho[i] = rho_i;                                                 \
        for(int k = 0; k < 6; k++) {                                        \
            if(k < 3) {                                                     \
                sum_mu += mu_i[k] * mu_i[k];                                \
                sum_lamb += lambda_i[k] * lambda_i[k];                      \
                adp_mu[i + k * nall] = mu_i[k];                             \
            }                                                               \
            adp_lambda[i + k * nall] = lambda_i[k];                         \
        }                                                                   \
        sum_lamb += 2.0 * (pow(lambda_i[3], 2) + pow(lambda_i[4], 2) +      \
                    pow(lambda_i[5], 2));                                   \
        energy = c1F * ucl_sqrt(rho_i) + c2F * pow(rho_i, 2) +              \
                 0.5 * sum_mu + 0.5 * sum_lamb - 1.0 / 6.0 * v_i * v_i;     \
        if(EVFLAG && eflag)                                                 \
            engv[ii] = energy;                                              \
    }

#define local_allocate_store_answer_anna_adp()                              \
    __local acctyp red_acc[6][BLOCK_PAIR];

#define store_answer_anna_adp(f, energy, virial, ii, inum,                  \
                              tid, e_base, t_per_atom,                      \
                              offset, eflag, vflag, ans, engv)              \
    if (t_per_atom > 1) {                                                   \
        red_acc[0][tid] = f.x;                                              \
        red_acc[1][tid] = f.y;                                              \
        red_acc[2][tid] = f.z;                                              \
        red_acc[3][tid] = energy;                                           \
        for (unsigned int s = t_per_atom / 2; s > 0; s >>= 1) {             \
            simdsync();                                                     \
            if (offset < s) {                                               \
                for (int n = 0; n < 4; n++)                                 \
                    red_acc[n][tid] += red_acc[n][tid + s];                 \
            }                                                               \
        }                                                                   \
        f.x = red_acc[0][tid];                                              \
        f.y = red_acc[1][tid];                                              \
        f.z = red_acc[2][tid];                                              \
        energy = red_acc[3][tid];                                           \
        if (EVFLAG && vflag) {                                              \
            simdsync();                                                     \
            for (int n = 0; n < 6; n++)                                     \
                red_acc[n][tid] = virial[n];                                \
            for (unsigned int s = t_per_atom / 2; s > 0; s >>= 1) {         \
                simdsync();                                                 \
                if (offset < s) {                                           \
                    for (int n = 0; n < 6; n++)                             \
                        red_acc[n][tid] += red_acc[n][tid + s];             \
                }                                                           \
            }                                                               \
            for (int n = 0; n < 6; n++)                                     \
                virial[n] = red_acc[n][tid];                                \
        }                                                                   \
    }                                                                       \
    if (offset == 0 && ii < inum) {                                         \
        int ei = ii;                                                        \
        if (EVFLAG && eflag) {                                              \
            engv[ei] += 0.5 * energy + e_base;                              \
            ei += inum;                                                     \
        }                                                                   \
        if (EVFLAG && vflag) {                                              \
            for (int n = 0; n < 6; n++) {                                   \
                engv[ei] = virial[i] * (acctyp)0.5;                         \
                ei += inum;                                                 \
            }                                                               \
        }                                                                   \
        ans[ii] = f;                                                        \
    }

#else

#define local_allocate_acc_numj()                                           \
    __local int red_accj_in[BLOCK_PAIR];                                    \
    __local int red_accj_ou[BLOCK_PAIR];

#define acc_numj(dev_nbor, in_out, ii, nbor_pitch, num_in,                  \
                 num_ou, tid, t_per_atom, offset)                           \
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
    } else {                                                                \
        in_out[offset] = num_in;                                            \
        in_out[offset + 20] = num_ou;                                       \
    }                                                                       \
    if(offset == 0) {                                                       \
        dev_nbor[ii + nbor_pitch] = num_in;                                 \
    }

#define local_allocate_acc_Gi()                                             \
    __local numtyp red_accGi[BLOCK_PAIR];

#define acc_Gi(Gi, nsf, tid, t_per_atom, offset)                            \
    if (t_per_atom > 1) {                                                   \
       for (int m = 0; m < nsf; m++) {                                      \
            for (unsigned int s = t_per_atom / 2; s > 0; s >>= 1) {         \
                G_i[m] += shfl_down(G_i[m], s, t_per_atom);                 \
            }                                                               \
            red_accGi[tid] = G_i[m];                                        \
            red_accGi[tid] = red_accGi[tid - offset];                       \
            G_i[m] = red_accGi[tid];                                        \
        }                                                                   \
    }

#define local_allocate_acc_hide()                                           \
    __local numtyp red_acchid[BLOCK_PAIR];

#define acc_hide(hid, t_hid, nnod, tid, t_per_atom, offset)                 \
    if(t_per_atom > 1) {                                                    \
        for(int i = 0; i < nnod; i++) {                                     \
            for (unsigned int s = t_per_atom / 2; s > 0; s >>= 1) {         \
                 t_hid[i] += shfl_down(t_hid[i], s, t_per_atom);            \
            }                                                               \
            red_acchid[tid] = t_hid[i];                                     \
            red_acchid[tid] = red_acchid[tid - offset];                     \
            hid[i] = red_acchid[tid];                                       \
        }                                                                   \
    }

#define local_allocate_store_energy_padp()

#define store_energy_padp(rho_i, mu_i, lambda_i, c1F, c2F, adp_rho,         \
                          adp_mu, adp_lambda, energy, ii, i, nall, inum,    \
                          tid, t_per_atom, offset, eflag, vflag, engv)      \
    if (t_per_atom > 1) {                                                   \
         for (unsigned int s = t_per_atom / 2; s > 0; s >>= 1) {            \
            rho_i += shfl_down(rho_i, s, t_per_atom);                       \
        }                                                                   \
        for (int k = 0; k < 3; k++) {                                       \
            for (unsigned int s = t_per_atom / 2; s > 0; s >>= 1) {         \
                mu_i[k] += shfl_down(mu_i[k], s, t_per_atom);               \
           }                                                                \
        }                                                                   \
        for (int k = 0; k < 6; k++) {                                       \
            for (unsigned int s = t_per_atom / 2; s > 0; s >>= 1) {         \
                lambda_i[k] += shfl_down(lambda_i[k], s, t_per_atom);       \
           }                                                                \
        }                                                                   \
    }                                                                       \
    if (offset == 0 && ii < inum) {                                         \
        acctyp v_i = lambda_i[0] + lambda_i[1] + lambda_i[2];               \
        acctyp sum_mu = 0.0, sum_lamb = 0.0;                                \
        adp_rho[i] = rho_i;                                                 \
        for(int k = 0; k < 6; k++) {                                        \
            if(k < 3) {                                                     \
                sum_mu += mu_i[k] * mu_i[k];                                \
                sum_lamb += lambda_i[k] * lambda_i[k];                      \
                adp_mu[i + k * nall] = mu_i[k];                             \
            }                                                               \
            adp_lambda[i + k * nall] = lambda_i[k];                         \
        }                                                                   \
        sum_lamb += 2.0 * (pow(lambda_i[3], 2) + pow(lambda_i[4], 2) +      \
                    pow(lambda_i[5], 2));                                   \
        energy = c1F * ucl_sqrt(rho_i) + c2F * pow(rho_i, 2) +              \
                 0.5 * sum_mu + 0.5 * sum_lamb - 1.0 / 6.0 * v_i * v_i;     \
        if(EVFLAG && eflag)                                                 \
            engv[ii] =  energy;                                             \
    }

#define local_allocate_store_answer_anna_adp()

#define store_answer_anna_adp(f, energy, virial, ii, inum,                  \
                              tid, e_base, t_per_atom,                      \
                              offset, eflag, vflag, ans, engv)              \
    if(t_per_atom > 1) {                                                    \
        for (unsigned int s = t_per_atom / 2; s > 0; s >>= 1) {             \
            f.x += shfl_down(f.x, s, t_per_atom);                           \
            f.y += shfl_down(f.y, s, t_per_atom);                           \
            f.z += shfl_down(f.z, s, t_per_atom);                           \
            if (EVFLAG)                                                     \
                energy += shfl_down(energy, s, t_per_atom);                 \
        }                                                                   \
        if (EVFLAG && vflag) {                                              \
            for (unsigned int s = t_per_atom / 2; s > 0; s >>= 1) {         \
                for (int n = 0; n < 6; n++)                                 \
                    virial[n] += shfl_down(virial[n], s, t_per_atom);       \
            }                                                               \
        }                                                                   \
    }                                                                       \
    if (offset == 0 && ii < inum) {                                         \
        int ei = ii;                                                        \
        if (EVFLAG && eflag) {                                              \
            engv[ei] += 0.5 * energy + e_base;                              \
            ei += inum;                                                     \
        }                                                                   \
        if (EVFLAG && vflag) {                                              \
            for (int n = 0; n < 6; n++) {                                   \
                engv[ei] = virial[n] * (acctyp)0.5;                         \
                ei += inum;                                                 \
            }                                                               \
        }                                                                   \
        ans[ii] = f;                                                        \
    }

#endif

//---------------------------------------------------------------------
	                // get the short neighbor list
//----------------------------------------------------------------------
__kernel void k_anna_adp_short_nbor(const __global numtyp4* restrict x_, 
									const numtyp cutMax, const int ntypes, 
									__global int* dev_nbor, 
									const int nbor_pitch, 
									__global int* dev_packed, 
									const int inum, const int t_per_atom) {

	int tid, ii, offset, n_stride;
	atom_info(t_per_atom, ii, tid, offset);

	local_allocate_acc_numj();
	if (ii < inum) {
		int i, nbor_j, nbor_end, jnum;																
		nbor_info(dev_nbor, dev_packed, nbor_pitch, t_per_atom, ii, 
				  offset, i, jnum, n_stride, nbor_end, nbor_j);

		numtyp4 ix; fetch4(ix, i, pos_tex);
		int nbor_begin = nbor_j;																			
		int index_in = 0;																			
		int index_ou = 0;
		int j_out[300], j_in[300];																	
		
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
			} else {
				j_out[index_ou] = sj_nomask;
				index_ou++;
			}
		}																								
		int in_out[40];																				
		acc_numj(dev_nbor, in_out, ii, nbor_pitch, index_in, 
				 index_ou, tid, t_per_atom, offset);												

		int sum_in = 0, sum_ou = dev_nbor[ii + nbor_pitch];
		nbor_begin -= offset;																		
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
	 /* __kernel extern "C" __global__, in "ucl_nv_kernel.h" file
	    embed energy, and angular energy will be calculated here*/
//----------------------------------------------------------------------
__kernel void k_energy(const __global numtyp4* restrict x_, const __global int* restrict map,
					   const int inum, const int t_per_atom, const __global int* dev_nbor, 
					   const __global int* dev_packed, const int nbor_pitch, const int nall, 
					   const int ntl, const int nhl, const int nnod, const int npsf, 
					   const int ntsf, const int nsf, const int nout, const __global int* flagact, 
					   const int eflag, const int vflag, const numtyp2 adp_const,
					   const __global numtyp* restrict weight_all, 
					   const __global numtyp* restrict bias_all, 
					   const __global numtyp* gadp_params, __global numtyp* restrict adp_rho, 
					   __global numtyp* restrict adp_mu, __global numtyp* restrict adp_lambda, 
					   __global numtyp* restrict ladp_params, __global acctyp* restrict engv) {

	local_allocate_acc_Gi();
	local_allocate_acc_hide();
	local_allocate_store_energy_padp();																

	numtyp cutMax = adp_const.y;
	numtyp Rc = ucl_sqrt(cutMax);
	numtyp coeff_fc = MY_PI / Rc;																	
	acctyp energy = 0.0;
	numtyp A0 = gadp_params[0];
	numtyp yy = gadp_params[1];
	numtyp gamma = gadp_params[2];
	numtyp C0 = gadp_params[3];
	numtyp c1F = gadp_params[4];
	numtyp c2F = gadp_params[5];
	numtyp r0 = gadp_params[10];
	numtyp hc = gadp_params[12];
	numtyp d1 = gadp_params[13];
	numtyp q1 = gadp_params[14];
	numtyp d3 = gadp_params[15];
	numtyp q3 = gadp_params[16];	
	
	//------------------starting calculation-------------------
	int tid, ii, offset, n_stride;																	
	atom_info(t_per_atom, ii, tid, offset);															
	
	if (ii < inum) {
		int i, nbor_j0, nbor_j, nbor_end, jnum;
		nbor_info(dev_nbor, dev_packed, nbor_pitch, t_per_atom, ii, offset, i, jnum, n_stride, nbor_end, nbor_j0);			
		numtyp4 ix; fetch4(ix, i, pos_tex);
		numtyp G_i[28] = { 0.0 };

		//-----------------all neighbors-----------------
		int idj = offset;
		for (nbor_j = nbor_j0; nbor_j < nbor_end; nbor_j += n_stride, idj += t_per_atom) {
			int j = dev_packed[nbor_j];																
			j &= NEIGHMASK;

			numtyp4 jx, tx; fetch4(jx, j, pos_tex);													
			numtyp deltx = ix.x - jx.x;
			numtyp delty = ix.y - jx.y;
			numtyp deltz = ix.z - jx.z;
			numtyp r2ij = deltx * deltx + delty * delty + deltz * deltz;
			numtyp rij = ucl_sqrt(r2ij);

			//-----------------pair symmetry function-----------------
			numtyp xij = 2.0 * rij / Rc - 1.0;
			numtyp fcij = 0.5 * (cos(rij * coeff_fc) + 1.0);
			tx.x = (numtyp)1.0;
			tx.y = xij;
			G_i[0] += fcij;
			G_i[1] += fcij * tx.y;
			for (int m = 2; m < npsf; m++) {
				tx.z = 2.0 * xij * tx.y - tx.x;
				tx.x = tx.y;
				tx.y = tx.z;
				G_i[m] += fcij * tx.z;
			}

			//-----------------triple symmetry function-----------------
			int  nbor_k, nbor_kend, k_loop;
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
				for (; nbor_k < nbor_kend; nbor_k++) {
					if (nbor_k == nbor_j)	continue;
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
					
					numtyp fcik = 0.5 * (cos(rik * coeff_fc) + 1.0);
					numtyp xik = 0.5 * (cos_theta + 1.0);
					numtyp fcijk = fcij * fcik;
					tx.x = (numtyp)1.0;
					tx.y = xik;
					G_i[npsf] += fcijk;
					G_i[npsf + 1] += fcijk * tx.y;
					for (int m = 2; m < ntsf; m++) {
						tx.z = 2.0 * xik * tx.y - tx.x;
						tx.x = tx.y;
						tx.y = tx.z;
						G_i[npsf + m] += fcijk * tx.z;
					}	// G_triple
				}	// neigbor_k
			}	// k_loop
		}	// j_loop
		acc_Gi(G_i, nsf, tid, t_per_atom, offset);													

		//----------------- feedforward for local parameters -----------------
		int id_w = 0, n_row[3], n_col[3], w_begin[3], b_begin[3];
		acctyp weight, bias;
		acctyp hid[6] = { 0.0 };																	
		
		n_row[0] = nnod; n_col[0] = nsf;
		n_row[1] = nnod; n_col[1] = nnod;
		n_row[2] = nout; n_col[2] = nnod;
		b_begin[0] = 0;	 	
		b_begin[1] = n_row[0];
		b_begin[2] = b_begin[1] + n_row[1];
		w_begin[0] = 0;	
		w_begin[1] = n_row[0] * n_col[0];
		w_begin[2] = w_begin[1] + n_row[1] * n_col[1];

		for (int n = 0; n < ntl - 1; n++) {
			int row_loop = 1 + n_row[n] / t_per_atom;
			if(n_row[n] % t_per_atom == 0)
				row_loop -=1;
			int flag_act = flagact[n];
			numtyp t_hid[6] = { 0.0 };																

			for(int k = 0; k < row_loop; k++) {														
				int id = offset + k * t_per_atom;
				if(id < n_row[n]) {																	
					fetch(bias, id + b_begin[n], bias_tex);
					id_w = w_begin[n] + id * n_col[n];
					if(n == 0) {
						for(int m = 0; m < n_col[n]; m++) {
							//fetch(weight, id_w, weight_tex);										
							weight = weight_all[id_w];
							t_hid[id] += weight * G_i[m];
							id_w++;
						}
					} else {
						for(int m = 0; m < n_col[n]; m++) {
							//fetch(weight, id_w, weight_tex);										
							weight = weight_all[id_w];
							t_hid[id] += weight * hid[m];
							id_w++;
						}
					}
					t_hid[id] += bias;
					if(flag_act == 0)
						t_hid[id] = t_hid[id];
					if(flag_act == 4) {																
						numtyp exp_2x = ucl_exp(coeff_b * t_hid[id]);
						t_hid[id] = coeff_a * ((exp_2x - 1.0) / (exp_2x + 1.0));
					}
				}
			}
			acc_hide(hid, t_hid, n_row[n], tid, t_per_atom, offset);								
		}
		
		//----------------- energy calculation -----------------
		if(nout < t_per_atom && offset < nout) {
			ladp_params[i + offset * nall] = hid[offset];
		}
		acctyp d2 = hid[0];
		acctyp q2 = hid[1];																			

		acctyp rho_i = 0.0, mu_i[3] = { 0.0 }, lambda_i[6] = { 0.0 };								
		for (nbor_j = nbor_j0; nbor_j < nbor_end; nbor_j += n_stride) {
			int j = dev_packed[nbor_j];																
			j &= NEIGHMASK;
			numtyp4 jx; fetch4(jx, j, pos_tex);														
			numtyp deltx = ix.x - jx.x;
			numtyp delty = ix.y - jx.y;
			numtyp deltz = ix.z - jx.z;
			numtyp r2ij = deltx * deltx + delty * delty + deltz * deltz;
			numtyp rij = ucl_sqrt(r2ij);

			numtyp stpf_x = (rij - Rc) / hc;
			numtyp stpf_x4 = stpf_x * stpf_x * stpf_x * stpf_x;
			numtyp adp_stpf = stpf_x4 / (1.0 + stpf_x4);
			numtyp adp_u = adp_stpf * (d1 * exp(-d2 * rij) + d3);
			numtyp adp_w = adp_stpf * (q1 * exp(-q2 * rij) + q3);

			mu_i[0] += adp_u * deltx;
			mu_i[1] += adp_u * delty;
			mu_i[2] += adp_u * deltz;

			lambda_i[0] += adp_w * deltx * deltx;
			lambda_i[1] += adp_w * delty * delty;
			lambda_i[2] += adp_w * deltz * deltz;
			lambda_i[3] += adp_w * deltx * delty;
			lambda_i[4] += adp_w * deltx * deltz;
			lambda_i[5] += adp_w * delty * deltz;

			numtyp rho_z = rij - r0;
			numtyp exp_z = exp(-gamma * rho_z);
			rho_i += adp_stpf * (A0 * pow(rho_z, yy) * exp_z * (1 + exp_z) + C0);
		}
		store_energy_padp(rho_i, mu_i, lambda_i, c1F, c2F, adp_rho, adp_mu, adp_lambda, 
						  energy, ii, i, nall, inum, tid, t_per_atom, offset, eflag, vflag, engv);
	}	// if ii
}

//----------------------------------------------------------------------
			// force of atom i and energy of pair part
//----------------------------------------------------------------------
__kernel void k_anna_adp(const __global numtyp4* restrict x_, const int ntypes, 
						 const int t_per_atom, const __global int* dev_nbor,
						 const __global int* dev_packed, const int nbor_pitch,
						 __global acctyp3 *ans, __global acctyp* restrict engv,
						 const int eflag, const int vflag, const int inum,
						 const __global numtyp* gadp_params, const numtyp2 adp_const,
						 const int nall, const __global numtyp* restrict adp_rho,
						 const __global numtyp* restrict adp_mu, 
						 const __global numtyp* restrict adp_lambda,
						 const __global numtyp* restrict ladp_params) {
	
	local_allocate_store_answer_anna_adp();
	numtyp e_base = adp_const.x;																
	numtyp cutMax = adp_const.y;
	numtyp Rc = ucl_sqrt(cutMax);
	numtyp A0 = gadp_params[0];
	numtyp yy = gadp_params[1];
	numtyp gamma = gadp_params[2];
	numtyp C0 = gadp_params[3];
	numtyp c1F = 0.5 * gadp_params[4];
	numtyp c2F = 2.0 * gadp_params[5];
	numtyp V0 = gadp_params[6];
	numtyp b1 = gadp_params[7];
	numtyp b2 = gadp_params[8];
	numtyp delta = gadp_params[9];
	numtyp r0 = gadp_params[10];
	numtyp r1 = gadp_params[11];
	numtyp hc = gadp_params[12];
	numtyp d1 = gadp_params[13];
	numtyp q1 = gadp_params[14];
	numtyp d3 = gadp_params[15];
	numtyp q3 = gadp_params[16];	

	acctyp3 f;	f.x = (acctyp)0.0; f.y = (acctyp)0.0; f.z = (acctyp)0.0;
	acctyp fx, fy, fz, fxi, fyi, fzi, fxj, fyj, fzj;
	acctyp energy = (acctyp)0.0;
	acctyp virial[6] = { 0.0 };																		

	//------------------ starting calculation -------------------
	int tid, ii, offset, n_stride;
	atom_info(t_per_atom, ii, tid, offset);															
	numtyp rho_i, rho_j, mu_i[3], mu_j[3], lambda_i[6], lambda_j[6];
	numtyp d2_i, q2_i;
	numtyp d2_j, q2_j;

	if (ii < inum) {
		int i, nbor_j, nbor_end, jnum;
		nbor_info(dev_nbor, dev_packed, nbor_pitch, t_per_atom, ii, offset, i, jnum, n_stride, nbor_end, nbor_j);		
		
		numtyp4 ix;		fetch4(ix, i, pos_tex);
		fetch(rho_i, i, rho_tex);	
		fetch(d2_i, i, ladp_tex);
		fetch(q2_i, i + nall, ladp_tex);

		for(int k = 0; k < 6; k++) {
			if(k < 3)
				fetch(mu_i[k], i + k * nall, imu_tex);
			fetch(lambda_i[k], i + k * nall, lambda_tex);
		}
		numtyp v_i = -1.0 / 3.0 * (lambda_i[0] + lambda_i[1] + lambda_i[2]);
		numtyp rep_coeff = V0 / (b2 - b1);
		numtyp x, y, z, r2ij, rij;

		// all neighbors
		for (; nbor_j < nbor_end; nbor_j += n_stride) {
			int j = dev_packed[nbor_j];															
			j &= NEIGHMASK;

			fetch(rho_j, j, rho_tex);		
			fetch(d2_j, j, ladp_tex);
			fetch(q2_j, j + nall, ladp_tex);

			for(int k = 0; k < 6; k++) {
				if(k < 3)
					fetch(mu_j[k], j + k * nall, imu_tex);
				fetch(lambda_j[k], j + k * nall, lambda_tex);
			}
			numtyp v_j = -1.0 / 3.0 * (lambda_j[0] + lambda_j[1] + lambda_j[2]);
			
			numtyp4 jx; fetch4(jx, j, pos_tex);														
			x = ix.x - jx.x;
			y = ix.y - jx.y;
			z = ix.z - jx.z;
			r2ij = x * x + y * y + z * z;
			rij = ucl_sqrt(r2ij);

			numtyp stpf_x = (rij - Rc) / hc;
			numtyp stpf_x3 = stpf_x * stpf_x * stpf_x;
			numtyp stpf_t1 = 1.0 + stpf_x * stpf_x3;
			numtyp stpf = stpf_x * stpf_x3 / stpf_t1;
			numtyp dstpf = 4.0 * stpf_x3 / stpf_t1 / stpf_t1 /hc;

            numtyp rho_z = rij - r0;
			numtyp exp_z = exp(-gamma * rho_z);
			numtyp z_yy = A0 * pow(rho_z, yy);
			numtyp ga_zyy = z_yy * gamma;
			numtyp drho = exp_z * (1.0 + exp_z) * (z_yy * (dstpf + stpf * yy / rho_z) - ga_zyy) + C0 * dstpf - ga_zyy * exp_z * exp_z;
			
			numtyp dfp_i = (c1F * pow(rho_i, -0.5) + c2F * rho_i) * drho;
			numtyp dfp_j = (c1F * pow(rho_j, -0.5) + c2F * rho_j) * drho;
				
			numtyp repul_z = rij / r1;
			numtyp zb1 = pow(repul_z, b1);
			numtyp zb2 = pow(repul_z, b2);
			numtyp drep_t = b2 * b1 / r1;
			numtyp rep_t1 = rep_coeff * (b2 / zb1 - b1 / zb2) + delta;
			numtyp drep = dstpf * rep_t1 + stpf * rep_coeff * (drep_t / repul_z * (-1.0 / zb1 + 1.0 / zb2));

			numtyp u_t_i = d1 * exp(-d2_i * rij);
			numtyp w_t_i = q1 * exp(-q2_i * rij);
			numtyp u_t_j = d1 * exp(-d2_j * rij);
			numtyp w_t_j = q1 * exp(-q2_j * rij);
			numtyp u_i = stpf * (u_t_i + d3);
			numtyp w_i = 2.0 * stpf * (w_t_i + q3);
			numtyp u_j = stpf * (u_t_j + d3);
			numtyp w_j = 2.0 * stpf * (w_t_j + q3);

			numtyp du_i = dstpf * (u_t_i + d3) - stpf * d2_i * u_t_i;
			numtyp dw_i = dstpf * (w_t_i + q3) - stpf * q2_i * w_t_i;
			numtyp du_j = dstpf * (u_t_j + d3) - stpf * d2_j * u_t_j;
			numtyp dw_j = dstpf * (w_t_j + q3) - stpf * q2_j * w_t_j;

			numtyp x2 = x * x, y2 = y * y, z2 = z * z, xy = x * y, xz = x * z, yz = y * z;
			numtyp dang_lamb1_i = dw_i * (lambda_i[0] * x2 + lambda_i[1] * y2 + lambda_i[2] * z2);
			numtyp dang_lamb2_i = dw_i * (lambda_i[3] * xy + lambda_i[4] * xz + lambda_i[5] * yz) * 2.0 + dang_lamb1_i;
			numtyp dang_lamb1_j = dw_j * (lambda_j[0] * x2 + lambda_j[1] * y2 + lambda_j[2] * z2);
			numtyp dang_lamb2_j = dw_j * (lambda_j[3] * xy + lambda_j[4] * xz + lambda_j[5] * yz) * 2.0 + dang_lamb1_j;
			numtyp df_t1_i = 0.5 * drep + dfp_i + du_i * (mu_i[0] * x + mu_i[1] * y + mu_i[2] * z) + dang_lamb2_i;
			numtyp df_t1_j = 0.5 * drep + dfp_j - du_j * (mu_j[0] * x + mu_j[1] * y + mu_j[2] * z) + dang_lamb2_j;
            numtyp df_t2_i = v_i * (dw_i * rij + w_i);
            numtyp df_t2_j = v_j * (dw_j * rij + w_j);

			// force for: atom i is the central atom
            fxi = df_t1_i * x / rij + w_i * (y * lambda_i[3] + z * lambda_i[4] + x * lambda_i[0]) + mu_i[0] * u_i + x * df_t2_i;
            fyi = df_t1_i * y / rij + w_i * (y * lambda_i[1] + z * lambda_i[5] + x * lambda_i[3]) + mu_i[1] * u_i + y * df_t2_i;
            fzi = df_t1_i * z / rij + w_i * (y * lambda_i[5] + z * lambda_i[2] + x * lambda_i[4]) + mu_i[2] * u_i + z * df_t2_i;
	
			// force for: atom j is the central atom
			fxj = -df_t1_j * x / rij - w_j * (y * lambda_j[3] + z * lambda_j[4] + x * lambda_j[0]) + mu_i[0] * u_j - x * df_t2_j;
            fyj = -df_t1_j * y / rij - w_j * (y * lambda_j[1] + z * lambda_j[5] + x * lambda_j[3]) + mu_i[1] * u_j - y * df_t2_j;
            fzj = -df_t1_j * z / rij - w_j * (y * lambda_j[5] + z * lambda_j[2] + x * lambda_j[4]) + mu_i[2] * u_j - z * df_t2_j;
			fx = fxj - fxi;
			fy = fyj - fyi;
			fz = fzj - fzi;

			f.x += fx;
			f.y += fy;
			f.z += fz;
			if (EVFLAG && eflag) {
				energy += stpf * rep_t1;
			}
			if (EVFLAG && vflag) {
				virial[0] += x * fx;
				virial[1] += y * fy;
				virial[2] += z * fz;
				virial[3] += x * fy;
				virial[4] += x * fz;
				virial[5] += y * fz;
			}
		}
		store_answer_anna_adp(f, energy, virial, ii, inum, tid, e_base, t_per_atom, offset, eflag, vflag, ans, engv);
	}
}
