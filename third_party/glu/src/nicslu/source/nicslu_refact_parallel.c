/*re-factorize another matrix with the same structure, in parallel, without partial pivoting*/
/*last modified: june 11, 2013*/
/*author: Chen, Xiaoming*/

#include "nicslu.h"
#include "nicslu_internal.h"
#include "timer_c.h"
#include "math.h"
#ifdef SSE2
#include <emmintrin.h>
#endif

extern size_t g_si, g_sd, g_sp;

void _I_NicsLU_ReFactorize_Cluster(SNicsLU *nicslu, SNicsLUThreadArg *tharg, unsigned int id)
{
	real__t *ax;
	uint__t *ai, *ap;
	uint__t cstart, cend, oldrow;
	uint__t *rowperm;
	uint__t i, ii, k, p, j;
	uint__t *data;
	uint__t start, end;
	int__t *pinv;
	void *lu_array;
	size_t *up;
	uint__t *llen, *ulen;
	uint__t lcol;
	real__t *l_row_data_i;
	uint__t *l_row_index_i;
	uint__t l_row_len_i;
	real__t *u_row_data_j;
	uint__t *u_row_index_j;
	uint__t u_row_len_j;
	real__t *x;
	real__t xj;
	real__t *ldiag;
	bool__t *fi;
	uint__t scale;
	real__t *cscale;
#ifdef SSE2
	__m128d _xj, _prod, _tval;
	uint__t _end;
#endif

	ax = nicslu->ax;
	ai = nicslu->ai;
	ap = nicslu->ap;
	rowperm = nicslu->row_perm;
	data = nicslu->aeg_refact_data;
	pinv = nicslu->pivot_inv;
	lu_array = nicslu->lu_array;
	up = nicslu->up;
	llen = nicslu->llen;
	ulen = nicslu->ulen;
	x = (real__t *)(nicslu->workspace_mt1[id]);
	ldiag = nicslu->ldiag;
	fi = (bool__t *)(nicslu->row_state);
	scale = nicslu->cfgi[2];
	cscale = nicslu->cscale;

	/*begin*/
	start = nicslu->cluster_start[id];
	end = nicslu->cluster_end[id];

	for (ii=start; ii<end; ++ii)
	{
		i = data[ii];
		oldrow = rowperm[i];
		cstart = ap[oldrow];
		cend = ap[oldrow+1];

		/*numeric*/
		if (scale == 1 || scale == 2)
		{
			for (k=cstart; k<cend; ++k)
			{
				j = ai[k];
				x[pinv[j]] = ax[k] / cscale[j];
			}
		}
		else
		{
			for (k=cstart; k<cend; ++k)
			{
				x[pinv[ai[k]]] = ax[k];
			}
		}

		u_row_len_j = ulen[i];
		l_row_len_i = llen[i];
		l_row_index_i = (uint__t *)(((byte__t *)lu_array) + up[i] + u_row_len_j*g_sp);
		l_row_data_i = (real__t *)(l_row_index_i + l_row_len_i);

		for (k=0; k<l_row_len_i; ++k)
		{
			lcol = l_row_index_i[k];
			u_row_len_j = ulen[lcol];
			u_row_index_j = (uint__t *)(((byte__t *)lu_array) + up[lcol]);
			u_row_data_j = (real__t *)(u_row_index_j + u_row_len_j);

			xj = x[lcol];
#ifndef SSE2
			for (p=0; p<u_row_len_j; ++p)
			{
				x[u_row_index_j[p]] -= xj * u_row_data_j[p];
			}
#else
			_xj = _mm_load1_pd(&xj);
			_end = (u_row_len_j&(uint__t)1)>0 ? u_row_len_j-1 : u_row_len_j;
			for (p=0; p<_end; p+=2)
			{
				_tval = _mm_loadu_pd(&(u_row_data_j[p]));
				_prod = _mm_mul_pd(_xj, _tval);
				_tval = _mm_load_sd(&(x[u_row_index_j[p]]));
				_tval = _mm_loadh_pd(_tval, &(x[u_row_index_j[p+1]]));
				_tval = _mm_sub_pd(_tval, _prod);
				_mm_storel_pd(&(x[u_row_index_j[p]]), _tval);
				_mm_storeh_pd(&(x[u_row_index_j[p+1]]), _tval);
			}
			if ((u_row_len_j&(uint__t)1) > 0)
			{
				x[u_row_index_j[_end]] -= xj * u_row_data_j[_end];
			}
#endif
		}

		u_row_len_j = ulen[i];
		u_row_index_j = (uint__t *)(((byte__t *)lu_array) + up[i]);
		u_row_data_j = (real__t *)(u_row_index_j + u_row_len_j);

		xj = x[i];
		x[i] = 0.;

		/*check the diag*/
		if (xj == 0.)
		{
			tharg->err = NICSLU_MATRIX_NUMERIC_SINGULAR;
			nicslu->thread_finish[id] = TRUE;
			return;
		}
		if (isNaN(xj))
		{
			tharg->err = NICSLU_NUMERIC_OVERFLOW;
			nicslu->thread_finish[id] = TRUE;
			return;
		}

		/*put data into L and U*/
		for (k=0; k<u_row_len_j; ++k)
		{
			lcol = u_row_index_j[k];
			u_row_data_j[k] = x[lcol] / xj;
			x[lcol] = 0.;
		}

		for (k=0; k<l_row_len_i; ++k)
		{
			lcol = l_row_index_i[k];
			l_row_data_i[k] = x[lcol];
			x[lcol] = 0.;
		}

		ldiag[i] = xj;

		fi[i] = TRUE;
	}

	tharg->err = NICS_OK;
	nicslu->thread_finish[id] = TRUE;
}

void _I_NicsLU_ReFactorize_Pipeline(SNicsLU *nicslu, SNicsLUThreadArg *tharg, unsigned int id)
{
	real__t *ax;
	uint__t *ai, *ap;
	uint__t cstart, cend, oldrow;
	uint__t *rowperm;
	uint__t i, ii, k, p, j;
	uint__t *data;
	uint__t start, end;
	int__t *pinv;
	void *lu_array;
	size_t *up;
	uint__t *llen, *ulen;
	uint__t lcol;
	real__t *l_row_data_i;
	uint__t *l_row_index_i;
	uint__t l_row_len_i;
	real__t *u_row_data_j;
	uint__t *u_row_index_j;
	uint__t u_row_len_j;
	real__t *x;
	real__t xj;
	real__t *ldiag;
	bool__t *fi;
	unsigned int threads;
	SNicsLUThreadArg *arg;
	volatile bool__t *wait;
	uint__t scale;
	real__t *cscale;
#ifdef SSE2
	__m128d _xj, _prod, _tval;
	uint__t _end;
#endif

	ax = nicslu->ax;
	ai = nicslu->ai;
	ap = nicslu->ap;
	rowperm = nicslu->row_perm;
	data = nicslu->aeg_refact_data;
	pinv = nicslu->pivot_inv;
	lu_array = nicslu->lu_array;
	up = nicslu->up;
	llen = nicslu->llen;
	ulen = nicslu->ulen;
	x = (real__t *)(nicslu->workspace_mt1[id]);
	ldiag = nicslu->ldiag;
	fi = (bool__t *)(nicslu->row_state);
	threads = (unsigned int)(nicslu->cfgi[7]);
	arg = (SNicsLUThreadArg *)(nicslu->thread_arg);
	scale = nicslu->cfgi[2];
	cscale = nicslu->cscale;

	/*begin*/
	start = nicslu->pipeline_start;
	end = nicslu->pipeline_end;

	for (ii=start+id; ii<end; ii+=threads)
	{
		i = data[ii];
		oldrow = rowperm[i];
		cstart = ap[oldrow];
		cend = ap[oldrow+1];

		/*numeric*/
		if (scale == 1 || scale == 2)
		{
			for (k=cstart; k<cend; ++k)
			{
				j = ai[k];
				x[pinv[j]] = ax[k] / cscale[j];
			}
		}
		else
		{
			for (k=cstart; k<cend; ++k)
			{
				x[pinv[ai[k]]] = ax[k];
			}
		}

		l_row_len_i = llen[i];
		u_row_len_j = ulen[i];
		l_row_index_i = (uint__t *)(((byte__t *)lu_array) + up[i] + u_row_len_j*g_sp);
		l_row_data_i = (real__t *)(l_row_index_i + l_row_len_i);

		for (k=0; k<l_row_len_i; ++k)
		{
			lcol = l_row_index_i[k];
			if (ulen[lcol] == 0) continue;/*bug fix*/
			
			/*here!!! wait!!!*/
		/*	_SpinWaitChar((volatile char *)&(fi[lcol]));*/
			wait = (volatile bool__t *)&(fi[lcol]);
			while (!(*wait))
			{
				unsigned int ti;
				for (ti=0; ti<threads; ++ti)
				{
					if (FAIL(arg[ti].err))
					{
						tharg->err = NICS_OK;
						nicslu->thread_finish[id] = TRUE;
						return;
					}
				}
			}

			u_row_len_j = ulen[lcol];
			u_row_index_j = (uint__t *)(((byte__t *)lu_array) + up[lcol]);
			u_row_data_j = (real__t *)(u_row_index_j + u_row_len_j);

			xj = x[lcol];
#ifndef SSE2
			for (p=0; p<u_row_len_j; ++p)
			{
				x[u_row_index_j[p]] -= xj * u_row_data_j[p];
			}
#else
			_xj = _mm_load1_pd(&xj);
			_end = (u_row_len_j&(uint__t)1)>0 ? u_row_len_j-1 : u_row_len_j;
			for (p=0; p<_end; p+=2)
			{
				_tval = _mm_loadu_pd(&(u_row_data_j[p]));
				_prod = _mm_mul_pd(_xj, _tval);
				_tval = _mm_load_sd(&(x[u_row_index_j[p]]));
				_tval = _mm_loadh_pd(_tval, &(x[u_row_index_j[p+1]]));
				_tval = _mm_sub_pd(_tval, _prod);
				_mm_storel_pd(&(x[u_row_index_j[p]]), _tval);
				_mm_storeh_pd(&(x[u_row_index_j[p+1]]), _tval);
			}
			if ((u_row_len_j&(uint__t)1) > 0)
			{
				x[u_row_index_j[_end]] -= xj * u_row_data_j[_end];
			}
#endif
		}

		u_row_len_j = ulen[i];
		u_row_index_j = (uint__t *)(((byte__t *)lu_array) + up[i]);
		u_row_data_j = (real__t *)(u_row_index_j + u_row_len_j);

		xj = x[i];
		x[i] = 0.;

		/*check the diag*/
		if (xj == 0.)
		{
			tharg->err = NICSLU_MATRIX_NUMERIC_SINGULAR;
			nicslu->thread_finish[id] = TRUE;
			return;
		}
		if (isNaN(xj))
		{
			tharg->err = NICSLU_NUMERIC_OVERFLOW;
			nicslu->thread_finish[id] = TRUE;
			return;
		}

		/*put data into L and U*/
		for (k=0; k<u_row_len_j; ++k)
		{
			lcol = u_row_index_j[k];
			u_row_data_j[k] = x[lcol] / xj;
			x[lcol] = 0.;
		}

		for (k=0; k<l_row_len_i; ++k)
		{
			lcol = l_row_index_i[k];
			l_row_data_i[k] = x[lcol];
			x[lcol] = 0.;
		}

		ldiag[i] = xj;

		fi[i] = TRUE;
	}

	tharg->err = NICS_OK;
	nicslu->thread_finish[id] = TRUE;
}

int NicsLU_ReFactorize_MT(SNicsLU *nicslu, real__t *ax0)
{
	uint__t n, i, j;
	uint__t thres;
	uint__t level;
	uint__t *data, *head;
	bool__t *thac, *thfi;
	uint__t *start, *end;
	uint__t clv, lv, len, lstart, lend, aegh, avg, /*avga1, */res;
	SNicsLUThreadArg *arg;
	int err, mode;
	uint__t *ulen;
	uint__t total, sub;
	unsigned int th, maxth, ti;
	real__t bal;

	/*check flags*/
	if (NULL == nicslu || NULL == ax0)
	{
		return NICSLU_ARGUMENT_ERROR;
	}
	if (!nicslu->flag[2])
	{
		return NICSLU_MATRIX_NOT_FACTORIZED;
	}
	if (!nicslu->flag[3])
	{
		return NICSLU_THREADS_NOT_INITIALIZED;
	}
	if (!nicslu->flag[4])
	{
		return NICSLU_SCHEDULER_NOT_INITIALIZED;
	}

	n = nicslu->n;
	maxth = (unsigned int)(nicslu->cfgi[5]);
	th = (unsigned int)(nicslu->cfgi[7]);
	if (th > maxth || th < 2)
	{
		th = maxth;
		nicslu->cfgi[7] = maxth;
	}
	thres = nicslu->cfgi[3];
	if ((unsigned int)thres < th)
	{
		thres = th;
	/*	nicslu->cfgi[3] = thres;*/
	}
	thac = nicslu->thread_active;
	thfi = nicslu->thread_finish;
	start = nicslu->cluster_start;
	end = nicslu->cluster_end;
	arg = (SNicsLUThreadArg *)(nicslu->thread_arg);
	ulen = nicslu->ulen;
	bal = nicslu->cfgf[4];

	/*begin*/
	TimerStart((STimer *)(nicslu->timer));

	/*create aegraph*/
	_I_NicsLU_CreateAEGraphForRefact(nicslu);
	
	level = nicslu->aeg_refact_level;
	data = nicslu->aeg_refact_data;
	head = nicslu->aeg_refact_header;

	/*mc64_scale*/
	_I_NicsLU_MC64ScaleForRefact(nicslu, ax0);

	/*scale*/
	err = _I_NicsLU_Scale(nicslu);
	if (FAIL(err)) return err;

	/*mark all rows un-finished*/
	memset(nicslu->row_state, 0, sizeof(byte__t)*n);

	/*clear x & return value*/
	for (ti=0; ti<th; ++ti)
	{
		memset(nicslu->workspace_mt1[ti], 0, sizeof(real__t)*n);
		arg[ti].err = NICS_OK;
	}

	/*start parallel re-factorization*/
	clv = 0;
	while (clv < level)
	{
		len = head[clv+1] - head[clv];
		mode = ((len <= thres) ? 1 : 0);

		nicslu->thread_work = ((mode == 0) ? NICSLU_WORK_REFACT_CLUSTER : NICSLU_WORK_REFACT_PIPELINE);
		lstart = clv;

		/*search lend*/
		if (mode == 0)/*cluster*/
		{
			lend = level;
			++clv;
			while (clv < level)
			{
				len = head[clv+1] - head[clv];
				if (len <= thres)
				{
					lend = clv;
					break;
				}
				++clv;
			}

			/*loop cluster*/
			lv = lstart;

			while (lv < lend)
			{
				aegh = head[lv];
				len = head[lv+1];
				total = 0;
				for (j=aegh; j<len; ++j)
				{
					i = data[j];
					total += ulen[i];
				}
				avg = (uint__t)((real__t)total / th * bal);

				j = aegh;
				sub = 0;
				while (j < len)
				{
					i = data[j];
					sub += ulen[i];
					++j;
					if (sub >= avg) break;
				}
				start[0] = aegh;
				end[0] = j;

			/*	aegh = head[lv];
				len = head[lv+1] - aegh;
				avg = len / th;
				avga1 = avg + 1;
				res = len % th;

				start[0] = aegh;
				end[0] = aegh + ((res>0) ? avga1 : avg);*/

				for (ti=1; ti<th; ++ti)
				{
					start[ti] = end[ti-1];

					sub = 0;
					while (j < len)
					{
						res = data[j];
						sub += ulen[res];
						++j;
						if (sub >= avg) break;
					}
					if (ti == th-1) end[ti] = len;
					else end[ti] = j;

				/*	end[ti] = start[ti] + ((ti<res) ? avga1 : avg);*/

					thfi[ti] = FALSE;
					thac[ti] = TRUE;
				}

				_I_NicsLU_ReFactorize_Cluster(nicslu, arg, 0);

				_SpinBarrier(1, th, (volatile char *)thfi);

				/*check the return value*/
				for (ti=0; ti<th; ++ti)
				{
					if (FAIL(arg[ti].err))
					{
						return arg[ti].err;
					}
				}

				++lv;
			}
		}
		else/*pipeline*/
		{
			lend = level;
			++clv;
			while (clv < level)
			{
				len = head[clv+1] - head[clv];
				if (len > thres)
				{
					lend = clv;
					break;
				}
				++clv;
			}

			nicslu->pipeline_start = head[lstart];
			nicslu->pipeline_end = head[lend];

			/*begin pipeline*/
			for (ti=1; ti<th; ++ti)
			{
				thfi[ti] = FALSE;
				thac[ti] = TRUE;
			}

			_I_NicsLU_ReFactorize_Pipeline(nicslu, arg, 0);

			_SpinBarrier(1, th, (volatile char *)thfi);
		}
	}

	/*finish*/
	TimerStop((STimer *)(nicslu->timer));
	nicslu->stat[2] = TimerGetRuntime((STimer *)(nicslu->timer));

	/*check the return value*/
	for (ti=0; ti<th; ++ti)
	{
		err = arg[ti].err;
		if (FAIL(err))
		{
			return err;
		}
	}

	return NICS_OK;
}
