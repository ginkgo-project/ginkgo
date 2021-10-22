/*numeric factorization in parallel, with partial pivoting*/
/*last modified: june 16, 2013*/
/*Author: Chen, Xiaoming*/

/*workspace storage*/
/*|-------|-------|-------|-------|-------|*/
/*uid     uid     udat    lid     ldat     */
/*pruned  unpruned*/

#include "nicslu.h"
#include "nicslu_internal.h"
#include "timer_c.h"
#include "math.h"
#include "thread.h"
#ifdef SSE2
#include <emmintrin.h>
#endif

extern size_t g_sd, g_si, g_sp, g_sis1, g_sps1;

void _I_NicsLU_Factorize_Cluster(SNicsLU *nicslu, SNicsLUThreadArg *tharg, unsigned int id)
{
	real__t *ax;
	uint__t *ai, *ap;
	uint__t cstart, cend, oldrow;
	uint__t *rowperm;
	uint__t n, i, ii, k, j, jnew, q;
	size_t lnnz, unnz;
	uint__t offd;
	real__t *ldiag;
	void **lu;
	void *lurow, *ptr;
	uint__t *llen, *ulen;
	int__t *p, *pinv;
	real__t tol;
	uint__t *u_row_index_j;
	real__t *u_row_data_j;
	uint__t u_row_len_j;
	uint__t scale;
	real__t *cscale;
	real__t *x;
	int__t *flag, *pend, *stack, *appos;
	byte__t *state;
	uint__t *tmpu;
	int__t top;
	uint__t start, end;
	uint__t *aeg;
	uint__t ul, ll;
	uint__t *lrow_index, *urow_index;
	real__t *lrow_data;
	size_t tl, *len_est;
	real__t xj;
	int__t diagcol;
	uint__t pivcol;
	real__t pivot;
	uint__t kb;
	int err;
#ifdef SSE2
	__m128d _xj, _prod, _tval;
	uint__t _end;
#endif

	/*set parameters*/
	ax = nicslu->ax;
	ai = nicslu->ai;
	ap = nicslu->ap;
	rowperm = nicslu->row_perm;
	n = nicslu->n;
	lnnz = 0;
	unnz = 0;
	offd = 0;
	ldiag = nicslu->ldiag;
	lu = nicslu->lu_array2;
	llen = nicslu->llen;
	ulen = nicslu->ulen;
	p = nicslu->pivot;
	pinv = nicslu->pivot_inv;
	tol = nicslu->cfgf[0];
	state = nicslu->row_state;
	aeg = nicslu->aeg_data;
	len_est = nicslu->len_est;
	scale = nicslu->cfgi[2];
	cscale = nicslu->cscale;

	/*|-----|-----|-----|-----|-----|*/
	/*x     flag  stack  appos tmpu*/
	/*local for each thread*/
	/*flag and x are cleared in the main thread*/
	x = (real__t *)(nicslu->workspace_mt1[id]);
	flag = (int__t *)(x + n);
	stack = flag + n;
	appos = stack + n;
	tmpu = (uint__t *)(appos + n);

	/*pend is common*/
	/*it is cleared in the main thread*/
	pend = (int__t *)(nicslu->workspace);

	/*begin*/
	start = nicslu->cluster_start[id];
	end = nicslu->cluster_end[id];

	for (ii=start; ii<end; ++ii)
	{
		i = aeg[ii];
		oldrow = rowperm[i];
		cstart = ap[oldrow];
		cend = ap[oldrow+1];

		/*symbolic*/
		top = _I_NicsLU_Symbolic_Cluster(lu, n, i, pinv, stack, flag, pend, appos, \
			tmpu, ulen, &ai[cstart], cend-cstart);

		/*numeric*/
		if (scale == 1 || scale == 2)
		{
			for (k=cstart; k<cend; ++k)
			{
				j = ai[k];
				x[j] = ax[k] / cscale[j];
			}
		}
		else
		{
			for (k=cstart; k<cend; ++k)
			{
				x[ai[k]] = ax[k];
			}
		}

		for (k=top; k<n; ++k)
		{
			j = stack[k];
			jnew = pinv[j];
			
			/*extract row jnew of U*/
			u_row_len_j = ulen[jnew];
			u_row_index_j = ((uint__t *)lu[jnew]) + u_row_len_j;
			u_row_data_j = (real__t *)(u_row_index_j + u_row_len_j);

			xj = x[j];
#ifndef SSE2
			for (q=0; q<u_row_len_j; ++q)
			{
				x[u_row_index_j[q]] -= xj * u_row_data_j[q];
			}
#else
			_xj = _mm_load1_pd(&xj);
			_end = (u_row_len_j&(uint__t)1)>0 ? u_row_len_j-1 : u_row_len_j;
			for (q=0; q<_end; q+=2)
			{
				_tval = _mm_loadu_pd(&(u_row_data_j[q]));
				_prod = _mm_mul_pd(_xj, _tval);
				_tval = _mm_load_sd(&(x[u_row_index_j[q]]));
				_tval = _mm_loadh_pd(_tval, &(x[u_row_index_j[q+1]]));
				_tval = _mm_sub_pd(_tval, _prod);
				_mm_storel_pd(&(x[u_row_index_j[q]]), _tval);
				_mm_storeh_pd(&(x[u_row_index_j[q+1]]), _tval);
			}
			if ((u_row_len_j&(uint__t)1) > 0)
			{
				x[u_row_index_j[_end]] -= xj * u_row_data_j[_end];
			}
#endif
		}

		ll = llen[i] = n-top;
		ul = ulen[i];

		tl = ul*g_si + (ul+ll)*g_sp;
		if (tl > len_est[i])
		{
			len_est[i] = tl;
			ptr = realloc(lu[i], tl);
			if (ptr == NULL)
			{
				tharg->err = NICSLU_MEMORY_OVERFLOW;
				nicslu->thread_finish[id] = TRUE;
				return;
			}
			lu[i] = ptr;
		}
		lurow = lu[i];

		/*pivoting*/
		diagcol = p[i];
		err = _I_NicsLU_Pivot_Parallel(diagcol, &ulen[i], tmpu, tol, x, \
			&pivcol, &pivot, lurow);
		if (FAIL(err))
		{
			tharg->err = err;
			nicslu->thread_finish[id] = TRUE;
			return;
		}
		--ul;

		/*copy u index*/
		urow_index = (uint__t *)lurow;
		lrow_index = urow_index + ul;
		for (k=0; k<ul; ++k)
		{
			urow_index[k] = lrow_index[k] = tmpu[k];
		}

		lrow_index = (uint__t *)(((byte__t *)lrow_index) + ul*g_sp);
		lrow_data = (real__t *)(lrow_index + ll);

		/*push into L*/
		for (k=top, q=0; k<n; ++k, ++q)
		{
			j = stack[k];
			lrow_index[q] = pinv[j];
			lrow_data[q] = x[j];
			x[j] = 0.;
		}

		ldiag[i] = pivot;

		lnnz += ll;
		unnz += ul;

		/*log the pivoting*/
		if (pivcol != diagcol)/*diagcol = p[i]*/
		{
			++offd;
			if (pinv[diagcol] < 0)
			{
				kb = -pinv[pivcol]-2;
				p[kb] = diagcol;
				pinv[diagcol] = -((int__t)kb)-2;
			}
		}
		p[i] = pivcol;
		pinv[pivcol] = i;/*column pivcol is the ith pivot*/

		/*prune*/
		_I_NicsLU_Prune_Parallel(pend, ll, ulen, pinv, pivcol, lrow_index, lu);

		state[i] = TRUE;

	}

	tharg->lnnz += lnnz;
	tharg->unnz += unnz;
	tharg->offdiag += offd;

	/*finish*/
	tharg->err = NICS_OK;

	nicslu->thread_finish[id] = TRUE;
}

#define CONTINUOUSLY_WAIT

void _I_NicsLU_Factorize_Pipeline(SNicsLU *nicslu, SNicsLUThreadArg *tharg, unsigned int id)
{
	real__t *ax;
	uint__t *ai, *ap;
	uint__t cstart, cend, oldrow;
	uint__t *rowperm;
	uint__t n, i, ii, k, j, jnew, q, l;
	size_t lnnz, unnz;
	uint__t offd;
	real__t *ldiag;
	void **lu;
	void *lurow, *ptr;
	uint__t *llen, *ulen;
	int__t *p, *pinv, tp;
	real__t tol;
	uint__t *u_row_index_j;
	real__t *u_row_data_j;
	uint__t u_row_len_j;
	unsigned int threads;
	uint__t scale;
	real__t *cscale;
	real__t *x;
	int__t *flag, *updated, *pend, *stack, *appos;
	byte__t *state;
	uint__t *tmpu;
	uint__t top;
	uint__t start, end;
	uint__t *aeg;
	uint__t ul, ll;
	uint__t *lrow_index, *urow_index;
	real__t *lrow_data;
	size_t tl, *len_est;
	real__t xj;
	int__t diagcol;
	uint__t pivcol;
	real__t pivot;
	uint__t kb;
	int err;
	int__t head, pos;
	uint__t col, ucol;
	int__t chkflg;
	int__t *pruned;
	int__t *busy;
	volatile byte__t *wait;
	SNicsLUThreadArg *arg;
#ifdef SSE2
	__m128d _xj, _prod, _tval;
	uint__t _end;
#endif

	/*set parameters*/
	ax = nicslu->ax;
	ai = nicslu->ai;
	ap = nicslu->ap;
	rowperm = nicslu->row_perm;
	n = nicslu->n;
	lnnz = 0;
	unnz = 0;
	offd = 0;
	ldiag = nicslu->ldiag;
	lu = nicslu->lu_array2;
	llen = nicslu->llen;
	ulen = nicslu->ulen;
	p = nicslu->pivot;
	pinv = nicslu->pivot_inv;
	tol = nicslu->cfgf[0];
	state = nicslu->row_state;
	aeg = nicslu->aeg_data;
	threads = (unsigned int)(nicslu->cfgi[7]);
	len_est = nicslu->len_est;
	arg = (SNicsLUThreadArg *)(nicslu->thread_arg);
	scale = nicslu->cfgi[2];
	cscale = nicslu->cscale;

	/*|-----|-----|-----|-----|-----|tmpu*/
	/*x     flag  stack  appos */
	/*|-----|-----|-----|*/
	/*updted busy  prud*/
	/*local for each thread*/
	/*flag, updated, busy, and x are cleared in the main thread*/
	x = (real__t *)(nicslu->workspace_mt1[id]);
	flag = (int__t *)(x + n);
	stack = flag + n;
	appos = stack + n;
	tmpu = (uint__t *)(appos + n);

	updated = (int__t *)(nicslu->workspace_mt2[id]);
	busy = updated + n;
	pruned = busy + n;

	/*pend is common*/
	/*it is cleared in the main thread*/
	pend = (int__t *)(nicslu->workspace);

	/*begin*/
	start = nicslu->pipeline_start;
	end = nicslu->pipeline_end;
	
	for (ii=start+id; ii<end; ii+=threads)
	{
		i = aeg[ii];
		oldrow = rowperm[i];
		cstart = ap[oldrow];
		cend = ap[oldrow+1];
		
		/*fetch x*/
		if (scale == 1 || scale == 2)
		{
			for (k=cstart; k<cend; ++k)
			{
				j = ai[k];
				x[j] = ax[k] / cscale[j];
			}
		}
		else
		{
			for (k=cstart; k<cend; ++k)
			{
				x[ai[k]] = ax[k];
			}
		}

		if (ii == 0) goto POST;
		wait = (volatile byte__t *)&state[aeg[ii-1]];
#ifndef CONTINUOUSLY_WAIT
		if (*wait) goto POST;
#endif

		chkflg = i;
		/*pre-symbolic and pre-numeirc*/

#ifdef CONTINUOUSLY_WAIT
		while (!(*wait))
		{
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
#endif

			chkflg += n;
			top = n;

			/*a snapshot of the current states*/
			for (l=nicslu->last_busy; l<ii; ++l)
			{
				busy[aeg[l]] = chkflg;
			}

			for (l=cstart; l<cend; ++l)
			{
				col = ai[l];

				if (flag[col] != chkflg)/*not visited*/
				{
					tp = pinv[col];
					if (tp >= 0)
					{
						if (busy[tp] != chkflg)
						{
							head = 0;
							stack[0] = col;

							while (head >= 0)
							{
								j = stack[head];
								jnew = pinv[j];/*column j is the jnewth pivot column*/

								if (flag[j] != chkflg)
								{
									flag[j] = chkflg;
									if (pend[jnew] < 0)/*unpruned*/
									{
										appos[head] = ulen[jnew];
										pruned[head] = 0;
									}
									else/*pruned*/
									{
										appos[head] = pend[jnew];
										pruned[head] = 1;
									}
								}

								if (pruned[head])
								{
									urow_index = (uint__t *)lu[jnew];
								}
								else
								{
									urow_index = ((uint__t *)lu[jnew]) + ulen[jnew];
								}

								for (pos=--appos[head]; pos>=0; --pos)
								{
									ucol = urow_index[pos];
									if (flag[ucol] != chkflg)
									{
										tp = pinv[ucol];
										if (tp >= 0)
										{
											if (busy[tp] != chkflg)
											{
												appos[head] = pos;
												stack[++head] = ucol;
												break;
											}
											else
											{
												flag[ucol] = chkflg;
											}
										}
										else
										{
											flag[ucol] = chkflg;
										}
									}
								}

								if (pos < 0)
								{
									--head;
#ifdef CONTINUOUSLY_WAIT
									if (updated[j] != i)
#endif
										stack[--top] = j;
								}
							}/*end while*/
						}
						else
						{
							flag[col] = chkflg;
						}
					}
					else
					{
						flag[col] = chkflg;
					}
				}
			}/*end for*/

			for (k=top; k<n; ++k)
			{
				j = stack[k];
				jnew = pinv[j];
				updated[j] = i;
				
				/*extract row jnew of U*/
				u_row_len_j = ulen[jnew];
				u_row_index_j = ((uint__t *)lu[jnew]) + u_row_len_j;
				u_row_data_j = (real__t *)(u_row_index_j + u_row_len_j);

				xj = x[j];
#ifndef SSE2
				for (q=0; q<u_row_len_j; ++q)
				{
					x[u_row_index_j[q]] -= xj * u_row_data_j[q];
				}
#else
				_xj = _mm_load1_pd(&xj);
				_end = (u_row_len_j&(uint__t)1)>0 ? u_row_len_j-1 : u_row_len_j;
				for (q=0; q<_end; q+=2)
				{
					_tval = _mm_loadu_pd(&(u_row_data_j[q]));
					_prod = _mm_mul_pd(_xj, _tval);
					_tval = _mm_load_sd(&(x[u_row_index_j[q]]));
					_tval = _mm_loadh_pd(_tval, &(x[u_row_index_j[q+1]]));
					_tval = _mm_sub_pd(_tval, _prod);
					_mm_storel_pd(&(x[u_row_index_j[q]]), _tval);
					_mm_storeh_pd(&(x[u_row_index_j[q+1]]), _tval);
				}
				if ((u_row_len_j&(uint__t)1) > 0)
				{
					x[u_row_index_j[_end]] -= xj * u_row_data_j[_end];
				}
#endif
			}
#ifdef CONTINUOUSLY_WAIT
		}
#endif

#ifndef CONTINUOUSLY_WAIT
	/*	_SpinWaitChar((volatile char *)wait);*/
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
#endif

POST:

		/*here all the children are finished*/
		/*post-symbolic*/
		top = _I_NicsLU_Symbolic_Pipeline(lu, n, i, pinv, stack, flag, pend, appos, \
			tmpu, ulen, &ai[cstart], cend-cstart, pruned);

		/*post-numeric*/
		for (k=top; k<n; ++k)
		{
			j = stack[k];
			if (updated[j] == i) continue;
			jnew = pinv[j];
			
			/*extract row jnew of U*/
			u_row_len_j = ulen[jnew];
			u_row_index_j = ((uint__t *)lu[jnew]) + u_row_len_j;
			u_row_data_j = (real__t *)(u_row_index_j + u_row_len_j);

			xj = x[j];
#ifndef SSE2
			for (q=0; q<u_row_len_j; ++q)
			{
				x[u_row_index_j[q]] -= xj * u_row_data_j[q];
			}
#else
			_xj = _mm_load1_pd(&xj);
			_end = (u_row_len_j&(uint__t)1)>0 ? u_row_len_j-1 : u_row_len_j;
			for (q=0; q<_end; q+=2)
			{
				_tval = _mm_loadu_pd(&(u_row_data_j[q]));
				_prod = _mm_mul_pd(_xj, _tval);
				_tval = _mm_load_sd(&(x[u_row_index_j[q]]));
				_tval = _mm_loadh_pd(_tval, &(x[u_row_index_j[q+1]]));
				_tval = _mm_sub_pd(_tval, _prod);
				_mm_storel_pd(&(x[u_row_index_j[q]]), _tval);
				_mm_storeh_pd(&(x[u_row_index_j[q+1]]), _tval);
			}
			if ((u_row_len_j&(uint__t)1) > 0)
			{
				x[u_row_index_j[_end]] -= xj * u_row_data_j[_end];
			}
#endif
		}

		ll = llen[i] = n-top;
		ul = ulen[i];

		tl = ul*g_si + (ul+ll)*g_sp;
		if (tl > len_est[i])
		{
			len_est[i] = tl;
			ptr = realloc(lu[i], tl);
			if (ptr == NULL)
			{
				tharg->err = NICSLU_MEMORY_OVERFLOW;
				nicslu->thread_finish[id] = TRUE;
				return;
			}
			lu[i] = ptr;
		}
		lurow = lu[i];

		/*pivoting*/
		diagcol = p[i];
		err = _I_NicsLU_Pivot_Parallel(diagcol, &ulen[i], tmpu, tol, x, \
			&pivcol, &pivot, lurow);
		if (FAIL(err))
		{
			tharg->err = err;
			nicslu->thread_finish[id] = TRUE;
			return;
		}
		--ul;

		/*copy u index*/
		urow_index = (uint__t *)lurow;
		lrow_index = urow_index + ul;
		for (k=0; k<ul; ++k)
		{
			urow_index[k] = lrow_index[k] = tmpu[k];
		}

		lrow_index = (uint__t *)(((byte__t *)lrow_index) + ul*g_sp);
		lrow_data = (real__t *)(lrow_index + ll);

		/*push into L*/
		for (k=top, q=0; k<n; ++k, ++q)
		{
			j = stack[k];
			lrow_index[q] = pinv[j];
			lrow_data[q] = x[j];
			x[j] = 0.;
		}

		ldiag[i] = pivot;

		lnnz += ll;
		unnz += ul;

		/*log the pivoting*/
		if (pivcol != diagcol)/*diagcol = p[i]*/
		{
			++offd;
			if (pinv[diagcol] < 0)
			{
				kb = -pinv[pivcol]-2;
				p[kb] = diagcol;
				pinv[diagcol] = -((int__t)kb)-2;
			}
		}
		p[i] = pivcol;
		pinv[pivcol] = i;/*column pivcol is the ith pivot*/

		/*prune*/
		_I_NicsLU_Prune_Parallel(pend, ll, ulen, pinv, pivcol, lrow_index, lu);

		nicslu->last_busy = ii + 1;
		
		state[i] = TRUE;
	}

	tharg->lnnz += lnnz;
	tharg->unnz += unnz;
	tharg->offdiag += offd;

	/*finish*/
	tharg->err = NICS_OK;

	nicslu->thread_finish[id] = TRUE;
}

/*main scheduler*/
int NicsLU_Factorize_MT(SNicsLU *nicslu)
{
	uint__t n, i, j, t;
	int__t *p, *pinv;
	int__t *pend;
	uint__t thres;
	SNicsLUThreadArg *arg;
	real__t tol;
	uint__t level, clv, len, lstart, lend, lv;
	uint__t *aeg_data, *aeg_head;
	bool__t *thfi;
	bool__t *thac;
	uint__t *start, *end;
	uint__t aegh;
	size_t *wkld, total, sub, avg;
	size_t tl, tu, nnz;
	uint__t offd;
	uint__t *ulen, *llen;
	uint__t ul, ll;
	void *lu, **lu2;
	size_t *up;
	uint__t *uip;
	int err, mode;
	unsigned int th, maxth, ti;
	real__t bal;

	/*check flags*/
	if (NULL == nicslu)
	{
		return NICSLU_ARGUMENT_ERROR;
	}
	nicslu->flag[2] = FALSE;
	if (!nicslu->flag[3])
	{
		return NICSLU_THREADS_NOT_INITIALIZED;
	}
	if (!nicslu->flag[4])
	{
		return NICSLU_SCHEDULER_NOT_INITIALIZED;
	}

	n = nicslu->n;
	p = nicslu->pivot;
	pinv = nicslu->pivot_inv;
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
	arg = (SNicsLUThreadArg *)(nicslu->thread_arg);
	tol = nicslu->cfgf[0];
	if (tol <= 1.e-32)
	{
		tol = 1.e-32;
		nicslu->cfgf[0] = tol;
	}
	else if (tol > 0.99999999)
	{
		tol = 0.99999999;
		nicslu->cfgf[0] = tol;
	}
	level = nicslu->aeg_level;
	aeg_data = nicslu->aeg_data;
	aeg_head = nicslu->aeg_header;
	thfi = nicslu->thread_finish;
	thac = nicslu->thread_active;
	start = nicslu->cluster_start;
	end = nicslu->cluster_end;
	wkld = nicslu->wkld_est;
	bal = nicslu->cfgf[4];

	/*begin*/
	TimerStart((STimer *)(nicslu->timer));

	/*mark all columns as non-pivotal*/
	for (i=0; i<n; ++i)
	{
		p[i] = i;
		pinv[i] = -((int__t)i)-2;
	}

	/*mark all rows un-pruned*/
	pend = (int__t *)(nicslu->workspace);
	memset(pend, 0xff, sizeof(int__t)*n);

	/*mark all rows un-finished*/
	memset(nicslu->row_state, 0, sizeof(byte__t)*n);

	/*clear flag, updated, busy, return value, and x*/
	for (ti=0; ti<th; ++ti)
	{
		memset(nicslu->workspace_mt1[ti], 0, sizeof(real__t)*n);/*x*/
		memset(((real__t *)(nicslu->workspace_mt1[ti]))+n, 0xff, sizeof(int__t)*n);/*flag*/
		memset(nicslu->workspace_mt2[ti], 0xff, sizeof(int__t)*(n+n));/*updated and busy*/
		arg[ti].lnnz = 0;
		arg[ti].unnz = 0;
		arg[ti].offdiag = 0;
		arg[ti].err = NICS_OK;
	}

	/*mode*/
	/*0: cluster*/
	/*1: pipeline*/

	/*start parallel factorization*/
	clv = 0;
	while (clv < level)
	{
		len = aeg_head[clv+1] - aeg_head[clv];
		mode = ((len <= thres) ? 1 : 0);

		nicslu->thread_work = ((mode == 0) ? NICSLU_WORK_FACT_CLUSTER : NICSLU_WORK_FACT_PIPELINE);
		lstart = clv;

		/*search lend*/
		if (mode == 0)/*cluster*/
		{
			lend = level;
			++clv;
			while (clv < level)
			{
				len = aeg_head[clv+1] - aeg_head[clv];
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
				aegh = aeg_head[lv];
				len = aeg_head[lv+1];
				total = 0;
				for (j=aegh; j<len; ++j)
				{
					i = aeg_data[j];
					total += wkld[i];
				}
				avg = (size_t)((real__t)total / th * bal);

				j = aegh;
				sub = 0;
				while (j < len)
				{
					i = aeg_data[j];
					sub += wkld[i];
					++j;
					if (sub >= avg) break;
				}
				start[0] = aegh;
				end[0] = j;

				for (ti=1; ti<th; ++ti)
				{
					start[ti] = end[ti-1];

					sub = 0;
					while (j < len)
					{
						t = aeg_data[j];
						sub += wkld[t];
						++j;
						if (sub >= avg) break;
					}
					if (ti == th-1) end[ti] = len;
					else end[ti] = j;

					thfi[ti] = FALSE;
					thac[ti] = TRUE;
				}

				_I_NicsLU_Factorize_Cluster(nicslu, arg, 0);

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
				len = aeg_head[clv+1] - aeg_head[clv];
				if (len > thres)
				{
					lend = clv;
					break;
				}
				++clv;
			}

			nicslu->pipeline_start = aeg_head[lstart];
			nicslu->pipeline_end = aeg_head[lend];
			nicslu->last_busy = nicslu->pipeline_start;

			/*begin pipeline*/
			for (ti=1; ti<th; ++ti)
			{
				thfi[ti] = FALSE;
				thac[ti] = TRUE;
			}

			_I_NicsLU_Factorize_Pipeline(nicslu, arg, 0);

			_SpinBarrier(1, th, (volatile char *)thfi);
		}
	}

	/*check the return value*/
	for (ti=0; ti<th; ++ti)
	{
		err = arg[ti].err;
		if (FAIL(err))
		{
			return err;
		}
	}

	/*nnz*/
	tl = 0;
	tu = 0;
	offd = 0;

	for (ti=0; ti<th; ++ti)
	{
		tl += arg[ti].lnnz;
		tu += arg[ti].unnz;
		offd += arg[ti].offdiag;
	}

	nnz = tl + tu;
	nicslu->l_nnz = tl + n;
	nicslu->u_nnz = tu + n;
	nicslu->lu_nnz = nnz + n;
	nicslu->stat[14] = (real__t)offd;
	nicslu->stat[26] = (real__t)(nicslu->l_nnz);
	nicslu->stat[27] = (real__t)(nicslu->u_nnz);
	nicslu->stat[28] = (real__t)(nicslu->lu_nnz);

	/*put into the array*/
	lu = nicslu->lu_array;
	lu2 = nicslu->lu_array2;
	ulen = nicslu->ulen;
	llen = nicslu->llen;
	up = nicslu->up;

	if (nnz > nicslu->lu_nnz_est)
	{
		nicslu->lu_nnz_est = nnz;
	}
	if (lu == NULL)
	{
		lu = malloc(g_sp*(nicslu->lu_nnz_est));
		if (lu == NULL)
		{
			return NICSLU_MEMORY_OVERFLOW;
		}
		nicslu->lu_array = lu;
	}
	else
	{
		void *ptr = realloc(lu, g_sp*(nicslu->lu_nnz_est));
		if (ptr == NULL)
		{
			return NICSLU_MEMORY_OVERFLOW;
		}
		nicslu->lu_array = ptr;
	}

#if 1
	/*sequential*/
	nnz = 0;
	for (i=0; i<n; ++i)
	{
		ul = ulen[i];
		ll = llen[i];

		up[i] = nnz;
		
		tl = (ul+ll) * g_sp;
		memcpy(((byte__t *)lu)+nnz, ((uint__t *)lu2[i])+ul, tl);

		nnz += tl;
	}
#else
	/*parallel*/
	nicslu->thread_work = NICSLU_WORK_COPY_DATA;

	nnz = 0;
	for (i=0; i<n; ++i)
	{
		ul = ulen[i];
		ll = llen[i];

		up[i] = nnz;
		ux[i] = nnz + g_si*ul;
		lp[i] = ux[i] + g_sd*ul;
		lx[i] = lp[i] + g_si*ll;
		
		tl = (ul+ll) * g_sp;
		nnz += tl;
	}

	nnz = (size_t)(nicslu->lu_nnz/(real__t)th * bal);

	j = 0;
	tl = 0;
	while (j < n)
	{
		tl += ulen[j] + llen[j];
		++j;
		if (tl >= nnz) break;
	}
	start[0] = 0;
	end[0] = j;

	for (ti=1; ti<th; ++ti)
	{
		start[ti] = end[ti-1];

		tl = 0;
		while (j < n)
		{
			tl += ulen[j] + llen[j];
			++j;
			if (tl >= nnz) break;
		}
		if (ti == th-1) end[ti] = n;
		else end[ti] = j;

		thfi[ti] = FALSE;
		thac[ti] = TRUE;
	}

	_I_NicsLU_CopyData(nicslu, 0);

	_SpinBarrier(1, th, (volatile char *)thfi);

#endif

	/*put U in the pivoting order*/
	for (i=0; i<n; ++i)
	{
		uip = (uint__t *)(((byte__t *)lu) + up[i]);
		ul = ulen[i];

		for (j=0; j<ul; ++j)
		{
			uip[j] = pinv[uip[j]];
		}
	}
	
	/*finish*/
	nicslu->flag[2] = TRUE;

	TimerStop((STimer *)(nicslu->timer));
	nicslu->stat[1] = TimerGetRuntime((STimer *)(nicslu->timer));

	return NICS_OK;
}
