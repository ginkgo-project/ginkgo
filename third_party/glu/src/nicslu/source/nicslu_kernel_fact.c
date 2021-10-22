/*numeric lu factorization, kernel functions*/
/*last modified: june 16, 2013*/
/*author: Chen, Xiaoming*/

#include "nicslu.h"
#include "nicslu_internal.h"
#include "timer_c.h"
#include "nicslu_default.h"
#ifdef SSE2
#include <emmintrin.h>
#endif

/*lu data structure*/
/*row 0: all u indexes    all u data     all l indexes    all l data*/
/*row 1: all u indexes    all u data     all l indexes    all l data*/
/*......*/

extern size_t g_sd, g_si, g_sp, g_sis1, g_sps1;

int NicsLU_Factorize(SNicsLU *nicslu)
{
	real__t *ax;
	uint__t *ai, *ap;
	uint__t cstart, cend, oldrow;
	uint__t *rowperm;
	size_t lnnz, unnz;/*the actual nnz*/
	real__t *ldiag;
	void *lu_array;
	size_t *up;
	uint__t *llen, *ulen;
	uint__t n;
	int__t *p, *pinv;
	real__t tol;
	uint__t offdp;
	uint__t scale;
	real__t *cscale;
	real__t *x;
	int__t *flag, *pend, *stack, *appos;
	size_t size;
	size_t m0, m1, used;
	uint__t i, j, jnew, k, q, kb;
	real__t xj;
	int__t top;
	int__t diagcol;
	uint__t pivcol;
	real__t pivot;
	uint__t ul, ll;
	uint__t *row_index;
	real__t *row_data;
	uint__t *u_row_index_j;
	real__t *u_row_data_j;
	uint__t u_row_len_j;
	size_t est;
	int err;
	real__t grow;
#ifdef SSE2
	__m128d _xj, _prod, _tval;
	uint__t _end;
#endif

	/*check flags*/
	if (NULL == nicslu)
	{
		return NICSLU_ARGUMENT_ERROR;
	}
	nicslu->flag[2] = FALSE;
	if (!nicslu->flag[1])
	{
		return NICSLU_MATRIX_NOT_ANALYZED;
	}

	/*set parameters*/
	n = nicslu->n;
	ax = nicslu->ax;
	ai = nicslu->ai;
	ap = nicslu->ap;
	rowperm = nicslu->row_perm;
	lnnz = 0;
	unnz = 0;
	ldiag = nicslu->ldiag;
	up = nicslu->up;
	llen = nicslu->llen;
	ulen = nicslu->ulen;
	p = nicslu->pivot;
	pinv = nicslu->pivot_inv;
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
	offdp = 0;
	scale = nicslu->cfgi[2];
	cscale = nicslu->cscale;
	grow = nicslu->cfgf[5];
	if (grow <= 1.)
	{
		grow = NICSLU_MEMORY_GROW;
		nicslu->cfgf[5] = grow;
	}

	/*begin*/
	TimerStart((STimer *)(nicslu->timer));

	/*mark all columns as non-pivotal*/
	for (i=0; i<n; ++i)
	{
		p[i] = i;
		pinv[i] = -((int__t)i)-2;
	}

	/*work*/
	/*|-----|-----|-----|-----|-----| */
	/* x     flag  pend  stack appos */
	x = (real__t *)(nicslu->workspace);
	flag = (int__t *)(x + n);
	pend = flag + n;
	stack = pend + n;
	appos = stack + n;

	memset(x, 0, sizeof(real__t)*n);
	memset(flag, 0xff, sizeof(int__t)*(n+n));/*clear flag and pend*/


	/*alloc lu data*/
	if (nicslu->lu_array == NULL)
	{
		est = nicslu->lu_nnz_est;
		size = g_sp*est;
		lu_array = malloc(size);
		if (NULL == lu_array)
		{
			return NICSLU_MEMORY_OVERFLOW;
		}
		nicslu->lu_array = lu_array;
		m0 = size;
		used = 0;
	}
	else
	{
		est = nicslu->lu_nnz_est;
		size = g_sp*est;
		lu_array = realloc(nicslu->lu_array, size);
		if (NULL == lu_array)
		{
			return NICSLU_MEMORY_OVERFLOW;
		}
		nicslu->lu_array = lu_array;
		m0 = size;
		used = 0;
	}

	/*numeric factorize*/
	for (i=0; i<n; ++i)
	{
		up[i] = used;
		oldrow = rowperm[i];
		cstart = ap[oldrow];
		cend = ap[oldrow+1];

		/*estimate the length*/
		m1 = used + n*g_sp;

		if (m1 > m0)
		{

			m0 = ( ( ((size_t)(m0 * grow)) + g_sps1) / g_sp) * g_sp;
			
			if (m1 > m0)
			{
				m0 = ( ( ((size_t)(m1 * grow)) + g_sps1) / g_sp) * g_sp;
			}

			lu_array = realloc(lu_array, m0);
			if (NULL == lu_array)
			{
				return NICSLU_MEMORY_OVERFLOW;
			}
			nicslu->lu_array = lu_array;
			est = m0/g_sp;
			nicslu->lu_nnz_est = est;
		}

		/*symbolic*/
		top = _I_NicsLU_Symbolic(n, i, pinv, stack, flag, pend, appos, \
			(uint__t *)(((byte__t *)lu_array) + up[i]), ulen, lu_array, up, &ai[cstart], cend-cstart);

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
			u_row_index_j = (uint__t *)(((byte__t *)lu_array) + up[jnew]);
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

		/*pivoting*/
		diagcol = p[i];/*column diagcol is the ith pivot*/
		err = _I_NicsLU_Pivot(diagcol, &ulen[i], up[i], tol, x, \
			&pivcol, &pivot, lu_array);
		if (FAIL(err))
		{
			return err;
		}

		/*update up, ux, lp, lx, ulen, llen*/
		ll = llen[i] = n-top;
		ul = ulen[i];

		row_index = (uint__t *)(((byte__t *)lu_array) + up[i] + ul*g_sp);
		row_data = (real__t *)(row_index + ll);

		/*push into L*/
		for (k=top, q=0; k<n; ++k, ++q)
		{
			j = stack[k];
			row_index[q] = pinv[j];/*!!! L is put in pivoting order here*/
			row_data[q] = x[j];
			x[j] = 0.;
		}

		ldiag[i] = pivot;

		lnnz += ll;
		unnz += ul;

		used += (ul+ll) * g_sp;

		/*log the pivoting*/
		if (pivcol != diagcol)/*diagcol = p[i]*/
		{
			++offdp;
			if (pinv[diagcol] < 0)
			{
				kb = -pinv[pivcol]-2;/*pinv[pivcol] must < 0*/
				p[kb] = diagcol;
				pinv[diagcol] = -(int__t)kb-2;
			}
		}
		p[i] = pivcol;
		pinv[pivcol] = i;
		
		/*prune*/
		_I_NicsLU_Prune(pend, ll, ulen, pinv, pivcol, row_index, up, lu_array);
	}

	/*put U in the pivoting order*/
	for (k=0; k<n; ++k)
	{
		row_index = (uint__t *)(((byte__t *)lu_array) + up[k]);
		ll = ulen[k];

		for (i=0; i<ll; ++i)
		{
			row_index[i] = pinv[row_index[i]];
		}
	}

	nicslu->l_nnz = lnnz + n;
	nicslu->u_nnz = unnz + n;
	nicslu->lu_nnz = lnnz + unnz + n;
	nicslu->stat[14] = (real__t)offdp;
	nicslu->stat[26] = (real__t)(nicslu->l_nnz);
	nicslu->stat[27] = (real__t)(nicslu->u_nnz);
	nicslu->stat[28] = (real__t)(nicslu->lu_nnz);

	nicslu->flag[2] = TRUE;

	/*finish*/
	TimerStop((STimer *)(nicslu->timer));
	nicslu->stat[1] = TimerGetRuntime((STimer *)(nicslu->timer));

	return NICS_OK;
}
