/*re-factorize another matrix with the same structure, without partial pivoting*/
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

int NicsLU_ReFactorize(SNicsLU *nicslu, real__t *ax0)
{
	real__t *ax;
	uint__t *ai, *ap;
	uint__t cstart, cend, oldrow;
	uint__t *rowperm;
	int err;
	uint__t n;
	void *lu_array;
	real__t *ldiag;
	size_t *up;
	uint__t *ulen, *llen;
	real__t *x;
	uint__t i, k, p, j;
	uint__t *pinv;
	uint__t lcol;
	real__t xj;
	uint__t scale;
	real__t *cscale;
	uint__t *u_row_index_j;
	real__t *u_row_data_j;
	uint__t u_row_len_j;
	uint__t *l_row_index_i;
	real__t *l_row_data_i;
	uint__t l_row_len_i;
#ifdef SSE2
	__m128d _xj, _prod, _tval;
	uint__t _end;
#endif

	/*check flags*/
	if (NULL == nicslu || NULL == ax0)
	{
		return NICSLU_ARGUMENT_ERROR;
	}
	if (!nicslu->flag[2])
	{
		return NICSLU_MATRIX_NOT_FACTORIZED;
	}

	ax = nicslu->ax;
	ai = nicslu->ai;
	ap = nicslu->ap;
	rowperm = nicslu->row_perm;
	n = nicslu->n;
	lu_array = nicslu->lu_array;
	ldiag = nicslu->ldiag;
	up = nicslu->up;
	ulen = nicslu->ulen;
	llen = nicslu->llen;
	x = (real__t *)(nicslu->workspace);
	pinv = (uint__t *)(nicslu->pivot_inv);
	scale = nicslu->cfgi[2];
	cscale = nicslu->cscale;

	/*begin*/
	TimerStart((STimer *)(nicslu->timer));

	/*mc64_scale*/
	_I_NicsLU_MC64ScaleForRefact(nicslu, ax0);

	/*scale*/
	err = _I_NicsLU_Scale(nicslu);
	if (FAIL(err)) return err;

	memset(x, 0, sizeof(real__t)*n);

	for (i=0; i<n; ++i)
	{
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
			return NICSLU_MATRIX_NUMERIC_SINGULAR;
		}
		if (isNaN(xj))
		{
			return NICSLU_NUMERIC_OVERFLOW;
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
	}

	/*finish*/
	TimerStop((STimer *)(nicslu->timer));
	nicslu->stat[2] = TimerGetRuntime((STimer *)(nicslu->timer));

	return NICS_OK;
}
