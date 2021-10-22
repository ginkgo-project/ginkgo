/*refine the solution, used for ill-conditioned matrix*/
/*last modified: june 7, 2013*/
/*author: Chen, Xiaoming*/

#include "nicslu.h"
#include "nicslu_internal.h"
#include "math.h"
#include "timer_c.h"

int NicsLU_Refine(SNicsLU *nicslu, real__t *x, real__t *b, real__t error, uint__t maxiter)
{
	uint__t n, i, iter;
	real__t rnorm, laste;
	real__t *rr;

	if (NULL == nicslu || NULL == x || NULL == b || error <= 0.)
	{
		return NICSLU_ARGUMENT_ERROR;
	}
	if (!nicslu->flag[2])
	{
		return NICSLU_MATRIX_NOT_FACTORIZED;
	}

	TimerStart((STimer *)(nicslu->timer));

	iter = 0;
	n = nicslu->n;
	rr = (real__t *)nicslu->workspace;

	if (maxiter > 0)
	{
		while (iter < maxiter)
		{
			++iter;
			/*r = Ax-b*/
			_I_NicsLU_Residual(nicslu, x, b, rr);
			rnorm = 0.;
			for (i=0; i<n; ++i)
			{
				rnorm += ABS(rr[i]);
			}
			/*Adx=r*/
			NicsLU_Solve(nicslu, rr);
			if (rnorm > error)
			{
				/*x = x-dx*/
				for (i=0; i<n; ++i)
				{
					x[i] -= rr[i];
				}
			}
			else break;
		}
	}
	else
	{
		laste = DBL_MAX;
		while (TRUE)
		{
			++iter;
			/*r = Ax-b*/
			_I_NicsLU_Residual(nicslu, x, b, rr);
			rnorm = 0.;
			for (i=0; i<n; ++i)
			{
				rnorm += ABS(rr[i]);
			}
			/*Adx=r*/
			NicsLU_Solve(nicslu, rr);
			if (rnorm > error)
			{
				if (rnorm >= laste) break;
				laste = rnorm;
				/*x = x+dx*/
				for (i=0; i<n; ++i)
				{
					x[i] -= rr[i];
				}
			}
			else break;
		}
	}

	TimerStop((STimer *)(nicslu->timer));
	nicslu->stat[15] = TimerGetRuntime((STimer *)(nicslu->timer));
	nicslu->stat[16] = (real__t)iter;

	return NICS_OK;
}
