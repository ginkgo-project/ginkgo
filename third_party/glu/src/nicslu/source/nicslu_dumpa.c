#include "nicslu.h"
#include "nicslu_internal.h"

int NicsLU_DumpA(SNicsLU *nicslu, real__t **ax, uint__t **ai, uint__t **ap)
{
	uint__t n, nnz;
	real__t *ax0;
	uint__t *ai0, *ap0, *rowperm, *pinv, oldrow, start, end;
	uint__t i, j, p;

	if (NULL == nicslu || NULL == ax || NULL == ai || NULL == ap)
	{
		return NICSLU_ARGUMENT_ERROR;
	}
	if (!nicslu->flag[2])
	{
		return NICSLU_MATRIX_NOT_FACTORIZED;
	}

	if (*ax != NULL)
	{
		free(*ax);
		*ax = NULL;
	}
	if (*ai != NULL)
	{
		free(*ai);
		*ai = NULL;
	}
	if (*ap != NULL)
	{
		free(*ap);
		*ap = NULL;
	}

	n = nicslu->n;
	nnz = nicslu->nnz;
	ax0 = nicslu->ax;
	ai0 = nicslu->ai;
	ap0 = nicslu->ap;
	rowperm = nicslu->row_perm;/*row_perm[i]=j-->row i in the permuted matrix is row j in the original matrix*/
	pinv = (uint__t *)nicslu->pivot_inv;/*pivot_inv[i]=j-->column i is the jth pivot column*/

	*ax = (real__t *)malloc(sizeof(real__t)*nnz);
	*ai = (uint__t *)malloc(sizeof(uint__t)*nnz);
	*ap = (uint__t *)malloc(sizeof(uint__t)*(n+1));
	if (NULL == *ax || NULL == *ai || NULL == *ap)
	{
		goto FAIL;
	}
	(*ap)[0] = 0;

	p = 0;
	for (i=0; i<n; ++i)
	{
		oldrow = rowperm[i];
		start = ap0[oldrow];
		end = ap0[oldrow+1];
		(*ap)[i+1] = (*ap)[i] + end - start;

		for (j=start; j<end; ++j)
		{
			(*ax)[p] = ax0[j];
			(*ai)[p++] = pinv[ai0[j]];
		}
	}

	return NICS_OK;

FAIL:
	if (*ax != NULL)
	{
		free(*ax);
		*ax = NULL;
	}
	if (*ai != NULL)
	{
		free(*ai);
		*ai = NULL;
	}
	if (*ap != NULL)
	{
		free(*ap);
		*ap = NULL;
	}
	return NICSLU_MEMORY_OVERFLOW;
}
