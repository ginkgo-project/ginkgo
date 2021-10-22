/*check the matrix*/
/*last modified: june 16, 2013*/
/*author: Chen, Xiaoming*/

#include "nicslu.h"
#include "nicslu_internal.h"

int _I_NicsLU_CheckMatrix(uint__t n, uint__t nnz, uint__t *ai, uint__t *ap)
{
	uint__t i;
	if (0 == n) return NICSLU_MATRIX_INVALID;
	if (0 == nnz) return NICSLU_MATRIX_INVALID;
	if (0 != ap[0]) return NICSLU_MATRIX_INVALID;
	if (nnz != ap[n]) return NICSLU_MATRIX_INVALID;
	for (i=1; i<n; ++i)
	{
		if (ap[i] > ap[i+1]) return NICSLU_MATRIX_INVALID;
	}
	for (i=0; i<nnz; ++i)
	{
		if (ai[i] >= n) return NICSLU_MATRIX_INVALID;
	}
	return NICS_OK;
}

int _I_NicsLU_Check(SNicsLU *nicslu)
{
	int err;
	uint__t p1, p2, p;
	uint__t i, j, n;
	int__t jlast;
	uint__t *ai, *ap;

	if (NULL == nicslu)
	{
		return NICSLU_ARGUMENT_ERROR;
	}
	if (!nicslu->flag[0])
	{
		return NICSLU_MATRIX_NOT_INITIALIZED;
	}

	err = NICS_OK;
	n = nicslu->n;
	ai = nicslu->ai;
	ap = nicslu->ap;

	for (i=0; i<n; ++i)
	{
		p1 = ap[i];
		p2 = ap[i+1];
		if (p1 > p2)
		{
			return NICSLU_MATRIX_INVALID;
		}
		jlast = -1;
		for (p=p1; p<p2; ++p)
		{
			j = ai[p];
			if ((uint__t)j >= n)
			{
				return NICSLU_MATRIX_INVALID;
			}
			if ((int__t)j == jlast)
			{
				return NICSLU_MATRIX_ENTRY_DUPLICATED;
			}
			if (j < (uint__t)jlast)
			{
				err = NICSLU_MATRIX_NOT_SORTED;
			}
			jlast = j;
		}
	}

	return err;
}
