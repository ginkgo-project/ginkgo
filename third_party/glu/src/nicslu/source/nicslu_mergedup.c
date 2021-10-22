/*last modified: june 11, 2013*/

#include "nicslu.h"
#include "nicslu_internal.h"

int NicsLU_MergeDuplicateEntries(uint__t n, uint__t *nnz, real__t **ax, uint__t **ai, uint__t **ap)
{
	real__t *nax, *bx;
	uint__t *nai, *nap, *bi, *bp, nz;
	uint__t i, j, end, col;
	int err;
	int__t pre;

	if (NULL == nnz || NULL == ax || NULL == ai || NULL == ap)
	{
		return NICSLU_ARGUMENT_ERROR;
	}
	if (NULL == *ax || NULL == *ai || NULL == *ap)
	{
		return NICSLU_ARGUMENT_ERROR;
	}
	if (0 == n || 0 == *nnz || *nnz != (*ap)[n])
	{
		return NICSLU_MATRIX_INVALID;
	}

	bx = *ax;
	bi = *ai;
	bp = *ap;
	nz = *nnz;

	nax = (real__t *)malloc(sizeof(real__t)*nz);
	nai = (uint__t *)malloc(sizeof(uint__t)*nz);
	nap = (uint__t *)malloc(sizeof(uint__t)*(1+n));

	if (NULL == nax || NULL == nai || NULL == nap)
	{
		if (nax != NULL) free(nax);
		if (nai != NULL) free(nai);
		if (nap != NULL) free(nap);
		return NICSLU_MEMORY_OVERFLOW;
	}

	err = _I_NicsLU_Sort(n, bx, bi, bp, nax, nai, nap);
	if (FAIL(err))
	{
		free(nax);
		free(nai);
		free(nap);
		return err;
	}

	nz = 0;
	*nap = 0;
	for (i=0; i<n; ++i)
	{
		pre = -1;
		end = bp[i+1];
		for (j=bp[i]; j<end; ++j)
		{
			col = bi[j];
			if (((int__t)col) != pre)
			{
				pre = col;
				nax[nz] = bx[j];
				nai[nz] = col;
				++nz;
			}
			else
			{
				nax[nz-1] += bx[j];
			}
		}
		nap[i+1] = nz;
	}

	*nnz = nz;
	nax = (real__t *)realloc(nax, sizeof(real__t)*nz);
	nai = (uint__t *)realloc(nai, sizeof(uint__t)*nz);
	/*here both arrays are shortened, so no errors can occur*/

	free(*ax);
	free(*ai);
	free(*ap);
	*ax = nax;
	*ai = nai;
	*ap = nap;

	return NICS_OK;
}
