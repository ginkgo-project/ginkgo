/*last modified: june 7, 2013*/

#include "nicslu.h"
#include "nicslu_internal.h"

int NicsLU_Sort(uint__t n, uint__t nnz, real__t *ax, uint__t *ai, uint__t *ap)
{
	uint__t i, j;
	uint__t *bi, *bp;
	real__t *bx;
	uint__t *len;
	uint__t end, col;

	if (ai == NULL || ap == NULL || ax == NULL)
	{
		return NICSLU_ARGUMENT_ERROR;
	}
	if (n == 0 || nnz == 0 || nnz != ap[n])
	{
		return NICSLU_MATRIX_INVALID;
	}

	bi = (uint__t *)malloc(sizeof(uint__t)*nnz);
	bp = (uint__t *)malloc(sizeof(uint__t)*(1+n));
	bx = (real__t *)malloc(sizeof(real__t)*nnz);
	len = (uint__t *)malloc(sizeof(uint__t)*n);

	if (NULL == bi || NULL == bp || NULL == bx || NULL == len)
	{
		if (bi != NULL) free(bi);
		if (bp != NULL) free(bp);
		if (bx != NULL) free(bx);
		if (len != NULL) free(len);
		return NICSLU_MEMORY_OVERFLOW;
	}

	memset(len, 0, sizeof(uint__t)*n);
	for (i=0; i<n; ++i)
	{
		end = ap[i+1];
		for (j=ap[i]; j<end; ++j)
		{
			col = ai[j];
			++len[col];
		}
	}
	bp[0] = 0;
	for (i=0; i<n; ++i)
	{
		bp[i+1] = bp[i] + len[i];
	}
	memcpy(len, bp, sizeof(uint__t)*n);

	for (i=0; i<n; ++i)
	{
		end = ap[i+1];
		for (j=ap[i]; j<end; ++j)
		{
			col = ai[j];
			bi[len[col]] = i;
			bx[len[col]] = ax[j];
			++len[col];
		}
	}

	memcpy(len, ap, sizeof(uint__t)*n);
	for (i=0; i<n; ++i)
	{
		end = bp[i+1];
		for (j=bp[i]; j<end; ++j)
		{
			col = bi[j];
			ai[len[col]] = i;
			ax[len[col]] = bx[j];
			++len[col];
		}
	}

	free(bi);
	free(bx);
	free(bp);
	free(len);
	return NICS_OK;
}

int _I_NicsLU_Sort(uint__t n, real__t *ax, uint__t *ai, uint__t *ap, \
						   real__t *bx, uint__t *bi, uint__t *bp)
{
	uint__t i, j;
	uint__t *len;
	uint__t end, col;

	len = (uint__t *)malloc(sizeof(uint__t)*n);
	if (NULL == len)
	{
		return NICSLU_MEMORY_OVERFLOW;
	}

	memset(len, 0, sizeof(uint__t)*n);
	for (i=0; i<n; ++i)
	{
		end = ap[i+1];
		for (j=ap[i]; j<end; ++j)
		{
			col = ai[j];
			++len[col];
		}
	}
	bp[0] = 0;
	for (i=0; i<n; ++i)
	{
		bp[i+1] = bp[i] + len[i];
	}
	memcpy(len, bp, sizeof(uint__t)*n);

	for (i=0; i<n; ++i)
	{
		end = ap[i+1];
		for (j=ap[i]; j<end; ++j)
		{
			col = ai[j];
			bi[len[col]] = i;
			bx[len[col]] = ax[j];
			++len[col];
		}
	}

	memcpy(len, ap, sizeof(uint__t)*n);
	for (i=0; i<n; ++i)
	{
		end = bp[i+1];
		for (j=bp[i]; j<end; ++j)
		{
			col = bi[j];
			ai[len[col]] = i;
			ax[len[col]] = bx[j];
			++len[col];
		}
	}

	free(len);
	return NICS_OK;
}
