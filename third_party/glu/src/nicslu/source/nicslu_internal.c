/*internal functions*/
/*last modified: june 11, 2013*/
/*author: Chen, Xiaoming*/

#include "nicslu.h"
#include "nicslu_internal.h"

extern size_t g_sd, g_si, g_sp, g_sis1, g_sps1;

int _I_NicsLU_AllocMatrixMemory(uint__t n, uint__t nnz, SNicsLU *nicslu)
{
	real__t *ax;
	uint__t *ai;
	uint__t *ap;
	real__t *rhs;

	ax = (real__t *)malloc(sizeof(real__t)*nnz);
	ai = (uint__t *)malloc(sizeof(int__t)*nnz);
	ap = (uint__t *)malloc(sizeof(int__t)*(1+n));
	rhs = (real__t *)malloc(sizeof(real__t)*n);

	if (NULL == ax || NULL == ai || NULL == ap || NULL == rhs)
	{
		if (ax != NULL) free(ax);
		if (ai != NULL) free(ai);
		if (ap != NULL) free(ap);
		if (rhs != NULL) free(rhs);
		return NICSLU_MEMORY_OVERFLOW;
	}

	nicslu->ax = ax;
	nicslu->ai = ai;
	nicslu->ap = ap;
	nicslu->rhs = rhs;

	return NICS_OK;
}

int _I_NicsLU_ConstructCSR(SNicsLU *nicslu, int__t *match, uint__t *ai, uint__t *ap)
{
	uint__t row;
	uint__t end;
	uint__t i, j;
	uint__t ct;
	uint__t n;
	uint__t *ai0, *ap0;

	n = nicslu->n;
	ai0 = nicslu->ai;
	ap0 = nicslu->ap;
	ct = 0;
	ap[0] = 0;
	
	for (i=0; i<n; ++i)
	{
		row = match[i];
		end = ap0[1+row];
		for (j=ap0[row]; j<end; ++j)
		{
			ai[ct++] = ai0[j];
		}
		ap[i+1] = ct;
	}

	return NICS_OK;
}

int _I_NicsLU_Permute(SNicsLU *nicslu, int__t *match, int__t *p, int__t *pinv)
{
	uint__t row, i, n, nnz;
	uint__t *ai;
	uint__t *rp, *rpi;

	n = nicslu->n;
	nnz = nicslu->nnz;
	ai = nicslu->ai;
	rp = nicslu->row_perm;
	rpi = nicslu->row_perm_inv;

	for (i=0; i<n; ++i)
	{
		row = match[p[i]];

		rpi[row] = i;
		rp[i] = row;
	}

	for (i=0; i<nnz; ++i)
	{
		ai[i] = pinv[ai[i]];
	}

	memcpy(nicslu->col_perm, p, sizeof(int__t)*n);
	memcpy(nicslu->col_perm_inv, pinv, sizeof(int__t)*n);

	return NICS_OK;
}

void _I_NicsLU_CopyData(SNicsLU *nicslu, unsigned int id)
{
	uint__t i, ul, ll;
	size_t tl;
	uint__t *ulen, *llen;
	size_t *up;
	void *lu, **lu2;
	uint__t start, end;

	ulen = nicslu->ulen;
	llen = nicslu->llen;
	up = nicslu->up;
	lu = nicslu->lu_array;
	lu2 = nicslu->lu_array2;

	start = nicslu->cluster_start[id];
	end = nicslu->cluster_end[id];

	for (i=start; i<end; ++i)
	{
		ul = ulen[i];
		ll = llen[i];
		
		tl = (ul+ll) * g_sp;
		memcpy(((byte__t *)lu)+up[i], ((uint__t *)lu2[i])+ul, tl);
	}

	nicslu->thread_finish[id] = TRUE;
}
