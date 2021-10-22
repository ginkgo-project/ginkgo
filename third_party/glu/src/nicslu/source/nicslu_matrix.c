/*create and destroy the matrix*/
/*last modified: june 16, 2013*/
/*author: Chen, Xiaoming*/

#include "nicslu.h"
#include "nicslu_internal.h"
#include "system.h"
#include "nicslu_default.h"

int NicsLU_CreateMatrix(SNicsLU *nicslu, uint__t n, uint__t nnz, real__t *ax, uint__t *ai, uint__t *ap)
{
	int err;
	void *ptr;
	size_t size;

	if (NULL == nicslu)
	{
		return NICSLU_ARGUMENT_ERROR;
	}

	/*destroy*/
	_I_NicsLU_DestroyMatrix(nicslu);

	if (NULL == ax || NULL == ai || NULL == ap)
	{
		return NICSLU_ARGUMENT_ERROR;
	}

	err = _I_NicsLU_CheckMatrix(n, nnz, ai, ap);
	if (FAIL(err))
	{
		return err;
	}

	err = _I_NicsLU_AllocMatrixMemory(n, nnz, nicslu);
	if (FAIL(err))
	{
		return err;
	}

	nicslu->n = n;
	nicslu->nnz = nnz;

	memcpy(nicslu->ax, ax, sizeof(real__t)*nnz);
	memcpy(nicslu->ai, ai, sizeof(int__t)*nnz);
	memcpy(nicslu->ap, ap, sizeof(int__t)*(1+n));

	/*perm*/
	size = sizeof(int__t)*n;
	ptr = malloc(size*6);
	if (NULL == ptr)
	{
		return NICSLU_MEMORY_OVERFLOW;
	}
	memset(ptr, 0, size);
	nicslu->row_perm = (uint__t *)ptr;
	ptr = ((uint__t *)ptr) + n;
	nicslu->row_perm_inv = (uint__t *)ptr;
	ptr = ((uint__t *)ptr) + n;
	nicslu->col_perm = (uint__t *)ptr;
	ptr = ((uint__t *)ptr) + n;
	nicslu->col_perm_inv = (uint__t *)ptr;
	ptr = ((uint__t *)ptr) + n;
	nicslu->pivot = (int__t *)ptr;
	ptr = ((uint__t *)ptr) + n;
	nicslu->pivot_inv = (int__t *)ptr;

	/*mc64_scale & scale*/
	size = sizeof(real__t)*n*3;
	ptr = malloc(size);
	if (NULL == ptr)
	{
		return NICSLU_MEMORY_OVERFLOW;
	}
	nicslu->col_scale_perm = (real__t *)ptr;
	ptr = ((real__t *)ptr) + n;
	nicslu->row_scale = (real__t *)ptr;
	ptr = ((real__t *)ptr) + n;
	nicslu->cscale = (real__t *)ptr;

	/*lu matrix*/
	size = sizeof(real__t)*n;
	ptr = malloc(size);
	if (NULL == ptr)
	{
		return NICSLU_MEMORY_OVERFLOW;
	}
	nicslu->ldiag = (real__t *)ptr;

	size = sizeof(size_t)*n;
	ptr = malloc(size);
	if (NULL == ptr)
	{
		return NICSLU_MEMORY_OVERFLOW;
	}
	nicslu->up = (size_t *)ptr;

	size = sizeof(uint__t)*(n+n);
	ptr = malloc(size);
	if (NULL == ptr)
	{
		return NICSLU_MEMORY_OVERFLOW;
	}
	nicslu->llen = (uint__t *)ptr;
	ptr = ((uint__t *)ptr) + n;
	nicslu->ulen = (uint__t *)ptr;

	/*work space*/
	size = (sizeof(int__t)*4+sizeof(real__t)) * n;
	ptr = malloc(size);
	if (NULL == ptr)
	{
		return NICSLU_MEMORY_OVERFLOW;
	}
	nicslu->workspace = ptr;

	/*finish*/
	nicslu->flag[0] = TRUE;

	return NICS_OK;
}

int _I_NicsLU_DestroyMatrix(SNicsLU *nicslu)
{
	if (NULL == nicslu)
	{
		return NICSLU_ARGUMENT_ERROR;
	}

	NicsLU_DestroyThreads(nicslu);
	_I_NicsLU_DestroyScheduler(nicslu);

	if (nicslu->flag != NULL)
	{
		memset(nicslu->flag, 0, sizeof(bool__t)*32);
	}
	if (nicslu->stat != NULL)
	{
		memset(nicslu->stat, 0, sizeof(real__t)*32);
		nicslu->stat[9] = GetProcessorNumber();
	}
	if (nicslu->cfgi != NULL)
	{
		memset(nicslu->cfgi, 0, sizeof(uint__t)*32);
		nicslu->cfgi[0] = 0;/*row/column*/
		nicslu->cfgi[1] = 1;/*mc64*/
		nicslu->cfgi[2] = 0;/*scale*/
		nicslu->cfgi[3] = NICSLU_PIPELINE_THRESHOLD;
		nicslu->cfgi[4] = NICSLU_STATIC_RNNZ_UB;
		nicslu->cfgi[5] = 1;/*threads created*/
		nicslu->cfgi[6] = NICSLU_AMD_FLAG1;
		nicslu->cfgi[7] = 1;/*threads used*/
		nicslu->cfgi[8] = GetProcessorNumber();
	}
	if (nicslu->cfgf != NULL)
	{
		memset(nicslu->cfgf, 0, sizeof(real__t)*32);
		nicslu->cfgf[0] = NICSLU_PIVOT_TOLERANCE;
		nicslu->cfgf[1] = NICSLU_STATIC_MEMORY_MULT;
		nicslu->cfgf[2] = NICSLU_AMD_FLAG2;
		nicslu->cfgf[3] = NICSLU_SYNC_CYCLES;
		nicslu->cfgf[4] = NICSLU_LOAD_BALANCE;
		nicslu->cfgf[5] = NICSLU_MEMORY_GROW;
	}

	if (nicslu->ax != NULL)
	{
		free(nicslu->ax);
		nicslu->ax = NULL;
	}
	if (nicslu->ai != NULL)
	{
		free(nicslu->ai);
		nicslu->ai = NULL;
	}
	if (nicslu->ap != NULL)
	{
		free(nicslu->ap);
		nicslu->ap = NULL;
	}
	if (nicslu->rhs != NULL)
	{
		free(nicslu->rhs);
		nicslu->rhs = NULL;
	}
	nicslu->n = 0;
	nicslu->nnz = 0;

	if (nicslu->row_perm != NULL)
	{
		free(nicslu->row_perm);
	}
	nicslu->row_perm = NULL;
	nicslu->row_perm_inv = NULL;
	nicslu->col_perm = NULL;
	nicslu->col_perm_inv = NULL;
	nicslu->pivot = NULL;
	nicslu->pivot_inv = NULL;

	if (nicslu->col_scale_perm != NULL)
	{
		free(nicslu->col_scale_perm);
		nicslu->col_scale_perm = NULL;
	}
	nicslu->row_scale = NULL;
	nicslu->cscale = NULL;

	nicslu->lu_nnz_est = 0;
	nicslu->lu_nnz = 0;
	nicslu->l_nnz = 0;
	nicslu->u_nnz = 0;

	if (nicslu->ldiag != NULL)
	{
		free(nicslu->ldiag);
		nicslu->ldiag = NULL;
	}
	if (nicslu->lu_array != NULL)
	{
		free(nicslu->lu_array);
		nicslu->lu_array = NULL;
	}
	if (nicslu->up != NULL)
	{
		free(nicslu->up);
		nicslu->up = NULL;
	}
	if (nicslu->llen != NULL)
	{
		free(nicslu->llen);
		nicslu->llen = NULL;
	}
	nicslu->ulen = NULL;

	if (nicslu->workspace != NULL)
	{
		free(nicslu->workspace);
		nicslu->workspace = NULL;
	}

	return NICS_OK;
}

int NicsLU_ResetMatrixValues(SNicsLU *nicslu, real__t *ax)
{
	int err;

	if (NULL == nicslu || NULL == ax)
	{
		return NICSLU_ARGUMENT_ERROR;
	}
	if (!nicslu->flag[0])
	{
		return NICSLU_MATRIX_NOT_INITIALIZED;
	}
	nicslu->flag[2] = FALSE;
	nicslu->flag[5] = FALSE;

	memcpy(nicslu->ax, ax, sizeof(real__t)*(nicslu->nnz));

	if (nicslu->flag[1])
	{
		_I_NicsLU_MC64Scale(nicslu);
		err = _I_NicsLU_Scale(nicslu);
		if (FAIL(err)) return err;
	}

	return NICS_OK;
}
