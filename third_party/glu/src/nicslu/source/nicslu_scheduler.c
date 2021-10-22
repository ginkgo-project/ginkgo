/*create and destroy the scheduler*/
/*last modified: june 7, 2013*/
/*author: Chen, Xiaoming*/

#include "nicslu.h"
#include "nicslu_internal.h"
#include "timer_c.h"

int NicsLU_CreateScheduler(SNicsLU *nicslu)
{
	uint__t n;
	size_t size;
	void *ptr;
/*	bool__t a, b;
	real__t nnz;*/
	int err;
	
	if (NULL == nicslu)
	{
		return NICSLU_ARGUMENT_ERROR;
	}
	if (!nicslu->flag[1])
	{
		return NICSLU_MATRIX_NOT_ANALYZED;
	}

	TimerStart((STimer *)(nicslu->timer));
	_I_NicsLU_DestroyScheduler(nicslu);

	n = nicslu->n;

	/*wkld_est & len_est*/
	size = sizeof(size_t)*(n+n);
	ptr = malloc(size);
	if (NULL == ptr)
	{
		return NICSLU_MEMORY_OVERFLOW;
	}
	nicslu->wkld_est = (size_t *)ptr;
	ptr = ((size_t *)ptr) + n;
	nicslu->len_est = (size_t *)ptr;

	/*row state*/
	size = sizeof(byte__t)*n;
	ptr = malloc(size);
	if (NULL == ptr)
	{
		return NICSLU_MEMORY_OVERFLOW;
	}
	nicslu->row_state = (byte__t *)ptr;

	/*scheduler*/
	size = sizeof(uint__t) * (4*n+2);
	ptr = malloc(size);
	if (NULL == ptr)
	{
		return NICSLU_MEMORY_OVERFLOW;
	}
	nicslu->aeg_data = (uint__t *)ptr;
	ptr = ((uint__t *)ptr) + n;
	nicslu->aeg_header = (uint__t *)ptr;
	ptr = ((uint__t *)ptr) + n + 1;
	nicslu->aeg_refact_data = (uint__t *)ptr;
	ptr = ((uint__t *)ptr) + n;
	nicslu->aeg_refact_header = (uint__t *)ptr;

	/*lu_array2*/
	size = sizeof(void *) * n;
	ptr = malloc(size);
	if (NULL == ptr)
	{
		return NICSLU_MEMORY_OVERFLOW;
	}
	memset(ptr, 0, size);
	nicslu->lu_array2 = (void **)ptr;

	/*calculate the scheduler*/
	err = _I_NicsLU_CreateETree(nicslu);
	if (FAIL(err)) return err;

	err = _I_NicsLU_StaticSymbolicFactorize(nicslu);
	if (FAIL(err)) return err;

	/*finish*/
	nicslu->flag[4] = TRUE;
	
/*	nnz = nicslu->stat[11];
	a = ((nnz/(real__t)(nicslu->nnz)) >= 1.99999999);
	b = ((nicslu->stat[10]/nnz) >= 50.);*/

	TimerStop((STimer *)(nicslu->timer));
	nicslu->stat[4] = TimerGetRuntime((STimer *)(nicslu->timer));

/*	err = (!a && !b);*/
	err = ((nicslu->stat[10]/nicslu->stat[11] < 50.) ? NICSLU_USE_SEQUENTIAL_FACTORIZATION : NICS_OK);
	nicslu->stat[13] = err;
	return err;
}

int _I_NicsLU_DestroyScheduler(SNicsLU *nicslu)
{
	uint__t i, n;
	void **ppv;

	if (NULL == nicslu)
	{
		return NICSLU_ARGUMENT_ERROR;
	}
	n = nicslu->n;

	if (nicslu->wkld_est != NULL)
	{
		free(nicslu->wkld_est);
		nicslu->wkld_est = NULL;
	}
	nicslu->len_est = NULL;
	if (nicslu->row_state != NULL)
	{
		free(nicslu->row_state);
		nicslu->row_state = NULL;
	}
	if (nicslu->aeg_data != NULL)
	{
		free(nicslu->aeg_data);
		nicslu->aeg_data = NULL;
	}
	nicslu->aeg_header = NULL;
	nicslu->aeg_refact_data = NULL;
	nicslu->aeg_refact_header = NULL;

	if (nicslu->lu_array2 != NULL)
	{
		ppv = nicslu->lu_array2;
		for (i=0; i<n; ++i)
		{
			if (ppv[i] != NULL)
			{
				free(ppv[i]);
			}
		}
		free(ppv);
		nicslu->lu_array2 = NULL;
	}

	nicslu->flag[4] = FALSE;
	return NICS_OK;
}
