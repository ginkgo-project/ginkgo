/*create and destroy threads*/
/*last modified: june 7, 2013*/
/*author: Chen, Xiaoming*/

#include "nicslu.h"
#include "nicslu_internal.h"
#include "thread.h"
#include "system.h"

int NicsLU_CreateThreads(SNicsLU *nicslu, unsigned int threads, bool__t check)
{
	uint__t n;
	void *ptr;
	size_t size;
	SNicsLUThreadArg *tharg;
	unsigned int i, cores;

	if (NULL == nicslu)
	{
		return NICSLU_ARGUMENT_ERROR;
	}
	if (!nicslu->flag[0])
	{
		return NICSLU_MATRIX_NOT_INITIALIZED;
	}

	n = nicslu->n;

	/*check threads*/
	if (check)
	{
		cores = (unsigned int)(nicslu->cfgi[8]);
		if (threads > cores)
		{
			threads = cores;
		}
	}
	if (threads == 0)
	{
		return NICSLU_ARGUMENT_ERROR;
	}
	if (threads == 1)
	{
		return NICSLU_SINGLE_THREAD;
	}

	/*exit existing threads first*/
	NicsLU_DestroyThreads(nicslu);
	nicslu->cfgi[5] = threads;
	nicslu->cfgi[7] = threads;

	/*thread_id*/
	size = sizeof(thread_id__t) * threads;
	ptr = malloc(size);
	if (NULL == ptr)
	{
		return NICSLU_MEMORY_OVERFLOW;
	}
	memset(ptr, 0, size);
	nicslu->thread_id = ptr;

	/*thread arg*/
	size = sizeof(SNicsLUThreadArg) * threads;
	ptr = malloc(size);
	if (NULL == ptr)
	{
		return NICSLU_MEMORY_OVERFLOW;
	}
	nicslu->thread_arg = ptr;
	for (i=0; i<threads; ++i)
	{
		tharg = &(((SNicsLUThreadArg *)ptr)[i]);
		tharg->nicslu = nicslu;
		tharg->id = i;
	}

	/*thread work*/
	nicslu->thread_work = NICSLU_WORK_NONE;

	/*thread active & thread finish*/
	size = sizeof(bool__t) * (threads+threads);
	ptr = malloc(size);
	if (NULL == ptr)
	{
		return NICSLU_MEMORY_OVERFLOW;
	}
	memset(ptr, 0, size);
	nicslu->thread_finish = (bool__t *)ptr;
	ptr = ((bool__t *)ptr) + threads;
	nicslu->thread_active = (bool__t *)ptr;

	/*cluster start and end*/
	size = sizeof(uint__t) * (threads+threads);
	ptr = malloc(size);
	if (NULL == ptr)
	{
		return NICSLU_MEMORY_OVERFLOW;
	}
	nicslu->cluster_start = (uint__t *)ptr;
	ptr = ((uint__t *)ptr) + threads;
	nicslu->cluster_end = (uint__t *)ptr;

	/*work space*/
	size = sizeof(void *) * threads;
	ptr = malloc(size);
	if (NULL == ptr)
	{
		return NICSLU_MEMORY_OVERFLOW;
	}
	memset(ptr, 0, size);
	nicslu->workspace_mt1 = (void **)ptr;

	ptr = malloc(size);
	if (NULL == ptr)
	{
		return NICSLU_MEMORY_OVERFLOW;
	}
	memset(ptr, 0, size);
	nicslu->workspace_mt2 = (void **)ptr;

	size = (sizeof(int__t)*4+sizeof(real__t)) * n;
	ptr = malloc(size*threads);
	if (NULL == ptr)
	{
		return NICSLU_MEMORY_OVERFLOW;
	}
	for (i=0; i<threads; ++i)
	{
		nicslu->workspace_mt1[i] = ptr;
		ptr = ((byte__t *)ptr) + size;
	}

	size = (sizeof(int__t)*3) * n;
	ptr = malloc(size*threads);
	if (NULL == ptr)
	{
		return NICSLU_MEMORY_OVERFLOW;
	}
	for (i=0; i<threads; ++i)
	{
		nicslu->workspace_mt2[i] = ptr;
		ptr = ((byte__t *)ptr) + size;
	}

	/*create threads*/
	for (i=1; i<threads; ++i)
	{
		if (_CreateThread( \
			_I_NicsLU_ThreadProc, \
			&(((SNicsLUThreadArg *)nicslu->thread_arg)[i]), \
			&(((thread_id__t *)nicslu->thread_id)[i]) \
			) != 0)
		{
			return NICSLU_THREADS_INIT_FAIL;
		}
	}
	((thread_id__t *)nicslu->thread_id)[0] = _GetCurrentThread();

	/*finish*/
	nicslu->flag[3] = TRUE;

	return NICS_OK;
}

int NicsLU_BindThreads(SNicsLU *nicslu, bool__t reset)
{
	int err;
	thread_id__t *id;
	unsigned int i, threads;

	if (NULL == nicslu)
	{
		return NICSLU_ARGUMENT_ERROR;
	}
	if (!nicslu->flag[3])
	{
		return NICSLU_THREADS_NOT_INITIALIZED;
	}

	err = NICS_OK;
	id = (thread_id__t *)(nicslu->thread_id);
	threads = (unsigned int)(nicslu->cfgi[5]);

	if (!reset)/*set*/
	{
		unsigned int core;
		for (i=0; i<threads; ++i)
		{
			core = i;
			err = _BindThreadToCores(id[i], &core, 1);
		}

		return (err==0 ? NICS_OK : NICSLU_BIND_THREADS_FAIL);
	}
	else/*reset*/
	{
		for (i=0; i<threads; ++i)
		{
			err = _UnbindThreadFromCores(id[i]);
		}

		return (err==0 ? NICS_OK : NICSLU_BIND_THREADS_FAIL);
	}
}

int NicsLU_DestroyThreads(SNicsLU *nicslu)
{
	bool__t mt;
	thread_id__t *thid;
	bool__t *thac;
	unsigned int i, th;

	if (NULL == nicslu)
	{
		return NICSLU_ARGUMENT_ERROR;
	}

	th = (unsigned int)(nicslu->cfgi[5]);
	mt = nicslu->flag[3];
	thid = (thread_id__t *)(nicslu->thread_id);
	thac = nicslu->thread_active;

	if (mt && th > 1)
	{
		/*set all threads active and wait to exit*/
		nicslu->thread_work = NICSLU_WORK_EXIT;
		for (i=1; i<th; ++i)
		{
			thac[i] = TRUE;
		}
		for (i=1; i<th; ++i)
		{
			_WaitThreadExit(thid[i]);
		}
	}
	nicslu->flag[3] = FALSE;
	nicslu->cfgi[5] = 1;
	nicslu->cfgi[7] = 1;
	nicslu->thread_work = NICSLU_WORK_NONE;

	if (nicslu->thread_id != NULL)
	{
		free(nicslu->thread_id);
		nicslu->thread_id = NULL;
	}
	if (nicslu->thread_arg != NULL)
	{
		free(nicslu->thread_arg);
		nicslu->thread_arg = NULL;
	}
	if (nicslu->thread_finish != NULL)
	{
		free(nicslu->thread_finish);
		nicslu->thread_finish = NULL;
	}
	if (nicslu->cluster_start != NULL)
	{
		free(nicslu->cluster_start);
		nicslu->cluster_start = NULL;
	}
	nicslu->cluster_end = NULL;
	if (nicslu->workspace_mt1 != NULL)
	{
		if (nicslu->workspace_mt1[0] != NULL)
		{
			free(nicslu->workspace_mt1[0]);
		}
		free(nicslu->workspace_mt1);
		nicslu->workspace_mt1 = NULL;
	}
	if (nicslu->workspace_mt2 != NULL)
	{
		if (nicslu->workspace_mt2[0] != NULL)
		{
			free(nicslu->workspace_mt2[0]);
		}
		free(nicslu->workspace_mt2);
		nicslu->workspace_mt2 = NULL;
	}
	nicslu->thread_active = NULL;

	return NICS_OK;
}
