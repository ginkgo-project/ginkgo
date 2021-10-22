/*initialize the NicsLU structure*/
/*last modified: june 7, 2013*/
/*author: Chen, Xiaoming*/

#include "nicslu.h"
#include "nicslu_internal.h"
#include "timer_c.h"
#include "system.h"
#include "nicslu_default.h"

int NicsLU_Initialize(SNicsLU *nicslu)
{
	if (NULL == nicslu)
	{
		return NICSLU_ARGUMENT_ERROR;
	}

	/*timer*/
	nicslu->timer = malloc(sizeof(STimer));
	if (TimerInit((STimer *)(nicslu->timer)) != 0) return NICSLU_GENERAL_FAIL;

	/*flags*/
	nicslu->flag = (bool__t *)malloc(sizeof(bool__t)*32);
	memset(nicslu->flag, 0, sizeof(bool__t)*32);

	/*statistics*/
	nicslu->stat = (real__t *)malloc(sizeof(real__t)*32);
	memset(nicslu->stat, 0, sizeof(real__t)*32);

	/*configurations*/
	nicslu->cfgi = (uint__t *)malloc(sizeof(uint__t)*32);
	nicslu->cfgf = (real__t *)malloc(sizeof(real__t)*32);
	memset(nicslu->cfgi, 0, sizeof(uint__t)*32);
	memset(nicslu->cfgf, 0, sizeof(real__t)*32);
	nicslu->cfgi[0] = 0;/*row/column mode*/
	nicslu->cfgi[1] = 1;/*mc64*/
	nicslu->cfgi[2] = 0;/*scale*/
	nicslu->cfgi[3] = NICSLU_PIPELINE_THRESHOLD;
	nicslu->cfgi[4] = NICSLU_STATIC_RNNZ_UB;
	nicslu->cfgi[5] = 1;/*threads created*/
	nicslu->cfgi[6] = NICSLU_AMD_FLAG1;
	nicslu->cfgi[7] = 1;/*threads used*/
	nicslu->cfgf[0] = NICSLU_PIVOT_TOLERANCE;
	nicslu->cfgf[1] = NICSLU_STATIC_MEMORY_MULT;
	nicslu->cfgf[2] = NICSLU_AMD_FLAG2;
	nicslu->cfgf[3] = NICSLU_SYNC_CYCLES;
	nicslu->cfgf[4] = NICSLU_LOAD_BALANCE;
	nicslu->cfgf[5] = NICSLU_MEMORY_GROW;

	nicslu->cfgi[8] = GetProcessorNumber();
	nicslu->stat[9] = (real__t)(nicslu->cfgi[8]);

	/*matrix data, 6 items*/
	nicslu->n = 0;
	nicslu->nnz = 0;
	nicslu->ax = NULL;
	nicslu->ai = NULL;
	nicslu->ap = NULL;
	nicslu->rhs = NULL;

	/*other matrix data, 9 items*/
	nicslu->row_perm = NULL;
	nicslu->row_perm_inv = NULL;
	nicslu->col_perm = NULL;
	nicslu->col_perm_inv = NULL;
	nicslu->col_scale_perm = NULL;
	nicslu->row_scale = NULL;
	nicslu->cscale = NULL;
	nicslu->pivot = NULL;
	nicslu->pivot_inv = NULL;

	/*lu matrix, 13 items*/
	nicslu->lu_nnz_est = 0;
	nicslu->lu_nnz = 0;
	nicslu->l_nnz = 0;
	nicslu->u_nnz = 0;
	nicslu->ldiag = NULL;
	nicslu->lu_array = NULL;
	nicslu->up = NULL;
	nicslu->llen = NULL;
	nicslu->ulen = NULL;
	nicslu->len_est = NULL;
	nicslu->wkld_est = NULL;
	nicslu->row_state = NULL;
	nicslu->lu_array2 = NULL;

	/*work space, 3 items*/
	nicslu->workspace = NULL;
	nicslu->workspace_mt1 = NULL;
	nicslu->workspace_mt2 = NULL;

	/*for parallelism, 10 items*/
	nicslu->last_busy = 0;
	nicslu->thread_id = NULL;
	nicslu->thread_arg = NULL;
	nicslu->thread_finish = NULL;
	nicslu->thread_active = NULL;
	nicslu->thread_work = NICSLU_WORK_NONE;
	nicslu->cluster_start = NULL;
	nicslu->cluster_end = NULL;
	nicslu->pipeline_start = 0;
	nicslu->pipeline_end = 0;

	/*aegraph, 6 items*/
	nicslu->aeg_level = 0;
	nicslu->aeg_data = NULL;
	nicslu->aeg_header = NULL;
	nicslu->aeg_refact_level = 0;
	nicslu->aeg_refact_data = NULL;
	nicslu->aeg_refact_header = NULL;

	/*end*/
	return NICS_OK;
}
