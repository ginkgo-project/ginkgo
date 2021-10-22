/*thread process function*/
/*last modified: june 7, 2013*/
/*author: Chen, Xiaoming*/

#include "nicslu.h"
#include "nicslu_internal.h"
#include "thread.h"

THREAD_DECL _I_NicsLU_ThreadProc(void *tharg)
{
	SNicsLUThreadArg *arg;
	SNicsLU *nicslu;
	unsigned int id;
	bool__t *ac;

	arg = (SNicsLUThreadArg *)tharg;
	nicslu = arg->nicslu;
	id = arg->id;
	ac = &(nicslu->thread_active[id]);

	while (TRUE)
	{
		_SpinWaitChar((volatile char *)ac);
		*ac = FALSE;

		switch (nicslu->thread_work)
		{
		case NICSLU_WORK_EXIT:
			goto RETURN;
			break;

		case NICSLU_WORK_FACT_CLUSTER:
			_I_NicsLU_Factorize_Cluster(nicslu, arg, id);
			break;

		case NICSLU_WORK_FACT_PIPELINE:
			_I_NicsLU_Factorize_Pipeline(nicslu, arg, id);
			break;

		case NICSLU_WORK_REFACT_CLUSTER:
			_I_NicsLU_ReFactorize_Cluster(nicslu, arg, id);
			break;

		case NICSLU_WORK_REFACT_PIPELINE:
			_I_NicsLU_ReFactorize_Pipeline(nicslu, arg, id);
			break;

	/*	case NICSLU_WORK_COPY_DATA:
			_I_NicsLU_CopyData(nicslu, id);
			break;*/

		case NICSLU_WORK_NONE:
		default:
			break;
		}
	}

RETURN:
	return THREAD_RETURN;
}
