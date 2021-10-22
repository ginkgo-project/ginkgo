/*destroy the NicsLU structure*/
/*last modified: june 7, 2013*/
/*author: Chen, Xiaoming*/

#include "nicslu.h"
#include "nicslu_internal.h"

int NicsLU_Destroy(SNicsLU *nicslu)
{
	if (NULL == nicslu)
	{
		return NICSLU_ARGUMENT_ERROR;
	}

	_I_NicsLU_DestroyMatrix(nicslu);

	free(nicslu->timer);
	nicslu->timer = NULL;

	free(nicslu->flag);
	nicslu->flag = NULL;

	free(nicslu->stat);
	nicslu->stat = NULL;

	free(nicslu->cfgi);
	nicslu->cfgi = NULL;

	free(nicslu->cfgf);
	nicslu->cfgf = NULL;

	return NICS_OK;
}
