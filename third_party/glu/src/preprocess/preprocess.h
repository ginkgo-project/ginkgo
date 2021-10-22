#ifndef __PREPROCESS__
#define __PREPROCESS__

#include "../nicslu/include/nics_config.h"
#include "../nicslu/include/nicslu.h"
#include "../../include/type.h"

#define IN__
#define OUT__

#ifdef __cplusplus
extern "C" {
#endif

int DumpA(SNicsLU *nicslu, double *ax, unsigned int *ai, unsigned int *ap);

int preprocess( \
	IN__ char *matrixName, \
    IN__ SNicsLU *nicslu,\
	OUT__ double **ax, \
	OUT__ unsigned int **ai, \
	OUT__ unsigned int **ap);


#ifdef __cplusplus
}
#endif

#endif
