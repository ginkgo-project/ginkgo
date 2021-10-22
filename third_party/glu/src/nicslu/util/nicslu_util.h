#ifndef __NICSLU_UTIL__
#define __NICSLU_UTIL__

#include "nics_config.h"

#define IN__
#define OUT__

#ifdef __cplusplus
extern "C" {
#endif

/*read triplet format to CSR or CSC, triplets are stored in column-wise*/
/*matrix from The University of Florida Sparse Matrix Collection is stored in this format*/
int \
	NicsLU_ReadTripletColumnToSparse( \
	IN__ char *file, \
	OUT__ uint__t *n, \
	OUT__ uint__t *nnz, \
	OUT__ real__t **ax, \
	OUT__ uint__t **ai, \
	OUT__ uint__t **ap);

/*read triplet format to CSR or CSC, triplets are stored in row-wise*/
int \
	NicsLU_ReadTripletRowToSparse( \
	IN__ char *file, \
	OUT__ uint__t *n, \
	OUT__ uint__t *nnz, \
	OUT__ real__t **ax, \
	OUT__ uint__t **ai, \
	OUT__ uint__t **ap);

#ifdef __cplusplus
}
#endif

#endif
