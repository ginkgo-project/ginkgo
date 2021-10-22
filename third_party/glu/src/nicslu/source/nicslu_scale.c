/*scale the matrix*/
/*last modified: june 7, 2013*/
/*author: Chen, Xiaoming*/

#include "nicslu.h"
#include "nicslu_internal.h"

int _I_NicsLU_Scale(SNicsLU *nicslu)
{
	uint__t n, i, j, col;
	uint__t scale;
	real__t *cscale;
	real__t val;
	real__t *ax;
	uint__t *ai, *ap;
	uint__t cend, oldrow;
	uint__t *rowperm;

	scale = nicslu->cfgi[2];
	if (scale == 0 || scale > 2) return NICS_OK;

	n = nicslu->n;
	cscale = nicslu->cscale;
	
	ax = nicslu->ax;
	ai = nicslu->ai;
	ap = nicslu->ap;
	rowperm = nicslu->row_perm;

	memset(cscale, 0, sizeof(real__t)*n);

	if (scale == 1)/*maximum*/
	{
		for (i=0; i<n; ++i)
		{
			oldrow = rowperm[i];
			cend = ap[oldrow+1];
			for (j=ap[oldrow]; j<cend; ++j)
			{
				col = ai[j];
				val = ax[j];
				if (val < 0.) val = -val;
				if (val > cscale[col])
				{
					cscale[col] = val;
				}
			}
		}
	}
	else if (scale == 2)/*sum*/
	{
		for (i=0; i<n; ++i)
		{
			oldrow = rowperm[i];
			cend = ap[oldrow+1];
			for (j=ap[oldrow]; j<cend; ++j)
			{
				col = ai[j];
				val = ax[j];
				if (val < 0.) val = -val;
				cscale[col] += val;
			}
		}
	}

	for (i=0; i<n; ++i)
	{
		if (cscale[i] == 0.) return NICSLU_MATRIX_NUMERIC_SINGULAR;
	}

	return NICS_OK;
}
