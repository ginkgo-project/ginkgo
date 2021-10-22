/*scale the matrix by the mc64 algorithm*/
/*last modified: june 7, 2013*/
/*author: Chen, Xiaoming*/

#include "nicslu.h"
#include "nicslu_internal.h"

void _I_NicsLU_MC64Scale(SNicsLU *nicslu)
{
	uint__t end;
	uint__t i, j;
	uint__t n;
	real__t *rs, *csp, *ax;
	uint__t *ai, *ap;
	real__t s;
#ifdef NICSLU_DEBUG
	FILE *fp;
#endif

	if (!nicslu->cfgi[1]) return;

	n = nicslu->n;
	ax = nicslu->ax;
	ai = nicslu->ai;/*changed in preprocess*/
	ap = nicslu->ap;
	rs = nicslu->row_scale;
	csp = nicslu->col_scale_perm;

#ifdef NICSLU_DEBUG
	fp = fopen("smat0.txt", "w");
	for (i=0; i<nicslu->nnz; ++i)
	{
		fprintf(fp, "%.16g\n", ax[i]);
	}
	fclose(fp);
#endif

	for (i=0; i<n; ++i)
	{
		s = rs[i];/*not permuted*/
		end = ap[i+1];/*not permuted*/
		for (j=ap[i]; j<end; ++j)
		{
			ax[j] *= s*csp[ai[j]];
		}
	}

#ifdef NICSLU_DEBUG
	fp = fopen("smat1.txt", "w");
	for (i=0; i<nicslu->nnz; ++i)
	{
		fprintf(fp, "%.16g\n", ax[i]);
	}
	fclose(fp);
#endif
}

void _I_NicsLU_MC64ScaleForRefact(SNicsLU *nicslu, real__t *ax0)
{
	uint__t end;
	uint__t i, j;
	uint__t n;
	real__t *rs, *csp, *ax;
	uint__t *ai, *ap;
	real__t s;

	if (!nicslu->cfgi[1]) return;

	n = nicslu->n;
	ax = nicslu->ax;
	ai = nicslu->ai;/*changed in preprocess*/
	ap = nicslu->ap;
	rs = nicslu->row_scale;
	csp = nicslu->col_scale_perm;

	for (i=0; i<n; ++i)
	{
		s = rs[i];/*not permuted*/
		end = ap[i+1];/*not permuted*/
		for (j=ap[i]; j<end; ++j)
		{
			ax[j] = ax0[j] * s * csp[ai[j]];
		}
	}
}
