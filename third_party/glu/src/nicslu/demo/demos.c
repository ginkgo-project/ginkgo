#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nicslu.h"
#include "nicslu_util.h"

int main(void)
{
	int ret;
	uint__t n, nnz, i;
	real__t *ax;
	uint__t *ai, *ap;
	SNicsLU *nicslu;
	real__t *x, *b, err;
	ax = NULL;
	ai = NULL;
	ap = NULL;
	x = NULL;
	b = NULL;

	nicslu = (SNicsLU *)malloc(sizeof(SNicsLU));
	NicsLU_Initialize(nicslu);

	ret = NicsLU_ReadTripletColumnToSparse("ASIC_100k.mtx", &n, &nnz, &ax, &ai, &ap);
	if (ret != NICS_OK) goto EXIT;

	x = (real__t *)malloc(sizeof(real__t)*(n+n));
	b = x + n;
	for (i=0; i<n+n; ++i) x[i] = 1.;

	NicsLU_CreateMatrix(nicslu, n, nnz, ax, ai, ap);
	nicslu->cfgf[0] = 1.;

	NicsLU_Analyze(nicslu);
	printf("analysis time: %.8g\n", nicslu->stat[0]);

	NicsLU_Factorize(nicslu);
	printf("factorization time: %.8g\n", nicslu->stat[1]);

	NicsLU_ReFactorize(nicslu, ax);
	printf("re-factorization time: %.8g\n", nicslu->stat[2]);

	NicsLU_Solve(nicslu, x);
	printf("substitution time: %.8g\n", nicslu->stat[3]);

	NicsLU_Residual(n, ax, ai, ap, x, b, &err, 1, 0);
	printf("Ax-b (1-norm): %.8g\n", err);
	NicsLU_Residual(n, ax, ai, ap, x, b, &err, 2, 0);
	printf("Ax-b (2-norm): %.8g\n", err);
	NicsLU_Residual(n, ax, ai, ap, x, b, &err, 0, 0);
	printf("Ax-b (infinite-norm): %.8g\n", err);

	printf("NNZ(L+U-I): %ld\n", nicslu->lu_nnz);

	NicsLU_Flops(nicslu, NULL);
	NicsLU_Throughput(nicslu, NULL);
	NicsLU_ConditionNumber(nicslu, NULL);
	printf("flops: %.8g\n", nicslu->stat[5]);
	printf("throughput (bytes): %.8g\n", nicslu->stat[12]);
	printf("condition number: %.8g\n", nicslu->stat[6]);
	NicsLU_MemoryUsage(nicslu, NULL);
	printf("memory (Mbytes): %.8g\n", nicslu->stat[21]/1024./1024.);
	
EXIT:
	NicsLU_Destroy(nicslu);
	free(ax);
	free(ai);
	free(ap);
	free(nicslu);
	free(x);
	return 0;
}
