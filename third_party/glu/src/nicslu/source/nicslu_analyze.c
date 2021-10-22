/*preprocess, including static pivoting and AMD ordering*/
/*last modified: june 11, 2013*/
/*author: Chen, Xiaoming*/

#include "nicslu.h"
#include "nicslu_internal.h"
#include "timer_c.h"
#include "math.h"

extern size_t g_sp;

int NicsLU_Analyze(SNicsLU *nicslu)
{
	int__t rank;
	int err;
	int__t *w;
	int__t *match, *match2;
	real__t *dw;
	uint__t n, nnz;
	real__t *ax;
	uint__t *ai, *ap;
	size_t size;

	uint__t *cp0;
	uint__t *ci0;
	uint__t *rp;
	uint__t *ri;
	uint__t *cp;
	uint__t *ci;
	int__t *len;
	uint__t nzaat;
	int__t *pe, *sp, *iw, *tp;
	int__t *p, *pinv;

	uint__t j;
	real__t t, *u, *v;

	real__t *rs, *csi;
	uint__t *cpi;
	uint__t i;
	uint__t *p1, *p2, *p3, *p4;

#ifdef NICSLU_DEBUG
	FILE *fp;
#endif

	if (NULL == nicslu)
	{
		return NICSLU_ARGUMENT_ERROR;
	}
	if (nicslu->flag[1]) return NICS_OK;
	if (!nicslu->flag[0])
	{
		return NICSLU_MATRIX_NOT_INITIALIZED;
	}

	/*begin*/
	TimerStart((STimer *)(nicslu->timer));

	/*set parameters*/
	rank = 0;
	err = NICS_OK;
	w = NULL;
	match = NULL;
	match2 = NULL;
	dw = NULL;
	n = nicslu->n;
	nnz = nicslu->nnz;
	ax = nicslu->ax;
	ai = nicslu->ai;
	ap = nicslu->ap;

	/*init*/
	p1 = nicslu->row_perm;
	p2 = nicslu->row_perm_inv;
	p3 = nicslu->col_perm;
	p4 = nicslu->col_perm_inv;
	rs = nicslu->row_scale;
	csi = nicslu->col_scale_perm;
	for (i=0; i<n; ++i)
	{
		p1[i] = p2[i] = p3[i] = p4[i] = i;
		rs[i] = csi[i] = 1.;
	}

	/**************************************************************************************/
	/*maximum matching*/
	size = n*5*sizeof(int__t);
	w = (int__t *)malloc(size);
	if (NULL == w)
	{
		return NICSLU_MEMORY_OVERFLOW;
	}

	size = sizeof(int__t) * (n+n);
	match = (int__t *)malloc(size);
	if (NULL == match)
	{
		free(w);
		return NICSLU_MEMORY_OVERFLOW;
	}
	match2 = match+n;

	/*workspace memory states
	 |-----|-----|-----|-----|-----|
	 flag(w) cheap  is   js    ps
	 w
	*/

	size = sizeof(real__t)*(3*n+nnz);
	dw = (real__t *)malloc(size);
	if (NULL == dw)
	{
		free(w);
		free(match);
		return NICSLU_MEMORY_OVERFLOW;
	}
	/*dw workspace
	      |-----|-----|-----|----------|
	       u     v     max   abs
		   */

	/*mc64*/
	rank = _I_NicsLU_MC64ad(n, nnz, ai, ap, ax, match, match2, 5*n, w, 3*n+nnz, dw);

	if (FAIL(rank))
	{
		free(w);
		free(match);
		if (dw != NULL) free(dw);
		return (int)rank;
	}

#ifdef NICSLU_DEBUG
	printf("final match: %d\n", rank);

	fp = fopen("match.txt", "w");
	for (i=0; i<n; ++i)
	{
		fprintf(fp, "match[%d] = %d\n", i, match[i]);
	}
	for (i=0; i<n; ++i)
	{
		fprintf(fp, "match2[%d] = %d\n", i, match2[i]);
	}
	fclose(fp);
#endif

	if ((uint__t)rank < n)/*not full rank*/
	{
		free(w);
		free(match);
		if (dw != NULL) free(dw);
		return NICSLU_MATRIX_STRUCTURAL_SINGULAR;
	}

	/**************************************************************************************/
	/*amd*/
	cp0 = NULL;
	ci0 = NULL;
	rp = NULL;
	ri = NULL;
	cp = NULL;
	ci = NULL;

	err = _I_NicsLU_Check(nicslu);

	if (FAIL(err))
	{
		free(w);
		free(match);
		if (dw != NULL) free(dw);
		return err;
	}

	size = sizeof(uint__t)*nnz;
	ci0 = (uint__t *)malloc(size);
	if (ci0 == NULL)
	{
		free(w);
		free(match);
		if (dw != NULL) free(dw);
		return NICSLU_MEMORY_OVERFLOW;
	}

	size = sizeof(uint__t)*(n+1);
	cp0 = (uint__t *)malloc(size);
	if (cp0 == NULL)
	{
		free(ci0);
		free(w);
		free(match);
		if (dw != NULL) free(dw);
		return NICSLU_MEMORY_OVERFLOW;
	}

	_I_NicsLU_ConstructCSR(nicslu, match, ci0, cp0);

	if (NICSLU_MATRIX_NOT_SORTED == err)/*need sort*/
	{
		size = sizeof(uint__t)*nnz;
		ri = (uint__t *)malloc(size);
		if (ri == NULL)
		{
			free(ci0);
			free(cp0);
			free(w);
			free(match);
			if (dw != NULL) free(dw);
			return NICSLU_MEMORY_OVERFLOW;
		}

		size = sizeof(uint__t)*(n+1);
		rp = (uint__t *)malloc(size);
		if (rp == NULL)
		{
			free(ci0);
			free(cp0);
			free(ri);
			free(w);
			free(match);
			if (dw != NULL) free(dw);
			return NICSLU_MEMORY_OVERFLOW;
		}

		_I_NicsLU_AMDSort(n, ci0, cp0, w, ri, rp);

		ci = ri;
		cp = rp;

		free(ci0);
		free(cp0);
	}
	else/*no sort*/
	{
		ci = ci0;
		cp = cp0;
	}

	/*a+a'*/
	size = sizeof(int__t)*n;
	len = (int__t *)malloc(size);
	if (NULL == len)
	{
		free(ci);
		free(cp);
		free(w);
		free(match);
		if (dw != NULL) free(dw);
		return NICSLU_MEMORY_OVERFLOW;
	}

	/*the nonzeros in each row of a+a' (exclu diag)*/
	nzaat = _I_NicsLU_AAT(n, ci, cp, len, w);

	free(w);

	size = sizeof(int__t)*(((size_t)n)*7+nzaat+nzaat/5);
	w = (int__t *)malloc(size);
	if (NULL == w)
	{
		free(match);
		if (dw != NULL) free(dw);
		free(ci);
		free(cp);
		free(len);
		return NICSLU_MEMORY_OVERFLOW;
	}

	/*workspace memory states
	       |----|----|----|----|----|----|----|----------|
	       pe   nv   head elen degree w   iw
		   */

	pe = w;
	sp = w+n;
	iw = w+6*n;
	tp = w+5*n;

	/*construct A+A'*/
	_I_NicsLU_AAT2(n, ci, cp, len, pe, sp, iw, tp);

	free(ci);
	free(cp);

	size = sizeof(int__t) * (n+n);
	p = (int__t *)malloc(size);
	if (NULL == p)
	{
		free(w);
		free(match);
		if (dw != NULL) free(dw);
		free(len);
		return NICSLU_MEMORY_OVERFLOW;
	}
	pinv = p + n;

	_I_NicsLU_AMD(n, nzaat, n+nzaat+nzaat/5, pe, iw, len, w, p, pinv, \
		nicslu->cfgf[2], nicslu->cfgi[6], &(nicslu->lu_nnz_est));
	if (nicslu->lu_nnz_est < (size_t)nnz) nicslu->lu_nnz_est = nnz;

	free(len);

#ifdef NICSLU_DEBUG
	fp = fopen("permute.txt", "w");
	for (i=0; i<n; ++i)
	{
		fprintf(fp, "%d\t%d\t%d\n", i, p[i], pinv[i]);
	}
	fclose(fp);
	printf("lu  %u\n", nicslu->lu_nnz_est);
#endif

	_I_NicsLU_Permute(nicslu, match, p, pinv);

	free(match);
	free(p);
	free(w);

	/*mc64_scale*/
	u = dw;/*col*/
	v = u + n;/*row*/

	rs = nicslu->row_scale;
	csi = nicslu->col_scale_perm;
	cpi = nicslu->col_perm_inv;

	for (j=0; j<n; ++j)
	{
		t = exp(v[j]);
		rs[j] = t;

		t = exp(u[j]);
		csi[cpi[j]] = t;
	}

	_I_NicsLU_MC64Scale(nicslu);
	err = _I_NicsLU_Scale(nicslu);
	if (FAIL(err))
	{
		if (dw != NULL) free(dw);
		return err;
	}

	if (dw != NULL) free(dw);

#ifdef NICSLU_DEBUG
	fp = fopen("mc64_scale.txt", "w");
	for (i=0; i<n; ++i)
	{
		fprintf(fp, "%.16g  %.16g\n", nicslu->row_scale[i], nicslu->col_scale_perm[i]);
	}
	fclose(fp);
#endif

	/**************************************************************************************/
	/*finish*/
	nicslu->flag[1] = TRUE;

	TimerStop((STimer *)(nicslu->timer));
	nicslu->stat[0] = TimerGetRuntime((STimer *)(nicslu->timer));

	return NICS_OK;
}
