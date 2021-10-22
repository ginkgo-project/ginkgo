/*static symbolic factorization*/
/*last modified: august 27, 2013*/
/*author: Chen, Xiaoming*/

#include "nicslu.h"
#include "nicslu_internal.h"
#include "system.h"
#include "math.h"
#include "nicslu_default.h"

extern size_t g_sp, g_sd, g_si;

int _I_NicsLU_StaticSymbolicFactorize(SNicsLU *nicslu)
{
	int err;
	uint__t *ai, *ap;
	uint__t cend, oldrow;
	uint__t *rowperm;
	uint__t n, i, top, j, col, p, ucol, lcol, k;
	int__t *flag, *pend, *stack, *appos;
	size_t est, used, tmp;
	uint__t *lu;
	uint__t unnz, *urow, *urowp, lnnz, *lrow;
	int__t head, pos, tail;
	uint__t *ulen, *llen;
	size_t *up;
	real__t memmul;
	size_t *len_est;
	uint__t prow;
	uint__t ll, ul;
	size_t tl;
	void **lu2;
	void *ptr;
	size_t *wkld;
	uint__t *lip;
	size_t flop;
	real__t ed, min, pflops, *end, *thfi;
	unsigned int minid, cores, tj;
	real__t sync, grow;

	err = NICS_OK;
	ai = nicslu->ai;
	ap = nicslu->ap;
	rowperm = nicslu->row_perm;
	n = nicslu->n;
	flag = (int__t *)(nicslu->workspace);
	pend = flag + n;
	stack = pend + n;
	appos = stack + n;
	len_est = nicslu->len_est;
	lu2 = nicslu->lu_array2;
	wkld = nicslu->wkld_est;
	grow = nicslu->cfgf[5];
	if (grow <= 1.)
	{
		grow = NICSLU_MEMORY_GROW;
		nicslu->cfgf[5] = grow;
	}

	flop = 0;
	memset(flag, 0xff, sizeof(int__t)*(n+n));
	est = nicslu->lu_nnz_est;
	used = 0;

	lu = NULL;
	llen = NULL;
	up = NULL;
	lu = (uint__t *)malloc(sizeof(uint__t)*est);
	if (NULL == lu)
	{
		err = NICSLU_MEMORY_OVERFLOW;
		goto FAIL;
	}
	llen = (uint__t *)malloc(sizeof(uint__t)*(n+n));
	if (NULL == llen)
	{
		err = NICSLU_MEMORY_OVERFLOW;
		goto FAIL;
	}
	ulen = llen + n;
	up = (size_t *)malloc(sizeof(size_t)*n);
	if (NULL == up)
	{
		err = NICSLU_MEMORY_OVERFLOW;
		goto FAIL;
	}

	up[0] = 0;
	for (i=0; i<n; ++i)
	{
		oldrow = rowperm[i];
		cend = ap[oldrow+1];

		/*length*/
		tmp = used + n;
		if (tmp > est)
		{
			est *= grow;
			if (tmp > est)
			{
				est = (size_t)(tmp * grow);
			}
			ptr = realloc(lu, sizeof(uint__t)*est);
			if (NULL == ptr)
			{
				err = NICSLU_MEMORY_OVERFLOW;
				goto FAIL;
			}
			lu = (uint__t *)ptr;
		}

		top = n;
		unnz = 0;
		urow = lu + up[i];

		/*symbolic factorization*/
		for (j=ap[oldrow]; j<cend; ++j)
		{
			col = ai[j];

			if (flag[col] != i)
			{
				if (col < i)/*dfs*/
				{
					head = 0;
					stack[0] = col;

					while (head >= 0)
					{
						p = stack[head];

						if (flag[p] != i)
						{
							flag[p] = i;
							appos[head] = ((pend[p]<0) ? ulen[p] : pend[p]);
						}

						urowp = lu + up[p];

						for (pos=--appos[head]; pos>=0; --pos)
						{
							ucol = urowp[pos];
							if (flag[ucol] != i)
							{
								if (ucol < i)/*dfs*/
								{
									appos[head] = pos;
									stack[++head] = ucol;
									break;
								}
								else if (ucol > i)/*U*/
								{
									flag[ucol] = i;
									urow[unnz++] = ucol;
								}
							}
						}

						if (pos < 0)
						{
							--head;
							stack[--top] = p;
						}
					}
				}
				else if (col > i)/*U*/
				{
					flag[col] = i;
					urow[unnz++] = col;
				}
			}
		}

		ulen[i] = unnz;
		lnnz = llen[i] = n - top;

		for (j=top; j<n; ++j)
		{
			urow[unnz++] = stack[j];/*l, unnz=rownnz*/
		}

		if (i+1 < n)
		{
			used = up[i+1] = up[i] + unnz;
		}
		else
		{
			used = up[i] + unnz;
		}

		/*prune*/
		lrow = urow + ulen[i];

		for (j=0; j<lnnz; ++j)
		{
			lcol = lrow[j];

			if (pend[lcol] < 0)
			{
				unnz = ulen[lcol];
				urowp = lu + up[lcol];

				for (k=0; k<unnz; ++k)
				{
					if (urowp[k] == i)
					{
						head = 0;
						tail = unnz;

						while (head < tail)
						{
							p = urowp[head];
							if (p <= i)
							{
								++head;
							}
							else
							{
								--tail;
								urowp[head] = urowp[tail];
								urowp[tail] = p;
							}
						}

						pend[lcol] = tail;
						break;
					}
				}
			}
		}
	}

	nicslu->stat[11] = (real__t)(used + n);

	memmul = nicslu->cfgf[1];
	if (memmul < 1.)
	{
		memmul = 1.;
		nicslu->cfgf[1] = memmul;
	}
	prow = nicslu->cfgi[4];
	if (prow == 0)
	{
		prow = 1;
		nicslu->cfgi[4] = prow;
	}

	/*total flops & alloc memory for lu_array2*/
	for (i=0; i<n; ++i)
	{
		ll = llen[i];
		ul = ulen[i];

		tl = 0;
		lip = lu + up[i] + ul;
		for (j=0; j<ll; ++j)
		{
			tl += (ulen[lip[j]]<<1);
		}
		wkld[i] = tl + ul;
		flop += wkld[i];

		if (ll < prow) ll = prow;
		if (ul < prow) ul = prow;

		ll *= memmul;
		ul *= memmul;

		if (ll > i) ll = i;
		if (ul > n-i) ul = n - i;

		len_est[i] = tl = ul*g_si + (ll+ul)*g_sp;

		ptr = malloc(tl);
		if (NULL == ptr)
		{
			err = NICSLU_MEMORY_OVERFLOW;
			goto FAIL;
		}
		lu2[i] = ptr;
	}
	nicslu->stat[10] = (real__t)flop;

	/*predict speedup*/
	end = (real__t *)nicslu->workspace;
	cores = (unsigned int)(nicslu->cfgi[8]);
	sync = nicslu->cfgf[3];
	if (cores > (unsigned int)n)/*an unusual case, the matrix is too small*/
	{
		nicslu->stat[7] = 0.;
	}
	else if (cores == 1)
	{
		nicslu->stat[7] = 1.;
	}
	else
	{
		thfi = end + n;
		memset(end, 0, sizeof(real__t)*(n+n));

		pflops = 0.;
		for (i=0; i<n; ++i)
		{
			min = DBL_MAX;
			for (tj=0; tj<cores; ++tj)
			{
				if (thfi[tj] < min)
				{
					min = thfi[tj];
					minid = tj;
				}
			}

			ll = llen[i];
			ul = ulen[i];
			lip = lu + up[i] + ul;

			ed = min;
			for (j=0; j<ll; ++j)
			{
				k = lip[j];
				ed = MAX(ed, end[k]);
				ed += (ulen[k]<<1);
				ed += sync;
			}
			ed += ul;
			ed += sync;
			end[i] = ed;
			thfi[minid] = ed;

			if (ed > pflops)
			{
				pflops = ed;
			}
		}

		if (pflops == 0.) nicslu->stat[7] = (real__t)n;
		else nicslu->stat[7] = nicslu->stat[10] / pflops;
	}

	/*maximum estimated speedup*/
	memset(end, 0, sizeof(real__t)*n);

	pflops = 0.;
	for (i=0; i<n; ++i)
	{
		ll = llen[i];
		ul = ulen[i];
		lip = lu + up[i] + ul;

		ed = 0.;
		for (j=0; j<ll; ++j)
		{
			k = lip[j];
			ed = MAX(ed, end[k]);
			ed += (ulen[k] << 1);
		}
		ed += ul;
		end[i] = ed;

		if (ed > pflops)
		{
			pflops = ed;
		}
	}

	if (pflops == 0.) nicslu->stat[8] = (real__t)n;
	else nicslu->stat[8] = nicslu->stat[10] / pflops;

	free(lu);
	free(llen);
	free(up);
	return NICS_OK;

FAIL:
	if (lu != NULL) free(lu);
	if (llen != NULL) free(llen);
	if (up != NULL) free(up);
	return err;
}
