/*calculate flops of numeric factorization*/
/*last modified: june 11, 2013*/
/*author: Chen, Xiaoming*/

#include "nicslu.h"
#include "nicslu_internal.h"

extern size_t g_si, g_sd, g_sp;

int NicsLU_Flops(SNicsLU *nicslu, real__t *op)
{
	uint__t n, i, j;
	real__t fop;
	void *lu;
	size_t *up;
	uint__t *ulen, *llen;
	uint__t *lip;
	uint__t ll, ul;

	if (NULL == nicslu)
	{
		return NICSLU_ARGUMENT_ERROR;
	}
	if (!nicslu->flag[2])
	{
		return NICSLU_MATRIX_NOT_FACTORIZED;
	}
	
	n = nicslu->n;
	fop = 0;
	lu = nicslu->lu_array;
	up = nicslu->up;
	ulen = nicslu->ulen;
	llen = nicslu->llen;

	for (i=0; i<n; ++i)
	{
		ll = llen[i];
		ul = ulen[i];

		lip = (uint__t *)(((byte__t *)lu) + up[i] + g_sp*ul);

		for (j=0; j<ll; ++j)
		{
			fop += (ulen[lip[j]] << 1);
		}

		fop += ul;
	}

	nicslu->stat[5] = fop;
	if (op != NULL) *op = fop;

	return NICS_OK;
}

int NicsLU_ThreadLoad(SNicsLU *nicslu, unsigned int threads, real__t **flops)
{
	uint__t j, k, p, t;
	void *lu;
	size_t *up;
	uint__t *ulen, *llen;
	uint__t ul, ll;
	uint__t *lip;
	uint__t clv, level, len, lv;
	uint__t *aeg_head, *aeg_data;
	int mode;
	uint__t thres;
	uint__t lstart, lend, aegh, start, end;
	size_t *wkld, total, sub, avg;
	unsigned int i;
	real__t bal;

	if (NULL == nicslu || NULL == flops)
	{
		return NICSLU_ARGUMENT_ERROR;
	}
	if (!nicslu->flag[2])
	{
		return NICSLU_MATRIX_NOT_FACTORIZED;
	}
	if (!nicslu->flag[4])
	{
		return NICSLU_SCHEDULER_NOT_INITIALIZED;
	}
	if (threads == 0)
	{
		return NICSLU_ARGUMENT_ERROR;
	}
	if (*flops != NULL)
	{
		free(*flops);
		*flops = NULL;
	}

	lu = nicslu->lu_array;
	up = nicslu->up;
	ulen = nicslu->ulen;
	llen = nicslu->llen;
	level = nicslu->aeg_level;
	aeg_head = nicslu->aeg_header;
	aeg_data = nicslu->aeg_data;
	thres = nicslu->cfgi[3];
	wkld = nicslu->wkld_est;
	bal = nicslu->cfgf[4];

	*flops = (real__t *)malloc(sizeof(real__t)*threads);
	if (NULL == *flops)
	{
		return NICSLU_MEMORY_OVERFLOW;
	}
	memset(*flops, 0, sizeof(real__t)*threads);

	clv = 0;
	while (clv < level)
	{
		len = aeg_head[clv+1] - aeg_head[clv];
		mode = ((len <= thres) ? 1 : 0);

		lstart = clv;

		/*search lend*/
		if (mode == 0)/*cluster*/
		{
			lend = level;
			++clv;
			while (clv < level)
			{
				len = aeg_head[clv+1] - aeg_head[clv];
				if (len <= thres)
				{
					lend = clv;
					break;
				}
				++clv;
			}

			/*loop cluster*/
			lv = lstart;

			while (lv < lend)
			{
				aegh = aeg_head[lv];
				len = aeg_head[lv+1];
				total = 0;
				for (j=aegh; j<len; ++j)
				{
					t = aeg_data[j];
					total += wkld[t];
				}
				avg = (size_t)((real__t)total / threads * bal);

				j = aegh;
				sub = 0;
				while (j < len)
				{
					t = aeg_data[j];
					sub += wkld[t];
					++j;
					if (sub >= avg) break;
				}

				start = j;

				/*thread 0: aegh~j*/
				for (k=aegh; k<j; ++k)
				{
					ll = llen[k];
					ul = ulen[k];

					lip = (uint__t *)(((byte__t *)lu) + up[k] + g_sp*ul);

					for (p=0; p<ll; ++p)
					{
						(*flops)[0] += (ulen[lip[p]] << 1);
					}

					(*flops)[0] += ul;
				}

				for (i=1; i<threads; ++i)
				{
					sub = 0;
					while (j < len)
					{
						t = aeg_data[j];
						sub += wkld[t];
						++j;
						if (sub >= avg) break;
					}
					if (i == threads-1) end = len;
					else end = j;

					/*thread i: start~end*/
					for (k=start; k<end; ++k)
					{
						ll = llen[k];
						ul = ulen[k];

						lip = (uint__t *)(((byte__t *)lu) + up[k] + g_sp*ul);

						for (p=0; p<ll; ++p)
						{
							(*flops)[i] += (ulen[lip[p]] << 1);
						}

						(*flops)[i] += ul;
					}

					start = end;
				}

				++lv;
			}
		}
		else/*pipeline*/
		{
			lend = level;
			++clv;
			while (clv < level)
			{
				len = aeg_head[clv+1] - aeg_head[clv];
				if (len > thres)
				{
					lend = clv;
					break;
				}
				++clv;
			}

			start = aeg_head[lstart];
			end = aeg_head[lend];

			/*begin pipeline*/
			for (i=0; i<threads; ++i)
			{
				for (k=start+i; k<end; k+=threads)
				{
					ll = llen[k];
					ul = ulen[k];

					lip = (uint__t *)(((byte__t *)lu) + up[k] + g_sp*ul);

					for (p=0; p<ll; ++p)
					{
						(*flops)[i] += (ulen[lip[p]] << 1);
					}

					(*flops)[i] += ul;
				}
			}
		}
	}

	return NICS_OK;
}
