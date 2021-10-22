/*calculate the memory throughput*/
/*last modified: june 7, 2013*/
/*author: Chen, Xiaoming*/

#include "nicslu.h"
#include "nicslu_internal.h"

extern size_t g_si, g_sd, g_sp;

int NicsLU_Throughput(SNicsLU *nicslu, real__t *thr)
{
	uint__t n, i, j;
	real__t total;
	uint__t *ulen, *llen, ul, ll, *rp, oldrow, *ap;
	size_t *up;
	uint__t *lip;
	void *lu;

	if (NULL == nicslu)
	{
		return NICSLU_ARGUMENT_ERROR;
	}
	if (thr != NULL) *thr = 0.;

	if (!nicslu->flag[2])
	{
		return NICSLU_MATRIX_NOT_FACTORIZED;
	}

	n = nicslu->n;
	total = 0.;
	ulen = nicslu->ulen;
	llen = nicslu->llen;
	up = nicslu->up;
	lu = nicslu->lu_array;
	rp = nicslu->row_perm;
	ap = nicslu->ap;

	for (i=0; i<n; ++i)
	{
		oldrow = rp[i];
		total += (ap[oldrow+1]-ap[oldrow]) * (g_si+2*g_sd);

		ll = llen[i];
		ul = ulen[i];

		lip = (uint__t *)(((byte__t *)lu) + up[i] + g_sp*ul);

		for (j=0; j<ll; ++j)
		{
			total += (g_si+3*g_sd) * ulen[lip[j]];
		}

		total += (ul+ll) * (g_si+3*g_sd);
	}

	nicslu->stat[12] = total;
	if (thr != NULL) *thr = total;

	return NICS_OK;
}
