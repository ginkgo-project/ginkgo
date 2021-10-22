#include "nicslu.h"
#include "nicslu_internal.h"

extern size_t g_si, g_sd, g_sp;

int NicsLU_DumpLU(SNicsLU *nicslu, real__t **lx, uint__t **li, size_t **lp, \
				  real__t **ux, uint__t **ui, size_t **up)
{
	uint__t n, i, j;
	size_t lnz, unz, ln, un;
	byte__t *lu;
	size_t *up0;
	uint__t *ulen, *llen;
	uint__t ul, ll;
	uint__t *index;
	real__t *data;
	real__t *ldiag;

	if (NULL == nicslu || NULL == lx || NULL == li || NULL == lp || NULL == ux || NULL == ui || NULL == up)
	{
		return NICSLU_ARGUMENT_ERROR;
	}
	if (!nicslu->flag[2])
	{
		return NICSLU_MATRIX_NOT_FACTORIZED;
	}

	if (*lx != NULL)
	{
		free(*lx);
		*lx = NULL;
	}
	if (*li != NULL)
	{
		free(*li);
		*li = NULL;
	}
	if (*lp != NULL)
	{
		free(*lp);
		*lp = NULL;
	}
	if (*ux != NULL)
	{
		free(*ux);
		*ux = NULL;
	}
	if (*ui != NULL)
	{
		free(*ui);
		*ui = NULL;
	}
	if (*up != NULL)
	{
		free(*up);
		*up = NULL;
	}

	n = nicslu->n;
	lnz = nicslu->l_nnz;
	unz = nicslu->u_nnz;
	lu = (byte__t *)(nicslu->lu_array);
	up0 = nicslu->up;
	ulen = nicslu->ulen;
	llen = nicslu->llen;
	ldiag = nicslu->ldiag;

	*lx = (real__t *)malloc(sizeof(real__t)*lnz);
	*ux = (real__t *)malloc(sizeof(real__t)*unz);
	*li = (uint__t *)malloc(sizeof(uint__t)*lnz);
	*ui = (uint__t *)malloc(sizeof(uint__t)*unz);
	*lp = (size_t *)malloc(sizeof(size_t)*(1+n));
	*up = (size_t *)malloc(sizeof(size_t)*(1+n));

	if (NULL == *lx || NULL == *li || NULL == *lp || NULL == *ux || NULL == *ui || NULL == *up)
	{
		goto FAIL;
	}

	(*lp)[0] = 0;
	(*up)[0] = 0;

	ln = 0;
	un = 0;

	for (i=0; i<n; ++i)
	{
		ul = ulen[i];
		ll = llen[i];

		/*l part*/
		index = (uint__t *)(lu + up0[i] + g_sp*ul);
		data = (real__t *)(index + ll);

		for (j=0; j<ll; ++j)
		{
			(*lx)[ln] = data[j];
			(*li)[ln] = index[j];
			++ln;
		}
		(*lx)[ln] = ldiag[i];
		(*li)[ln] = i;
		++ln;
		(*lp)[i+1] = ln;

		/*u part*/
		index = (uint__t *)(lu + up0[i]);
		data = (real__t *)(index + ul);

		(*ux)[un] = 1.;
		(*ui)[un] = i;
		++un;
		for (j=0; j<ul; ++j)
		{
			(*ux)[un] = data[j];
			(*ui)[un] = index[j];
			++un;
		}
		(*up)[i+1] = un;
	}

	return NICS_OK;

FAIL:
	if (*lx != NULL)
	{
		free(*lx);
		*lx = NULL;
	}
	if (*li != NULL)
	{
		free(*li);
		*li = NULL;
	}
	if (*lp != NULL)
	{
		free(*lp);
		*lp = NULL;
	}
	if (*ux != NULL)
	{
		free(*ux);
		*ux = NULL;
	}
	if (*ui != NULL)
	{
		free(*ui);
		*ui = NULL;
	}
	if (*up != NULL)
	{
		free(*up);
		*up = NULL;
	}
	return NICSLU_MEMORY_OVERFLOW;
}
