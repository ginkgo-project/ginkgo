/*common kernel functions*/
/*last modified: june 11, 2013*/
/*author: Chen, Xiaoming*/

#include "nicslu.h"
#include "nicslu_internal.h"
#include "math.h"

extern size_t g_sd, g_si, g_sp, g_sis1, g_sps1;

int__t _I_NicsLU_Symbolic(uint__t n, uint__t k, \
	int__t *pinv, int__t *stack, int__t *flag, int__t *pend, int__t *appos, \
	uint__t *uindex, uint__t *ulen, void *lu, size_t *up, uint__t *aidx, uint__t arownnz)
{
	int__t top;
	uint__t i, col, j, jnew;
	int__t head, pos;
	uint__t *uidx;
	uint__t ucol;
	uint__t unnz;

	top = n;
	unnz = 0;

	for (i=0; i<arownnz; ++i)
	{
		col = aidx[i];
		
		if (flag[col] != k)/*not visited*/
		{
			if (pinv[col] >= 0)/*column col is pivotal, start dfs*/
			{
				head = 0;
				stack[0] = col;

				while (head >= 0)
				{
					j = stack[head];/*j is the original value*/
					jnew = pinv[j];/*column j is the jnewth pivot column*/

					if (flag[j] != k)
					{
						flag[j] = k;
						appos[head] = ((pend[jnew]<0) ? ulen[jnew] : pend[jnew]);
					}

					uidx = (uint__t *)(((byte__t *)lu) + up[jnew]);
					for (pos=--appos[head]; pos>=0; --pos)
					{
						ucol = uidx[pos];
						if (flag[ucol] != k)
						{
							if (pinv[ucol] >= 0)/*dfs*/
							{
								appos[head] = pos;
								stack[++head] = ucol;
								break;
							}
							else/*directly push into u*/
							{
								flag[ucol] = k;
								uindex[unnz++] = ucol;
							}
						}
					}

					if (pos < 0)
					{
						--head;
						stack[--top] = j;
					}
				}/*end while*/
			}
			else/*directly push into u*/
			{
				flag[col] = k;
				uindex[unnz++] = col;
			}
		}
	}/*end for*/
	
	ulen[k] = unnz;
	return top;
}

int _I_NicsLU_Pivot(int__t diagcol, uint__t *ulen, size_t up, \
	real__t tol, real__t *x, uint__t *p_pivcol, real__t *p_pivot, void *lu)
{
	uint__t lens1, p, i;
	uint__t last_col;
	real__t *u_row;
	real__t tx, xabs;
	int__t pdiag, ppivcol;
	real__t abs_pivot;
	uint__t pivcol;
	real__t pivot;
	uint__t *uip;

	if (*ulen == 0) return NICSLU_MATRIX_STRUCTURAL_SINGULAR;

	uip = (uint__t *)(((byte__t *)lu) + up);
	lens1 = (*ulen) - 1;
	last_col = uip[lens1];
	*ulen = lens1;/*<==> *ulen--*/
	u_row = (real__t *)(uip + lens1);

	pdiag = -1;
	abs_pivot = -1.;
	ppivcol = -1;

	for (p=0; p<lens1; ++p)
	{
		i = uip[p];/*column index*/
		tx = x[i];
		x[i] = 0.;
		u_row[p] = tx;/*put u into lu data structure*/
		xabs = ABS(tx);

		/*search the diag col*/
		if (i == diagcol)
		{
			pdiag = p;
		}
		/*search the maximum pivot*/
		if (xabs > abs_pivot)
		{
			abs_pivot = xabs;
			ppivcol = p;
		}
	}

	xabs = x[last_col];
	if (xabs < 0.) xabs = -xabs;
	/*xabs = ABS(x[last_col]);*/
	/*if the last entry is the maximum*/
	if (xabs > abs_pivot)
	{
		abs_pivot = xabs;
		ppivcol = -1;
	}

	/*if the diag is large enough, then the pivot is the diag*/
	/*currently xabs = ABS(x[last_col])*/
	if (last_col == diagcol)
	{
		if (xabs >= tol * abs_pivot)
		{
			abs_pivot = xabs;
			ppivcol = -1;
		}
	}
	else if (pdiag >= 0)/*last_col != diagcol*/
	{
		xabs = u_row[pdiag];
		if (xabs < 0.) xabs = -xabs;
		/*xabs = ABS(u_row[pdiag]);*/

		if (xabs >= tol * abs_pivot)
		{
			abs_pivot = xabs;
			ppivcol = pdiag;
		}
	}

	if (ppivcol >= 0)/*the pivot is not the last*/
	{
		pivcol = uip[ppivcol];
		pivot = u_row[ppivcol];
		uip[ppivcol] = last_col;
		u_row[ppivcol] = x[last_col];
	}
	else /*the pivot is the last. ppivcol=-1*/ /*the last is the maximum or the diag is not large enough*/
	{
		pivcol = last_col;
		pivot = x[last_col];
	}
	x[last_col] = 0.;

	*p_pivcol = pivcol;
	*p_pivot = pivot;

	if (pivot == 0.0)
	{
		return NICSLU_MATRIX_NUMERIC_SINGULAR;
	}
	if (isNaN(pivot))
	{
		return NICSLU_NUMERIC_OVERFLOW;
	}

	for (p=0; p<lens1; ++p)
	{
		u_row[p] /= pivot;
	}

	return NICS_OK;
}

void _I_NicsLU_Prune(int__t *pend, uint__t llen, uint__t *ulen, int__t *pinv, \
	int__t pivcol, uint__t *lip, size_t *ui, void *lu)
{
	uint__t p, i, j, p2;
	uint__t *uip;
	uint__t ul;
	int__t phead, ptail;
	real__t x, *urx;

	for (p=0; p<llen; ++p)
	{
		j = lip[p];

		if (pend[j] < 0)
		{
			ul = ulen[j];
			uip = (uint__t *)(((byte__t *)lu) + ui[j]);

			for (p2=0; p2<ul; ++p2)
			{
				if (uip[p2] == pivcol)
				{
					urx = (real__t *)(uip + ul);
					phead = 0;
					ptail = ul;

					while (phead < ptail)
					{
						i = uip[phead];
						if (pinv[i] >= 0)
						{
							++phead;
						}
						else
						{
							--ptail;
							uip[phead] = uip[ptail];
							uip[ptail] = i;
							x = urx[phead];
							urx[phead] = urx[ptail];
							urx[ptail] = x;
						}
					}

					pend[j] = ptail;
					break;
				}
			}
		}
	}
}
