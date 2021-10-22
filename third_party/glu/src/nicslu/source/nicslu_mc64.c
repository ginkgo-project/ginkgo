/*hsl_mc64 algorithm*/
/*it's modified from mc64ad.c in SuperLU_DIST package*/
/* http://crd.lbl.gov/~xiaoye/SuperLU/ */

#include "nicslu.h"
#include "nicslu_internal.h"
#include "math.h"

#define c__1	1
#define c__2	2

int__t _I_NicsLU_MC64ad(uint__t n, uint__t nnz, uint__t *ai, uint__t *ap, real__t *ax, \
	int__t *match, int__t *match2, uint__t liw, int__t *iw, uint__t ldw, real__t *dw)
{
	uint__t i, j;
	real__t *dw_max;
	real__t *dw_abs;
	real__t rinf;
	real__t fact;
	uint__t start, end;
	real__t d;
	real__t *dd;
	int__t rt_ct;

	dw_max = dw+(n+n);
	dw_abs = dw_max+n;
	rinf = DBL_MAX/n;

	for (j=0; j<n; ++j)
	{
		fact = 0.0;
		start = ap[j];
		end = ap[j+1];
		for (i=start; i<end; ++i)
		{
			d = ax[i];
			dw_abs[i] = ABS(d);
			d = dw_abs[i];
			if (d > fact)
			{
				fact = d;
			}
		}

		dw_max[j] = fact;
		if (fact != 0.0)
		{
			fact = log(fact);/*fact=log(maximum value in row j)*/
		}
		else
		{
		/*	fact = rinf;*/
			return NICSLU_MATRIX_STRUCTURAL_SINGULAR;
		}

		for (i=start; i<end; ++i)
		{
			d = dw_abs[i];
			if (d != 0.0)
			{
				dw_abs[i] = fact-log(d);
			}
			else
			{
				dw_abs[i] = rinf;
			}
		}
	}


	dd = &dw[n];

	rt_ct = _I_NicsLU_MC64wd(n, nnz, ai, ap, dw_abs, match, iw, &iw[n], &iw[2*n], &iw[3*n], &iw[4*n], dw, dd);

	if (FAIL(rt_ct))
	{
		return rt_ct;
	}
	else if ((uint__t)rt_ct < n)
	{
		return NICSLU_MATRIX_STRUCTURAL_SINGULAR;
	}

	for (i=0; i<n; ++i)
	{
		match2[match[i]] = i;

		d = dw_max[i];
		if (d != 0.0)
		{
			dd[i] -= log(d);
		}
		else
		{
			dd[i] = 0.0;
		}
	}

	return n;
}

/*binary heap operation functions*/
void _I_NicsLU_MC64dd(uint__t i, uint__t n, int__t *q, real__t *d__, int__t *l, uint__t iway)
{
	/*i=col*/

	real__t di;
	uint__t pos, posk, qk;
	uint__t idum;

	di = d__[i];
	pos = l[i];

	if (iway == 1)
	{
		for (idum=0; idum<n; ++idum)
		{
			if (pos <= 0)
			{
				goto FINISH;
			}
			posk = (pos-1)>>1;
			qk = q[posk];
			if (di <= d__[qk])
			{
				goto FINISH;
			}
			q[pos] = qk;
			l[qk] = pos;
			pos = posk;
		}
	}
	else
	{
		for (idum=0; idum<n; ++idum)
		{
			if (pos <= 0)
			{
				goto FINISH;
			}
			posk = (pos-1)>>1;
			qk = q[posk];
			if (di >= d__[qk])
			{
				goto FINISH;
			}
			q[pos] = qk;
			l[qk] = pos;
			pos = posk;
		}
	}

FINISH:
	q[pos] = i;
	l[i] = pos;
}

void _I_NicsLU_MC64ed(int__t *qlen, uint__t n, int__t *q, real__t *d__, int__t *l, uint__t iway)
{
	/*qlen>=1*/
	uint__t i;
	real__t di, dk, dr;
	int__t pos, posk;
	uint__t idum;
	int__t ql = *qlen-1;

	i = q[ql];
	di = d__[i];
	--(*qlen);
	pos = 0;

	if (iway == 1)
	{
		for (idum=0; idum<n; ++idum)
		{
			posk = ((pos+1)<<1)-1;
			if (posk > ql)
			{
				goto FINISH;
			}
			dk = d__[q[posk]];
			if (posk < ql)
			{
				dr = d__[q[posk+1]];
				if (dk < dr)
				{
					++posk;
					dk = dr;
				}
			}
			if (di >= dk)
			{
				goto FINISH;
			}
			q[pos] = q[posk];
			l[q[pos]] = pos;
			pos = posk;
		}
	}
	else
	{
		for (idum=0; idum<n; ++idum)
		{
			posk = ((pos+1)<<1)-1;
			if (posk > ql)
			{
				goto FINISH;
			}
			dk = d__[q[posk]];
			if (posk < ql)
			{
				dr = d__[q[posk+1]];
				if (dk > dr)
				{
					++posk;
					dk = dr;
				}
			}
			if (di <= dk)
			{
				goto FINISH;
			}
			q[pos] = q[posk];
			l[q[pos]] = pos;
			pos = posk;
		}
	}

FINISH:
	q[pos] = i;
	l[i] = pos;
}

void _I_NicsLU_MC64fd(int__t pos0, int__t *qlen, uint__t n, int__t *q, real__t *d__, int__t *l, uint__t iway)
{
	uint__t i;
	real__t di, dk, dr;
	uint__t idum;
	int__t qk, pos, posk;
	int__t ql = *qlen-1;

	if (pos0 == ql)
	{
		--(*qlen);
		return;
	}

	i = q[ql];
	di = d__[i];
	--(*qlen);
	pos = pos0;

	if (iway == 1)
	{
		for (idum=0; idum<n; ++idum)
		{
			if (pos <= 0)
			{
				goto STEP1;
			}
			posk = (pos-1)>>1;
			qk = q[posk];
			if (di <= d__[qk])
			{
				goto STEP1;
			}
			q[pos] = qk;
			l[qk] = pos;
			pos = posk;
		}

STEP1:
		q[pos] = i;
		l[i] = pos;

		for (idum=0; idum<n; ++idum)
		{
			posk = ((pos+1)<<1)-1;
			if (posk > ql)
			{
				goto FINISH;
			}
			dk = d__[q[posk]];
			if (posk < ql)
			{
				dr = d__[q[posk+1]];
				if (dk < dr)
				{
					++posk;
					dk = dr;
				}
			}
			if (di >= dk)
			{
				goto FINISH;
			}
			qk = q[posk];
			q[pos] = qk;
			l[qk] = pos;
			pos = posk;
		}

	}

	else
	{
		for (idum=0; idum<n; ++idum)
		{
			if (pos <= 0)
			{
				goto STEP2;
			}
			posk = (pos-1)>>1;
			qk = q[posk];
			if (di >= d__[qk])
			{
				goto STEP2;
			}
			q[pos] = qk;
			l[qk] = pos;
			pos = posk;
		}

STEP2:
		q[pos] = i;
		l[i] = pos;

		for (idum=0; idum<n; ++idum)
		{
			posk = ((pos+1)<<1)-1;
			if (posk > ql)
			{
				goto FINISH;
			}
			dk = d__[q[posk]];
			if (posk < ql)
			{
				dr = d__[q[posk+1]];
				if (dk > dr)
				{
					++posk;
					dk = dr;
				}
			}
			if (di <= dk)
			{
				goto FINISH;
			}
			qk = q[posk];
			q[pos] = qk;
			l[qk] = pos;
			pos = posk;
		}
	}

FINISH:
	q[pos] = i;
	l[i] = pos;
}

/****************************************************************************************************************/
int__t _I_NicsLU_MC64wd(uint__t n, uint__t nnz, uint__t *ai, uint__t *ap, real__t *ax, int__t *iperm, int__t *jperm, \
					int__t *out, int__t *pr, int__t *q, int__t *l, real__t *u, real__t *d__)
{
	/*u: minimum value of each column*/
	/*l: now row i uses the l[i]th element as pivot*/

	uint__t i, j, k;
	uint__t start;
	real__t di;
	real__t vj;
	uint__t i0, k0;
	uint__t kk, kk1, kk2;
	uint__t col1;
	uint__t num = 0;
	int__t jj;
	uint__t jord;
	real__t dmin, csp;
	int__t qlen, low, up;
	uint__t q0;
	real__t dnew;
	uint__t isp, jsp;
	uint__t jdum;
	real__t dq0;
	uint__t end;
	uint__t col;
	int__t k1;

	for (i=0; i<n; ++i)
	{
		u[i] = DBL_MAX;
	}
	memset(d__, 0, sizeof(real__t)*n);
	memset(iperm, 0xff, sizeof(int__t)*n);
	memset(jperm, 0xff, sizeof(int__t)*n);
	memset(l, 0xff, sizeof(int__t)*n);
	memcpy(pr, ap, sizeof(uint__t)*n);

	for (j=0; j<n; ++j)
	{
		end = ap[j+1];
		for (k=ap[j]; k<end; ++k)
		{
			col = ai[k];
			if (ax[k] > u[col])
			{
				continue;
			}
			u[col] = ax[k];
			iperm[col] = j;
			l[col] = k;
		}
	}

	for (i=0; i<n; ++i)
	{
		/*iperm[i]>=0 && jperm[iperm[i]]<0*/
		jj = iperm[i];
		if (jj < 0)
		{
			continue;
		}
		iperm[i] = -1;
		if (jperm[jj] >= 0)
		{
			continue;
		}
		++num;
		iperm[i] = jj;
		jperm[jj] = l[i];
	}

	if (num == n)
	{
		goto FINAL_STEP;
	}

	/*scan unassigned rows*/
	for (j=0; j<n; ++j)
	{
		if (jperm[j] >= 0)
		{
			continue;
		}

		start = ap[j];
		end = ap[j+1];
		if (start >= end)
		{
			return NICSLU_MATRIX_STRUCTURAL_SINGULAR;
		/*	continue;*/
		}

		/*scan a row*/
		vj = DBL_MAX;
		for (k=start; k<end; ++k)
		{
			col = ai[k];
			di = ax[k]-u[col];
			if (di > vj)
			{
				continue;
			}
			if (di < vj || di == DBL_MAX)
			{
				goto STEP1;
			}
			if (iperm[col] >= 0 || iperm[i0] < 0)
			{
				continue;
			}
STEP1:
			vj = di;
			i0 = col;
			k0 = k;
		}

		d__[j] = vj;
		k = k0;
		col = i0;
		if (iperm[col] < 0)
		{
			goto STEP3;
		}

		for (k=k0; k<end; ++k)
		{
			col = ai[k];
			if (ax[k]-u[col] > vj)
			{
				continue;
			}
			jj = iperm[col];

			/*scan remaining part of assigned row jj*/
			kk1 = pr[jj];
			kk2 = ap[jj+1];
			if (kk1 >= kk2)
			{
				continue;
			}
			for (kk=kk1; kk<kk2; ++kk)
			{
				col1 = ai[kk];
				if (iperm[col1] >= 0)
				{
					continue;
				}
				if (ax[kk]-u[col1] <= d__[jj])
				{
					goto STEP2;
				}
			}
			pr[jj] = kk2;
		}
		continue;

STEP2:

		jperm[jj] = kk;
		iperm[col1] = jj;
		pr[jj] = kk+1;

STEP3:

		++num;
		jperm[j] = k;
		iperm[col] = j;
		pr[j] = k+1;
	}

	if (num == n)
	{
		goto FINAL_STEP;
	}

	/*prepare for main loop*/
	for (i=0; i<n; ++i)
	{
		d__[i] = DBL_MAX;
	}
	memset(l, 0xff, sizeof(int__t)*n);

	/*main loop
	Dijkstra's algorithm for solving the single source shortest path problem*/

	for (jord=0; jord<n; ++jord)
	{
		if (jperm[jord] >= 0)
		{
			continue;
		}
/*
		dmin: length of the shortest path in the tree
		csp: the cost of the shortest augmenting path to unassigned column
		ai(ISP): The corresponding column index is JSP*/
		
		dmin = DBL_MAX;
		qlen = 0;
		low = n;
		up = n;
		csp = DBL_MAX;

		/*Build shortest path tree starting from unassigned row (root) JORD*/
		j = jord;
		pr[j] = -2;
		

		/*scan row j*/
		end = ap[j+1];
		for (k=ap[j]; k<end; ++k)
		{
			col = ai[k];
			dnew = ax[k]-u[col];

			if (dnew >= csp)
			{
				continue;
			}
			if (iperm[col] < 0)
			{
				csp = dnew;
				isp = k;
				jsp = j;
			}
			else
			{
				if (dnew < dmin)
				{
					dmin = dnew;
				}
				d__[col] = dnew;
				q[qlen++] = k;

			}
		}

		/*Initialize heap Q and Q2 with rows held in Q(0:QLEN-1)*/
		
		q0 = qlen;
		qlen = 0;


		/*scan the queue*/
		for (kk=0; kk<q0; ++kk)
		{

			k = q[kk];

			col = ai[k];

			if (d__[col] >= csp)
			{
				d__[col] = DBL_MAX;
				continue;
			}
			if (d__[col] <= dmin)/*queue B*/
			{
				q[--low] = col;
				l[col] = low;

			}
			else/*d__[col] > dmin, queue Q*/
			{
			/*	qlen++;*/
				l[col] = qlen++;
				_I_NicsLU_MC64dd(col, n, q, d__, l, c__2);
			}
			/*update tree*/
			jj = iperm[col];
			out[jj] = k;
			pr[jj] = j;
		}
		
		for (jdum=0; jdum<num; ++jdum)
		{
			/*if q2 is empty, extract columns from q*/
			if (low == up)
			{
				if (qlen == 0)
				{
					goto STEP6;
				}
				col = q[0];
				if (d__[col] >= csp)
				{
					goto STEP6;
				}
				dmin = d__[col];
				/*qlen>=1*/
STEP4:
				_I_NicsLU_MC64ed(&qlen, n, q, d__, l, c__2);
				q[--low] = col;
				l[col] = low;
				if (qlen == 0)
				{
					goto STEP5;
				}
				col = q[0];
				if (d__[col] > dmin)
				{
					goto STEP5;
				}
				goto STEP4;
			}

STEP5:
			q0 = q[up-1];
			dq0 = d__[q0];
			if (dq0 >= csp)
			{
				goto STEP6;
			}
			--up;
			j = iperm[q0];
			vj = dq0-ax[jperm[j]]+u[q0];

			/*scan row that matches with column q0*/
			end = ap[j+1];
			for (k=ap[j]; k<end; ++k)
			{
				col = ai[k];
				if (l[col] >= up)
				{
					continue;
				}
				dnew = vj+ax[k]-u[col];
				if (dnew >= csp)
				{
					continue;
				}
				if (iperm[col] < 0)
				{
					csp = dnew;
					isp = k;
					jsp = j;
				}
				else
				{
					di = d__[col];
					if (di <= dnew)
					{
						continue;
					}
					if (l[col] >= low)
					{
						continue;
					}
					d__[col] = dnew;
					if (dnew <= dmin)
					{
						if (l[col] >= 0)
						{
							_I_NicsLU_MC64fd(l[col], &qlen, n, q, d__, l, c__2);
						}
						q[--low] = col;
						l[col] = low;
					}
					else
					{
						if (l[col] < 0)
						{
						/*	qlen++;*/
							l[col] = qlen++;
						}
						_I_NicsLU_MC64dd(col, n, q, d__, l, c__2);
					}

					jj = iperm[col];
					out[jj] = k;
					pr[jj] = j;
				}
			}
		}
		
STEP6:
		if (csp == DBL_MAX)
		{
			goto STEP8;
		}

		
		++num;
		col = ai[isp];
		iperm[col] = jsp;
		jperm[jsp] = isp;
		j = jsp;

		for (jdum=0; jdum<num; ++jdum)
		{
			jj = pr[j];
			if (jj == -2)
			{
				goto STEP7;
			}
			k = out[j];
			col = ai[k];
			iperm[col] = jj;
			jperm[jj] = k;
			j = jj;
		}

STEP7:

		for (kk=up; kk<n; ++kk)
		{
			col = q[kk];
			u[col] = u[col]+d__[col]-csp;
		}

STEP8:
		for (kk=low; kk<n; ++kk)
		{
			col = q[kk];
			d__[col] = DBL_MAX;
			l[col] = -1;
		}

		for (k1=0; k1<qlen; ++k1)
		{
			col = q[k1];
			d__[col] = DBL_MAX;
			l[col] = -1;
		}

	}

FINAL_STEP:

	for (j=0; j<n; ++j)
	{
		jj = jperm[j];
		if (jj >= 0)
		{
			d__[j] = ax[jj]-u[ai[jj]];

		}
		else
		{
			d__[j] = 0.0;
		}
		if (iperm[j] < 0)
		{
			u[j] = 0.0;
		}
	}

	if (num == n)
	{
		return n;
	}

	memset(jperm, 0xff, sizeof(int__t)*n);

	k = 0;
	for (i=0; i<n; ++i)
	{
		if (iperm[i] < 0)
		{
			out[k++] = i;
		}
		else
		{
			j = iperm[i];
			jperm[j] = i;
		}
	}

	k = 0;
	for (j=0; j<n; ++j)
	{
		if (jperm[j] >= 0)
		{
			continue;
		}
		jdum = out[k++];
		iperm[jdum] = j;
	}

	return num;
}
