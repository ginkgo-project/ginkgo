/*approximate mimimum degree*/
/*it's modified from Tim Davis's AMD package, which is distributed under the GNU LGPL license*/
/* http://www.cise.ufl.edu/research/sparse/amd/ */
/* http://www.cise.ufl.edu/research/sparse/klu/ */

#include "nicslu.h"
#include "nicslu_internal.h"
#include "math.h"
#include <limits.h>

#define AMD_FLIP(i)		(-(i)-2)

static int__t __ClearFlag(int__t wflg, int__t wbig, int__t *w, int__t n)
{
	int__t x;
	if (wflg<2 || wflg>=wbig)
	{
		for (x=0; x<n; ++x)
		{
			if (w[x] != 0) w[x] = 1;
		}
		wflg = 2;
	}
	return wflg;
}

void _I_NicsLU_AMDSort(uint__t n, uint__t *ai, uint__t *ap, int__t *w, uint__t *ri, uint__t *rp)
{
	uint__t p, p2, i, j;

	memset(w, 0, sizeof(int__t)*n);
	
	for (j=0; j<n; ++j)
	{
		p2 = ap[j+1];
		for (p=ap[j]; p<p2; ++p)
		{
			++w[ai[p]];
		}
	}

	rp[0] = 0;
	for (i=0; i<n; ++i)
	{
		rp[i+1] = rp[i]+w[i];
	}
	memcpy(w, rp, sizeof(int__t)*n);

	for (j=0; j<n; ++j)
	{
		p2 = ap[j+1];
		for (p=ap[j]; p<p2; ++p)
		{
			ri[w[ai[p]]++] = j;
		}
	}
}

uint__t _I_NicsLU_AAT(uint__t n, uint__t *ai, uint__t *ap, int__t *len, int__t *tp)
{
	uint__t k, p1, p2, p, j, pj2, pj, i;
	uint__t nzaat;
/*	uint__t nzdiag;
	uint__t nzboth;
	uint__t nz;*/

	nzaat = 0;
/*	nzdiag = 0;
	nzboth = 0;
	nz = ap[n];*/
	memset(len, 0, sizeof(int__t)*n);

	for (k=0; k<n; ++k)
	{
		p1 = ap[k];
		p2 = ap[k+1];

		for (p=p1; p<p2;)
		{
			j = ai[p];
			if (j < k)
			{
				++len[j];
				++len[k];
				++p;
			}
			else if (j == k)
			{
				++p;
			/*	++nzdiag;*/
				break;
			}
			else
			{
				break;
			}

			pj2 = ap[j+1];
			for (pj=tp[j]; pj<pj2;)
			{
				i = ai[pj];
				if (i < k)
				{
					++len[i];
					++len[j];
					++pj;
				}
				else if (i == k)
				{
					++pj;
				/*	++nzboth;*/
					break;
				}
				else
				{
					break;
				}
			}
			tp[j] = pj;
		}/*end scanning row k*/
		tp[k] = p;
	}/*end scanning all rows*/

	for (j=0; j<n; ++j)
	{
		pj2 = ap[j+1];
		for (pj=tp[j]; pj<pj2; ++pj)
		{
			++len[ai[pj]];
			++len[j];
		}
	}

	for (k=0; k<n; ++k)
	{
		nzaat += len[k];
	}

	return nzaat;
}

void _I_NicsLU_AAT2(uint__t n, uint__t *ai, uint__t *ap, int__t *len, int__t *pe, int__t *sp, int__t *iw, int__t *tp)
{
	uint__t k, p, p1, p2, j, i, pj, pj2;

	uint__t pfree;
	pfree = 0;

	for (j=0; j<n; ++j)
	{
		pe[j] = sp[j] = pfree;
		pfree += len[j];
	}

	for (k=0; k<n; ++k)
	{
		p1 = ap[k];
		p2 = ap[k+1];

		for (p=p1; p<p2;)
		{
			j = ai[p];
			if (j < k)
			{
				iw[sp[j]++] = k;
				iw[sp[k]++] = j;
				++p;
			}
			else if (j == k)
			{
				++p;
				break;
			}
			else
			{
				break;
			}

			pj2 = ap[j+1];
			for (pj=tp[j]; pj<pj2;)
			{
				i = ai[pj];
				if (i < k)
				{
					iw[sp[i]++] = j;
					iw[sp[j]++] = i;
					++pj;
				}
				else if (i == k)
				{
					++pj;
					break;
				}
				else
				{
					break;
				}
			}
			tp[j] = pj;
		}
		tp[k] = p;
	}

	for (j=0; j<n; ++j)
	{
		pj2 = ap[j+1];
		for (pj=tp[j]; pj<pj2; ++pj)
		{
			i = ai[pj];
			iw[sp[i]++] = j;
			iw[sp[j]++] = i;
		}
	}
}

void _I_NicsLU_AMD(int__t n, int__t pfree, int__t iwlen, int__t *pe, int__t *iw, \
	int__t *len, int__t *work, int__t *last, int__t *next, real__t alpha, int__t aggr, \
	size_t *lu_nnz)
{
	/*pe, iw, len, last, next*/
	int__t *nv, *head, *elen, *degree, *w;

	/*local integers*/
	int__t deg, degme, dext, lemax, e, elenme, eln, ilast, inext, jlast, jnext, \
		k, knt1, knt2, knt3, lenj, ln, me, mindeg, nleft, nvi, nvj, nvpiv, slenme, \
		wbig, we, wflg, wnvi, ok, ncmpa, dense, nel, ndense, i, j;
	uint__t hash;

	/*local pointers*/
	int__t p, p1, p2, p3, p4, pdst, pend, pj, pme, pme1, pme2, pme3, pn, psrc, pend2;

	real__t f, r, dmax, lnz, lnzme;

	lnz = 0;
	dmax = 1;

	nv = work+n;
	head = nv+n;
	elen = head+n;
	degree = elen+n;
	w = degree+n;


	me = -1;
	mindeg = 0;
	ncmpa = 0;
	nel = 0;
	lemax = 0;

	dense = (int__t)((alpha<0.0) ? (n-2) : (alpha*sqrt((real__t)n)));
	dense = MAX(16, dense);
	dense = MIN(n, dense);

	memset(last, 0xff, sizeof(int__t)*n);
	memset(head, 0xff, sizeof(int__t)*n);
	memset(next, 0xff, sizeof(int__t)*n);
	memset(elen, 0, sizeof(int__t)*n);
	memcpy(degree, len, sizeof(int__t)*n);
	for (i=0; i<n; ++i)
	{
		nv[i] = 1;
		w[i] = 1;
	}

#ifdef INT64__
	wbig = LLONG_MAX - n;
#else
	wbig = INT_MAX - n;
#endif
	wflg = __ClearFlag(0, wbig, w, n);

	ndense = 0;

	for (i=0; i<n; ++i)
	{
		deg = degree[i];
		if (deg == 0)
		{
			elen[i] = AMD_FLIP(1);
			++nel;
			pe[i] = -1;
			w[i] = 0;
		}
		else if (deg > dense)
		{
			++ndense;
			nv[i] = 0;
			elen[i] = -1;
			++nel;
			pe[i] = -1;
		}
		else
		{
			inext = head[deg];
			if (inext != -1) last[inext] = i;
			next[i] = inext;
			head[deg] = i;
		}
	}


	while (nel < n)
	{

		/*get pivot of minimum degree*/

		/*find next supervariable for elimination*/
		for (deg=mindeg; deg<n; ++deg)
		{
			me = head[deg];
			if (me != -1) break;
		}
		mindeg = deg;

		/*remove chosen variable from link list*/
		inext = next[me];
		if (inext != -1) last[inext] = -1;
		head[deg] = inext;

		/*me represents the elimination of pivots nel to nel+nv[me]-1
		place me itself as the first in this set*/
		elenme = elen[me];
		nvpiv = nv[me];
		nel += nvpiv;

		/*construct new element

		at this point, me is the pivotal supervariable
		flag the variable me as being in lme by negating nv[me]*/
		nv[me] = -nvpiv;
		degme = 0;

		if (elenme == 0)
		{
			/*construct the new element in place*/

			pme1 = pe[me];
			pme2 = pme1-1;
			pme3 = pme1+len[me];
			for (p=pme1; p<pme3; ++p)
			{
				i = iw[p];
				nvi = nv[i];
				if (nvi > 0)
				{
					/*store i in new list*/
					degme += nvi;
					nv[i] = -nvi;
					iw[++pme2] = i;

					/*remove variable i from degree list*/
					ilast = last[i];
					inext = next[i];
					if (inext != -1) last[inext] = ilast;
					if (ilast != -1) next[ilast] = inext;
					else head[degree[i]] = inext;
				}
			}
		}
		else
		{
			/*construct the new element in empty space, iw[pfree...]*/

			p = pe[me];
			pme1 = pfree;
			slenme = len[me]-elenme;

			for (knt1=1; knt1<=elenme+1; ++knt1)
			{
				if (knt1 > elenme)
				{
					/*search the supervariables in me*/
					e = me;
					pj = p;
					ln = slenme;
				}
				else
				{
					/*search the elements in me*/
					e = iw[p++];
					pj = pe[e];
					ln = len[e];
				}

				/*search for different supervariables and add them to the next list*/

				for (knt2=1; knt2<=ln; ++knt2)
				{
					i = iw[pj++];
					nvi = nv[i];

					if (nvi > 0)
					{
						/*compress iw*/
						if (pfree >= iwlen)
						{
							pe[me] = p;
							len[me] -= knt1;
							if (len[me] == 0) pe[me] = -1;
							pe[e] = pj;
							len[e] = ln-knt2;
							if (len[e] == 0) pe[e] = -1;

							++ncmpa;

							/*store first entry of each object in pe
							flip the first entry in each object*/
							for (j=0; j<n; ++j)
							{
								pn = pe[j];
								if (pn >= 0)
								{
									pe[j] = iw[pn];
									iw[pn] = AMD_FLIP(j);
								}
							}

							psrc = 0;
							pdst = 0;
							pend = pme1-1;

							while (psrc <= pend)
							{
								/*search the next fliped entry*/
								j = AMD_FLIP(iw[psrc++]);
								if (j >= 0)
								{
									iw[pdst] = pe[j];
									pe[j] = pdst++;
									lenj = len[j];
									for (knt3=0; knt3<lenj-1; ++knt3)
									{
										iw[pdst++] = iw[psrc++];
									}
								}
							}

							/*move the new partially-constructed element*/
							p1 = pdst;
							for (psrc=pme1; psrc<pfree; ++psrc)
							{
								iw[pdst++] = iw[psrc];
							}
							pme1 = p1;
							pfree = pdst;
							pj = pe[e];
							p = pe[me];
						}

						/*i is a principal variable not yet placed in lme
						store i in new list*/
						degme += nvi;
						nv[i] = -nvi;
						iw[pfree++] = i;

						/*remove variable i from degree link list*/

						ilast = last[i];
						inext = next[i];
						if (inext != -1) last[inext] = ilast;
						if (ilast != -1) next[ilast] = inext;
						else head[degree[i]] = inext;
					}
				}
				if (e != me)
				{
					pe[e] = AMD_FLIP(me);
					w[e] = 0;
				}
			}
			pme2 = pfree-1;
		}/*end else (elenme!=0)*/

		/*me has been converted into an element in iw[pme1..pme2]
		degree holds the external degree of new element*/
		degree[me] = degme;
		pe[me] = pme1;
		len[me] = pme2-pme1+1;
		elen[me] = AMD_FLIP(nvpiv+degme);

		wflg = __ClearFlag(wflg, wbig, w, n);

		/*find set differences*/
		for (pme=pme1; pme<=pme2; ++pme)
		{
			i = iw[pme];
			eln = elen[i];
			if (eln > 0)
			{
				nvi = -nv[i];
				wnvi = wflg-nvi;
				pend2 = pe[i]+eln;
				for (p=pe[i]; p<pend2; ++p)
				{
					e = iw[p];
					we = w[e];
					if (we >= wflg)
					{
						we -= nvi;
					}
					else if (we != 0)
					{
						we = degree[e]+wnvi;
					}
					w[e] = we;
				}
			}
		}

		/*update degree*/
		for (pme=pme1; pme<=pme2; ++pme)
		{
			i = iw[pme];
			p1 = pe[i];
			p2 = p1+elen[i]-1;
			pn = p1;
			hash = 0;
			deg = 0;

			if (aggr)
			{
				for (p=p1; p<=p2; ++p)
				{
					e = iw[p];
					we = w[e];
					if (we != 0)
					{
						dext = we-wflg;
						if (dext > 0)
						{
							deg += dext;
							iw[pn++] = e;
							hash += e;
						}
						else
						{
							pe[e] = AMD_FLIP(me);
							w[e] = 0;
						}
					}
				}
			}
			else
			{
				for (p=p1; p<=p2; ++p)
				{
					e = iw[p];
					we = w[e];
					if (we != 0)
					{
						dext = we-wflg;
						deg += dext;
						iw[pn++] = e;
						hash += e;
					}
				}
			}

			/*count the number of elements in i*/
			elen[i] = pn-p1+1;

			/*scan the supervariables in the list associated with i*/
			p3 = pn;
			p4 = p1+len[i];
			for (p=p2+1; p<p4; ++p)
			{
				j = iw[p];
				nvj = nv[j];
				if (nvj > 0)
				{
					deg += nvj;
					iw[pn++] = j;
					hash += j;
				}
			}

			/*update the degree and check for mass elimination*/
			if (elen[i]==1 && p3==pn)
			{
				pe[i] = AMD_FLIP(me);
				nvi = -nv[i];
				degme -= nvi;
				nvpiv += nvi;
				nel += nvi;
				nv[i] = 0;
				elen[i] = -1;
			}
			else
			{
				degree[i] = MIN(degree[i], deg);

				/*add me to the list for i*/
				iw[pn] = iw[p3];
				iw[p3] = iw[p1];
				iw[p1] = me;
				len[i] = pn-p1+1;

				hash = hash % (uint__t)n;

				j = head[hash];
				if (j <= -1)
				{
					next[i] = AMD_FLIP(j);
					head[hash] = AMD_FLIP(i);
				}
				else
				{
					next[i] = last[j];
					last[j] = i;
				}
				last[i] = hash;
			}
		}

		degree[me] = degme;
		lemax = MAX(lemax, degme);
		wflg += lemax;
		wflg = __ClearFlag(wflg, wbig, w, n);

		/*supervariable detection*/
		for (pme=pme1; pme<=pme2; ++pme)
		{
			i = iw[pme];
			if (nv[i] < 0)
			{
				hash = last[i];
				j = head[hash];
				if (j == -1)
				{
					i = -1;
				}
				else if (j < -1)
				{
					i = AMD_FLIP(j);
					head[hash] = -1;
				}
				else
				{
					i = last[j];
					last[j] = -1;
				}

				while (i!=-1 && next[i]!=-1)
				{
					ln = len[i];
					eln = elen[i];

					pend2 = pe[i]+ln;
					for (p=pe[i]+1; p<pend2; ++p)
					{
						w[iw[p]] = wflg;
					}

					jlast = i;
					j = next[i];

					while (j != -1)
					{
						ok = (len[j]==ln && elen[j]==eln);
						pend2 = pe[j]+ln;
						for (p=pe[j]+1; ok && p<pend2; ++p)
						{
							if (w[iw[p]] != wflg) ok = 0;
						}
						if (ok)
						{
							pe[j] = AMD_FLIP(i);
							nv[i] += nv[j];
							nv[j] = 0;
							elen[j] = -1;
							j = next[j];
							next[jlast] = j;
						}
						else
						{
							jlast = j;
							j = next[j];
						}
					}

					++wflg;
					i = next[i];
				}
			}
		}

		/*resotre degree lists and remove nonprincipal supervariables from element*/

		p = pme1;
		nleft = n-nel;
		for (pme=pme1; pme<=pme2; ++pme)
		{
			i = iw[pme];
			nvi = -nv[i];
			if (nvi > 0)
			{
				nv[i] = nvi;
				deg = degree[i]+degme-nvi;
				deg = MIN(deg, nleft-nvi);

				inext = head[deg];
				if (inext != -1) last[inext] = i;
				next[i] = inext;
				last[i] = -1;
				head[deg] = i;

				mindeg = MIN(mindeg, deg);
				degree[i] = deg;

				iw[p++] = i;
			}
		}

		/*finalize the new element*/
		nv[me] = nvpiv;
		len[me] = p-pme1;
		if (0 == len[me])
		{
			pe[me] = -1;
			w[me] = 0;
		}
		if (elenme != 0)
		{
			pfree = p;
		}

		/*lnz*/
		f = (real__t)nvpiv;
		r = (real__t)(degme+ndense);
		dmax = MAX(dmax, f+r);
		lnzme = f*r + (f-1)*f*0.5;
		lnz += lnzme;

	}/*end while (nel < n)*/

	/*lnz*/
	if (ndense > 0)
	{
		f = (real__t)ndense;
		dmax = MAX(dmax, f);
		lnzme = (f-1)*f*0.5;
		lnz += lnzme;

	}
	*lu_nnz = (size_t)(lnz+lnz);

	/*post ordering*/
	for (i=0; i<n; ++i)
	{
		pe[i] = AMD_FLIP(pe[i]);
		elen[i] = AMD_FLIP(elen[i]);
	}
/*	for (i=0; i<n; ++i)
	{
		elen[i] = AMD_FLIP(elen[i]);
	}*/
	for (i=0; i<n; ++i)
	{
		if (nv[i] == 0)
		{
			j = pe[i];
			if (j == -1)
			{
				continue;
			}

			while (nv[j] == 0)
			{
				j = pe[j];
			}
			e = j;

			j = i;
			while (nv[j] == 0)
			{
				jnext = pe[j];
				pe[j] = e;
				j = jnext;
			}
		}
	}



	/*postorder the assembly tree*/
	_I_NicsLU_PostOrder(n, pe, nv, elen, w, head, next, last);

	memset(head, 0xff, sizeof(int__t)*n);
	memset(next, 0xff, sizeof(int__t)*n);
	for (e=0; e<n; ++e)
	{
		k = w[e];
		if (k != -1)
		{
			head[k] = e;
		}
	}

	nel = 0;
	for (k=0; k<n; ++k)
	{
		e = head[k];
		if (e == -1) break;
		next[e] = nel;
		nel += nv[e];
	}

	for (i=0; i<n; ++i)
	{
		if (nv[i] == 0)
		{
			e = pe[i];
			if (e != -1)
			{
				next[i] = next[e];
				++next[e];
			}
			else
			{
				next[i] = nel++;
			}
		}
	}

	for (i=0; i<n; ++i)
	{
		last[next[i]] = i;
	}

}

void _I_NicsLU_PostOrder(int__t n, int__t *parent, int__t *nv, int__t *fsize, int__t *order, int__t *child, int__t *sibling, int__t *stack)
{
	int__t i, j, k, par, frsize, f, fprev, maxfr, bigfp, bigf, fnext;

	memset(child, 0xff, sizeof(int__t)*n);
	memset(sibling, 0xff, sizeof(int__t)*n);

	for (j=n-1; j>=0; --j)
	{
		if (nv[j] > 0)
		{
			par = parent[j];
			if (par != -1)
			{
				sibling[j] = child[par];
				child[par] = j;
			}
		}
	}

	for (i=0; i<n; ++i)
	{
		if (nv[i]>0 && child[i]!=-1)
		{
			fprev = -1;
			maxfr = -1;
			bigfp = -1;
			bigf = -1;
			for (f=child[i]; f!=-1; f=sibling[f])
			{
				frsize = fsize[f];
				if (frsize >= maxfr)
				{
					maxfr = frsize;
					bigfp = fprev;
					bigf = f;
				}
				fprev = f;
			}

			fnext = sibling[bigf];

			if (fnext != -1)
			{
				if (bigfp == -1)
				{
					child[i] = fnext;
				}
				else
				{
					sibling[bigfp] = fnext;
				}
				sibling[bigf] = -1;
				sibling[fprev] = bigf;
			}
		}
	}

	memset(order, 0xff, sizeof(int__t)*n);
	k = 0;
	for (i=0; i<n; ++i)
	{
		if (parent[i]==-1 && nv[i]>0)
		{
			k = _I_NicsLU_PostTree(i, k, child, sibling, order, stack);
		}
	}
}

int__t _I_NicsLU_PostTree(int__t root, int__t k, int__t *child, int__t *sibling, int__t *order, int__t *stack)
{
	int__t f, head, h, i;

	head = 0;
	stack[0] = root;
	while (head >= 0)
	{
		i = stack[head];
		if (child[i] != -1)
		{
			for (f=child[i]; f!=-1; f=sibling[f])
			{
				++head;
			}
			h = head;
			for (f=child[i]; f!=-1; f=sibling[f])
			{
				stack[h--] = f;
			}
			child[i] = -1;
		}
		else
		{
			--head;
			order[i] = k++;
		}
	}

	return k;
}
