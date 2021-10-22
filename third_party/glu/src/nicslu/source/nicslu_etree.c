/*create the etree and the aegraph*/
/*last modified: june 7, 2013*/
/*author: Chen, Xiaoming*/

#include "nicslu.h"
#include "nicslu_internal.h"
#include "math.h"

static __inline int__t find_path(int__t i, int__t *pp)
{
    register int__t p, gp;
    
    p = pp[i];
    gp = pp[p];
    while (gp != p)
	{
		pp[i] = gp;
		i = gp;
		p = pp[i];
		gp = pp[p];
    }
    return p;
}


int _I_NicsLU_CreateETree(SNicsLU *nicslu)
{
	uint__t n, i, j, col, row, oldrow, cend;
	uint__t *rowperm, *ap, *ai;
	int__t *first, *pp, *root, *parent;
	int__t cset, rset, rroot;
	uint__t level;
	uint__t *data;
	uint__t *header;
	uint__t *inlevel, *lv_len, *tlen;
	uint__t lv;

	n = nicslu->n;
	rowperm = nicslu->row_perm;
	ap = nicslu->ap;
	ai = nicslu->ai;

	/*|----------|----------|----------|----------|*/
	/*first      pp         root       parent*/
	first = (int__t *)(nicslu->workspace);
	pp = first + n;
	root = pp + n;
	parent = root + n;

	memset(pp, 0, sizeof(int__t)*(n+n));/*clear root and pp*/

	/*first: the first nonzero of each column, i.e. the minimum row index of each column*/
	for (i=0; i<n; ++i)
	{
		first[i] = n;
	}

	for (i=0; i<n; ++i)
	{
		oldrow = rowperm[i];
		cend = ap[oldrow+1];
		for (j=ap[oldrow]; j<cend; ++j)
		{
			col = ai[j];
			if ((int__t)i < first[col])
			{
				first[col] = i;
			}
		}
	}

	/*calculate the etree*/
	for (i=0; i<n; ++i)
	{
		pp[i] = i;
		cset = i;
		root[cset] = i;
		parent[i] = n;

		oldrow = rowperm[i];
		cend = ap[oldrow+1];
		for (j=ap[oldrow]; j<cend; ++j)
		{
			row = first[ai[j]];
			if (row >= i) continue;
			rset = find_path(row, pp);
			rroot = root[rset];
			if (rroot != (int__t)i)
			{
				parent[rroot] = i;
				pp[cset] = rset;
				cset = rset;
				root[cset] = i;
			}
		}
	}

#ifdef NICSLU_DEBUG
	{
		FILE *fp = fopen("etree.txt", "w");
		for (i=0; i<n; ++i)
		{
			fprintf(fp, "%d\t%d\t%d\n", i, root[i], parent[i]);
		}
		fclose(fp);
	}
#endif
	/*build the aegraph*/

	/*|----------|----------|----------|----------|*/
	/*inlevel    lv_len     tlen       parent*/
	inlevel = (uint__t *)nicslu->workspace;
	lv_len = inlevel + n;
	memset(inlevel, 0, sizeof(uint__t)*n*2);
	tlen = (uint__t *)(lv_len + n);
	
	for (i=0; i<n; ++i)
	{
		j = parent[i];
		if (j < n)
		{
			lv = inlevel[i] + 1;
			if (lv > inlevel[j])
			{
				inlevel[j] = lv;
			}

		/*	rroot = et_child[j];
			et_next[i] = rroot;
			et_child[j] = i;*/
		}
	}

	level = 0;
	for (i=0; i<n; ++i)
	{
		lv = inlevel[i];
		++lv_len[lv];
		if (lv > level)
		{
			level = lv;
		}
	}
	++level;
	nicslu->aeg_level = level;

	data = nicslu->aeg_data;
	header = nicslu->aeg_header;

	header[0] = 0;
	for (i=0; i<level; ++i)
	{
		header[i+1] = header[i] + lv_len[i];
	}
	memcpy(tlen, header, sizeof(uint__t)*level);

	for (i=0; i<n; ++i)
	{
		data[tlen[inlevel[i]]++] = i;
	}

#ifdef NICSLU_DEBUG
	{
		FILE *fp = fopen("aegraph.txt", "w");
		for (i=0; i<level; ++i)
		{
			fprintf(fp, "%d\t%d\n", i, header[i+1]-header[i]);
		}
		fclose(fp);
	}
#endif

	return NICS_OK;
}
