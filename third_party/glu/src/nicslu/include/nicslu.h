/*interface of NICSLU*/
/*last modified: june 7, 2013*/
/*author: Chen, Xiaoming*/

#ifndef __NICSLU__
#define __NICSLU__

#include "nics_config.h"

/*return code*/
#define NICSLU_GENERAL_FAIL							(-1)
#define NICSLU_ARGUMENT_ERROR						(-2)
#define NICSLU_MEMORY_OVERFLOW						(-3)
#define NICSLU_FILE_CANNOT_OPEN						(-4)
#define NICSLU_MATRIX_STRUCTURAL_SINGULAR			(-5)
#define NICSLU_MATRIX_NUMERIC_SINGULAR				(-6)
#define NICSLU_MATRIX_INVALID						(-7)
#define NICSLU_MATRIX_ENTRY_DUPLICATED				(-8)
#define NICSLU_THREADS_NOT_INITIALIZED				(-9)
#define NICSLU_MATRIX_NOT_INITIALIZED				(-10)
#define NICSLU_SCHEDULER_NOT_INITIALIZED			(-11)
#define NICSLU_SINGLE_THREAD						(-12)
#define NICSLU_THREADS_INIT_FAIL					(-13)
#define NICSLU_MATRIX_NOT_ANALYZED					(-14)
#define NICSLU_MATRIX_NOT_FACTORIZED				(-15)
#define NICSLU_NUMERIC_OVERFLOW						(-16)
#define NICSLU_USE_SEQUENTIAL_FACTORIZATION			(+1)
#define NICSLU_BIND_THREADS_FAIL					(+2)

/*******************************************************************************/
/*definition of the main structure*/
typedef struct tagSNicsLU
{
	/*flags*/
	bool__t *flag;

	/*statistics*/
	real__t *stat;

	/*configurations*/
	uint__t *cfgi;
	real__t *cfgf;

	/*matrix data, 6 items*/
	uint__t n;				/*dimension*/
	uint__t nnz;			/*nonzeros of A*/
	real__t *ax;			/*value*/
	uint__t *ai;			/*column/row index*/
	uint__t *ap;			/*row/column header*/
	real__t *rhs;			/*for solve and tsolve*/

	/*other matrix data, 9 items*/
	uint__t *row_perm;		/*row_perm[i]=j-->row i in the permuted matrix is row j in the original matrix*/
	uint__t *row_perm_inv;	/*row_perm_inv[i]=j-->row i in the original matrix is row j in the permuted matrix*/
	uint__t *col_perm;
	uint__t *col_perm_inv;
	real__t *col_scale_perm;/*permuted*/
	real__t *row_scale;		/*not permuted*/
	real__t *cscale;
	int__t *pivot;			/*pivot[i]=j-->column j is the ith pivot column*/
	int__t *pivot_inv;		/*pivot_inv[i]=j-->column i is the jth pivot column*/

	/*lu matrix, 13 items*/
	size_t lu_nnz_est;		/*estimated total nnz by AMD*/
	size_t lu_nnz;			/*nonzeros of factorized matrix L+U-I*/
	size_t l_nnz;			/*inclu diag*/
	size_t u_nnz;			/*inclu diag*/
	real__t *ldiag;			/*udiag=1.0*/
	void *lu_array;			/*lu index and data*/
	size_t *up;				/*u row header, the header of each row*/
	uint__t *llen;			/*exclu diag*/
	uint__t *ulen;			/*exclu diag*/
	size_t *len_est;		/*estimated len, for parallelism, in bytes*/
	size_t *wkld_est;		/*estimated workload, for parallelism*/
	byte__t *row_state;		/*row state, finished or un-finished*/
	void **lu_array2;		/*for parallelism*/

	/*work space, 3 items*/
	void *workspace;
	void **workspace_mt1;
	void **workspace_mt2;

	/*for parallelism, 10 items*/
	volatile int thread_work;
	void *thread_id;		/*thread id, internal structure*/
	void *thread_arg;		/*thread argument, internal structure*/
	bool__t *thread_active;
	bool__t *thread_finish;
	uint__t *cluster_start;
	uint__t *cluster_end;
	uint__t pipeline_start;
	uint__t pipeline_end;
	uint__t last_busy;

	/*aegraph, 6 items*/
	uint__t aeg_level;
	uint__t *aeg_data;
	uint__t *aeg_header;
	uint__t aeg_refact_level;
	uint__t *aeg_refact_data;
	uint__t *aeg_refact_header;

	/*timer*/
	void *timer;

} SNicsLU;


#define IN__
#define OUT__
#define INOUT__

#ifdef __cplusplus
extern "C" {
#endif

/*******************************************************************************/
/*initialize the main structure. must be called at first*/
/*call it ONLY ONCE. don't repeatedly call this routine*/
int \
	NicsLU_Initialize( \
	INOUT__ SNicsLU *nicslu);

/*destroy the main structure and free all memories. must be called in the last*/
int \
	NicsLU_Destroy( \
	INOUT__ SNicsLU *nicslu);

/*initialize the matrix*/
/*all configurations should be set AFTER this routine*/
/*if it is called repeatedly, the previous allocated memory will be freed*/
int \
	NicsLU_CreateMatrix( \
	INOUT__ SNicsLU *nicslu, \
	IN__ uint__t n, \
	IN__ uint__t nnz, \
	IN__ real__t *ax, \
	IN__ uint__t *ai, \
	IN__ uint__t *ap);

/*create the scheduler for parallel factorization and re-factorization*/
/*return 0: suggest using parallel version*/
/*return >0: suggest using sequential version*/
/*the suggestion is only for NicsLU_Factorize_MT*/
/*NicsLU_ReFactorize_MT can always get effective speedups*/
/*if it is called repeatedly, the previous allocated memory will be freed*/
int \
	NicsLU_CreateScheduler( \
	INOUT__ SNicsLU *nicslu);

/*create threads for parallel factoriztion and re-factorization*/
/*it first calls NicsLU_DestroyThreads to destroy the previously created threads*/
int \
	NicsLU_CreateThreads( \
	INOUT__ SNicsLU *nicslu, \
	IN__ unsigned int threads, \
	IN__ bool__t check);/*whether to check the validity of #threads*/

/*bind threads to cores or unbind*/
int \
	NicsLU_BindThreads( \
	IN__ SNicsLU *nicslu, \
	IN__ bool__t unbind);

/*destroy threads. this routine is included in NicsLU_Destroy*/
int \
	NicsLU_DestroyThreads( \
	INOUT__ SNicsLU *nicslu);

/*pre-processing, including ordering and static pivoting*/
int \
	NicsLU_Analyze( \
	INOUT__ SNicsLU *nicslu);

/*LU factorization, with partial pivoting*/
/*before called, NicsLU_Analyze must be called*/
int \
	NicsLU_Factorize( \
	INOUT__ SNicsLU *nicslu);

/*LU factorization, without partial pivoting*/
/*before called, NicsLU_Factorize or NicsLU_Factorize_MT must be called at least once*/
int \
	NicsLU_ReFactorize( \
	INOUT__ SNicsLU *nicslu, \
	IN__ real__t *ax);

/*multi-threaded version of NicsLU_Factorize*/
int \
	NicsLU_Factorize_MT( \
	INOUT__ SNicsLU *nicslu);

/*multi-threaded version of NicsLU_ReFactorize*/
/*before called, NicsLU_Factorize or NicsLU_Factorize_MT must be called at least once*/
int \
	NicsLU_ReFactorize_MT( \
	INOUT__ SNicsLU *nicslu, \
	IN__ real__t *ax);

/*solve the linear system Ax=b after LU factorization*/
int \
	NicsLU_Solve( \
	INOUT__ SNicsLU *nicslu, \
	INOUT__ real__t *rhs);/*for inputs, it's b, for outputs, it's overwritten by x*/

/*when there are many zeros in b, this routine is faster than NicsLU_Solve*/
int \
	NicsLU_SolveFast( \
	INOUT__ SNicsLU *nicslu, \
	INOUT__ real__t *rhs);/*for inputs, it's b, for outputs, it's overwritten by x*/

/*when values of A are changed but the nonzero pattern is not changed, this routine resets the values*/
/*then a new LU factorization can be performed*/
/*if the nonzero pattern is also changed, call NicsLU_CreateMatrix and re-preform the whole process*/
int \
	NicsLU_ResetMatrixValues( \
	INOUT__ SNicsLU *nicslu, \
	IN__ real__t *ax);

/*refine the results. this routine is not always successful*/
int \
	NicsLU_Refine( \
	INOUT__ SNicsLU *nicslu, \
	INOUT__ real__t *x, \
	IN__ real__t *b, \
	IN__ real__t eps, \
	IN__ uint__t maxiter);/*if set to 0, then no constraint for iteration count*/

/*total memory access in LU factorization, including read and write, in bytes*/
/*call it after factorization*/
int \
	NicsLU_Throughput( \
	IN__ SNicsLU *nicslu, \
	OUT__ real__t *thr);

/*number of floating-point operations in LU factorization*/
/*call it after factorization*/
int \
	NicsLU_Flops( \
	INOUT__ SNicsLU *nicslu, \
	OUT__ real__t *flops);

/*flops of each thread*/
/*call it after factorization*/
int \
	NicsLU_ThreadLoad( \
	IN__ SNicsLU *nicslu, \
	IN__ unsigned int threads, \
	OUT__ real__t **thread_flops);

/*extract A after pre-processing*/
int \
	NicsLU_DumpA( \
	IN__ SNicsLU *nicslu, \
	OUT__ real__t **ax, \
	OUT__ uint__t **ai, \
	OUT__ uint__t **ap);

/*extract LU factors after LU factorization*/
int \
	NicsLU_DumpLU( \
	IN__ SNicsLU *nicslu, \
	OUT__ real__t **lx, \
	OUT__ uint__t **li, \
	OUT__ size_t **lp, \
	OUT__ real__t **ux, \
	OUT__ uint__t **ui, \
	OUT__ size_t **up);

/*condition number estimation*/
/*call it after factorization*/
int \
	NicsLU_ConditionNumber( \
	INOUT__ SNicsLU *nicslu, \
	OUT__ real__t *cond);

/*memory used by NICSLU, in bytes*/
/*return an approximate value*/
int \
	NicsLU_MemoryUsage( \
	IN__ SNicsLU *nicslu, \
	OUT__ real__t *memuse);


/*******************************************************************************/
/*the following routines are without the SNicsLU structure*/

/*residual error = |Ax-b|*/
/*1-norm(norm=1), 2-norm(norm=2), ¡Þ-norm(norm=other)*/
/*mode=0:row mode, mode=1:column mode*/
int \
	NicsLU_Residual( \
	IN__ uint__t n, \
	IN__ real__t *ax, \
	IN__ uint__t *ai, \
	IN__ uint__t *ap, \
	IN__ real__t *x, \
	IN__ real__t *b, \
	OUT__ real__t *error, \
	IN__ int norm, \
	IN__ int mode);

/*transpose the matrix stored in CSR or CSC format*/
int \
	NicsLU_Transpose( \
	IN__ uint__t n, \
	IN__ uint__t nnz, \
	INOUT__ real__t *ax, \
	INOUT__ uint__t *ai, \
	INOUT__ uint__t *ap);

/*sort the CSR or CSC storage*/
/*using a radix-sort like method*/
int \
	NicsLU_Sort( \
	IN__ uint__t n, \
	IN__ uint__t nnz, \
	INOUT__ real__t *ax, \
	INOUT__ uint__t *ai, \
	INOUT__ uint__t *ap);

/*merge duplicated entries in CSR/CSC*/
/*this routine also sorts the matrix*/
int \
	NicsLU_MergeDuplicateEntries( \
	IN__ uint__t n, \
	INOUT__ uint__t *nnz, \
	INOUT__ real__t **ax, \
	INOUT__ uint__t **ai, \
	INOUT__ uint__t **ap);

#ifdef __cplusplus
}
#endif

#endif
