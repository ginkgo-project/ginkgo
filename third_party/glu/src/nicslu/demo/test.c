#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "math.h"
#include "nicslu.h"
#include "nicslu_util.h"

void help_message() {
    printf("************************\n");
    printf("NICSLU test program V0.1\n");
    printf("Usage: test -i inputfile\n");
    // printf("[-p] for enable pivoting");
}

void print_matrix(real__t *A, uint__t dim)
{
    uint__t i, j;
        for (i = 0; i < dim; ++i)
        {
                for (j = 0; j < dim; ++j)
                        printf("%f\t",A[i * dim + j]);
                printf("\n");
        }
}

int check_zero_pivot(real__t *ax, uint__t *ai, uint__t *ap, uint__t dim, real__t tol, uint__t printout)
{
    uint__t num1 = 0;
    uint__t num2 = 0;
    uint__t i, j;
    for (i = 0; i < dim; ++i)
    {
        j = 0;
        while (j < ap[i + 1] - ap[i])
        {
            if (i == ai[ap[i]+j])
            {
                if ((((ax[ap[i]+j]) > 0) && ((ax[ap[i]+j]) < tol)) || (((ax[ap[i]+j]) < 0) && ((ax[ap[i]+j]) > -tol)))
                // if(ABS((ax[ap[i]+j])) < tol)
                {
                    if (printout)
                    {
                        printf("%d's pivot is %f, less than tol\n", i, ax[ap[i]+j] );
                    }
                    num1++;
                }
                break;            
            }
            j++;
        }
        if (j == ap[i + 1] - ap[i])
        {
            if (printout)
            {
                printf("%d's pivot is 0\n", i );  
            }
            num2++;
        }
    }
    printf("-------------Small pivot no.: %d\n", num1);
    printf("-------------Zero  pivot no.: %d\n\n", num2);
    return num1 + num2;
        
}

void check_error(real__t *A, real__t *ref, uint__t dim1, uint__t dim2, real__t Tolerance, real__t mux_error_range, uint__t printout)
{
    uint__t error_num = 0;
    uint__t col, row;
    for(col = 0; col < dim2; col++)
    {
        for(row = 0; row < dim1; row++)
        {
            // choose absolute error
            if (ref[row * dim2 + col] < mux_error_range && ref[row * dim2 + col] > -mux_error_range)
            {
                real__t error;
                error = A[row * dim2 + col] - ref[row * dim2 + col];
                if (error > Tolerance)
                {
                    if (printout)
                    {
                        printf("(%d, %d), Absolute Difference: %f\n", row, col, error);
                        printf("cal: %f; ref: %f", A[row * dim2 + col], ref[row * dim2 + col]);
                    }
                    error_num++;
                    // exit(1);         
                }
            }
            // choose relative error
            else
            {
                real__t relativeError ;
                relativeError = (A[row * dim2 + col] - ref[row * dim2 + col]) / ref[row * dim2 + col];
                if (relativeError > Tolerance)
                {
                    if (printout)
                    {
                        printf("(%d, %d), Relative Difference: %f\n", row, col, relativeError);
                        printf("cal: %f; ref: %f", A[row * dim2 + col], ref[row * dim2 + col]);
                    }   
                    error_num++;
                    // exit(1);                 
                }
            }           
        }
    }
    printf("\nError no.: %d\n",error_num);
    // cout << "\nTEST PASSED\n\n" << endl;
}

int my_DumpA(SNicsLU *nicslu, real__t **ax, uint__t **ai, uint__t **ap)
{
    uint__t n, nnz;
    real__t *ax0;
    uint__t *ai0, *ap0, *rowperm, *pinv, oldrow, start, end;
    uint__t i, j, p;

    if (NULL == nicslu || NULL == ax || NULL == ai || NULL == ap)
    {
        return -1;
    }

    if (*ax != NULL)
    {
        free(*ax);
        *ax = NULL;
    }
    if (*ai != NULL)
    {
        free(*ai);
        *ai = NULL;
    }
    if (*ap != NULL)
    {
        free(*ap);
        *ap = NULL;
    }

    n = nicslu->n;
    nnz = nicslu->nnz;
    ax0 = nicslu->ax;
    ai0 = nicslu->ai;
    ap0 = nicslu->ap;
    rowperm = nicslu->row_perm;/*row_perm[i]=j-->row i in the permuted matrix is row j in the original matrix*/
    pinv = (uint__t *)nicslu->pivot_inv;/*pivot_inv[i]=j-->column i is the jth pivot column*/

    for (i = 0; i < n; ++i)
        pinv[i] = i;


    *ax = (real__t *)malloc(sizeof(real__t)*nnz);
    *ai = (uint__t *)malloc(sizeof(uint__t)*nnz);
    *ap = (uint__t *)malloc(sizeof(uint__t)*(n+1));
    if (NULL == *ax || NULL == *ai || NULL == *ap)
    {
        goto FAIL;
    }
    (*ap)[0] = 0;

    p = 0;
    for (i=0; i<n; ++i)
    {
        oldrow = rowperm[i];
        start = ap0[oldrow];
        end = ap0[oldrow+1];
        (*ap)[i+1] = (*ap)[i] + end - start;

        for (j=start; j<end; ++j)
        {
            (*ax)[p] = ax0[j];
            (*ai)[p++] = pinv[ai0[j]];
        }
    }

    return 0;

FAIL:
    if (*ax != NULL)
    {
        free(*ax);
        *ax = NULL;
    }
    if (*ai != NULL)
    {
        free(*ai);
        *ai = NULL;
    }
    if (*ap != NULL)
    {
        free(*ap);
        *ap = NULL;
    }
    return -2;
}


int main(int argc, char *argv[])
{
    //////////////////////////////////////////////////////////////////////////////
    if (argc < 3) {
        help_message();
        return -1;
    }

    char *matrixName;
    if (strcmp(argv[1], "-i") == 0) 
    {
        if(2 > argc) 
        {
            help_message();
            return -1;
        }       
        matrixName = argv[2];
        // printf("%s\n", matrixName);
    }
    // else if (strcmp(argv[tt], "-p") == 0) {
    //     pivot = true;
    //     tt += 1;
    // }
    else {
        help_message();
        return -1;
    }



    //////////////////////////////////////////////////////////////////////////////

    int ret;
    uint__t n, nnz, i;
    // real__t *ax0;
    // uint__t *ai0, *ap0;
    SNicsLU *nicslu;
    real__t *ax;
    uint__t *ai, *ap;
    real__t *lx, *ux;
    uint__t *li, *ui;
    size_t *lp, *up;

    // ax0 = NULL;
    // ai0 = NULL;
    // ap0 = NULL;
    ax = NULL;
    ai = NULL;
    ap = NULL;

    lx = ux = NULL;
    li = ui = NULL;
    lp = up = NULL;

    // real__t *A0;
    // real__t *L0, *U0;
    // real__t *LU0;

    // uint__t col, col_i;
    int error_i, error_u;

    real__t check_pivot_tol = 0.001;


    nicslu = (SNicsLU *)malloc(sizeof(SNicsLU));
    NicsLU_Initialize(nicslu);

    ret = NicsLU_ReadTripletColumnToSparse(matrixName, &n, &nnz, &ax, &ai, &ap);
    if (ret == NICSLU_MATRIX_INVALID)
    {    
        printf("Read invalid matrix\n");
        goto EXIT;
    }
    else if (ret == NICSLU_FILE_CANNOT_OPEN) 
    {    
        printf("File cannot open\n");
        goto EXIT;
    }
    else if (ret != NICS_OK) 
    {    
        printf("Open file error\n");
        goto EXIT;
    }

    printf("Original matrix: \n");
    // A0=(real__t *)malloc(n*n*sizeof(real__t));
    // for (col = 0; col < n; col++)
    //     for (col_i = 0; col_i < ap0[col+1]-ap0[col]; col_i++)
    //         A0[ai0[ap0[col]+col_i] * n + col] = ax0[ap0[col]+col_i];
    // print_matrix(A0, n);
    check_zero_pivot(ax, ai, ap, n, check_pivot_tol, 0);
    

    NicsLU_CreateMatrix(nicslu, n, nnz, ax, ai, ap);
    nicslu->cfgf[0] = 1;

    // printf("%d\n", nicslu->n);
    // printf("%d\n", nicslu->nnz);
    // printf("ax:\n");
    // for (i = 0; i < nnz; ++i)
    // {
    //     printf("%d: %f\n", i, nicslu->ax[i]);
    // }

    NicsLU_Analyze(nicslu);
    printf("analysis time: %.8g\n", nicslu->stat[0]);

    // printf("ax:\n");
    // for (i = 0; i < nnz; ++i)
    // {
    //     printf("%d: %f\n", i, nicslu->ax[i]);
    // }


    // printf("p:\n");
    // for (i = 0; i < n; i ++)
    // {
    //   printf("no. %d: %d\n", i, nicslu->pivot[i]);
    // }
    // printf("pinv:\n");
    // for (i = 0; i < n; i ++)
    // {
    //   printf("no. %d: %d\n", i, nicslu->pivot_inv[i]);
    // }
    // printf("rowperm:\n");
    // for (i = 0; i < n; i ++)
    // {
    //   printf("no. %d: %d\n", i, nicslu->row_perm[i]);
    // }


    my_DumpA(nicslu, &ax, &ai, &ap);

    // printf("Intermediate matrix: \n");
    // A0=(real__t *)malloc(n*n*sizeof(real__t));
    // for (col = 0; col < n; col++)
    //     for (col_i = 0; col_i < n; col_i++)
    //         A0[col * n + col_i] = 0;
    // for (col = 0; col < nicslu->n; col++)
    //     for (col_i = 0; col_i < ap[col+1]-ap[col]; col_i++)
    //         A0[ai[ap[col]+col_i] * n + col] = ax[ap[col]+col_i];
    // print_matrix(A0, n);

    // printf("ax:\n");
    // for (i = 0; i < nicslu->nnz; i ++)
    // {
    //   printf("no. %d: %f\n", i, ax[i]);
    // }

    // printf("ai:\n");
    // for (i = 0; i < nicslu->nnz; i ++)
    // {
    //   printf("no. %d: %d\n", i, ai[i]);
    // }

    // printf("ap:\n");
    // for (i = 0; i < n + 1; i ++)
    // {
    //   printf("no. %d: %d\n", i, ap[i]);
    // }

    check_zero_pivot(ax, ai, ap, n, check_pivot_tol, 0);


    NicsLU_Factorize(nicslu);
    printf("factorization time: %.8g\n", nicslu->stat[1]);


// /////////////////////////////////////////////////////////////////////////////////////////////

//     ax = NULL;
//     ai = NULL;
//     ap = NULL;
    NicsLU_DumpA(nicslu, &ax, &ai, &ap);

    printf("Final matrix: \n");
    // A0=(real__t *)malloc(n*n*sizeof(real__t));
    // for (col = 0; col < n; col++)
    //     for (col_i = 0; col_i < ap[col+1]-ap[col]; col_i++)
    //         A0[ai[ap[col]+col_i] * n + col] = ax[ap[col]+col_i];
    // print_matrix(A0, n);
    check_zero_pivot(ax, ai, ap, n, check_pivot_tol, 0);

//////////////////////////////////////////////////////////////////////////////
    NicsLU_DumpLU(nicslu, &lx, &li, &lp, &ux, &ui, &up);   
    error_i = 0;
    error_u = 0;

    uint__t real_l_nnz, real_u_nnz;
    real_l_nnz = nicslu->u_nnz;
    real_u_nnz = nicslu->l_nnz;



    // L0=(real__t *)malloc(n*n*sizeof(real__t));
    // for (col = 0; col < n; col++)
    //     for (col_i = 0; col_i < up[col+1]-up[col]; col_i++)
    //         L0[ui[up[col]+col_i] * n + col] = ux[up[col]+col_i];
    // printf("L0:\n");
    // print_matrix(L0, n);

    for(i = 0; i < real_l_nnz; i++)
        if(isnan(lx[i]) || isinf(lx[i])) 
        // if (lx[i] != lx[i])
        {
            // printf("Pivoting failed\n" );
            // printf("%d: %f\n", i, lx[i]);
            error_i++;
        }
    // printf("%f\n", ux[real_l_nnz-1]);

    printf("checked %d\n", i);
    printf("l_nnz is %d, error num is %d\n", real_l_nnz, error_i);


    // U0=(real__t *)malloc(n*n*sizeof(real__t));
    // for (col = 0; col < n; col++)
    //     for (col_i = 0; col_i < lp[col+1]-lp[col]; col_i++)
    //         U0[li[lp[col]+col_i] * n + col] = lx[lp[col]+col_i];
    // printf("U0:\n");
    // print_matrix(U0, n);

    for(i = 0; i < real_u_nnz; i++)
        if(isnan(ux[i]) || isinf(ux[i])) 
        // if (ux[i] != ux[i])
        {
            // printf("Pivoting failed\n" );
            // printf("%d: %f\n", i, ux[i]);
            error_u++;
        }
    // printf("%f\n", lx[real_u_nnz-1]);

    printf("checked %d\n", i);
    printf("u_nnz is %d, error num is %d\n", real_u_nnz, error_u);

    // printf("p:\n");
    // for (i = 0; i < n; i ++)
    // {
    //   printf("no. %d: %d\n", i, nicslu->pivot[i]);
    // }

    // printf("pinv:\n");
    // for (i = 0; i < n; i ++)
    // {
    //   printf("no. %d: %d\n", i, nicslu->pivot_inv[i]);
    // }


    // printf("Check A ?= LU\n");
  //   LU0=(real__t *)malloc(n*n*sizeof(real__t));

  //   const real__t relativeTolerance = 1e-6;
  //   uint__t row;

  // for(row = 0; row < n; ++row) {
  //   for(col = 0; col < n; ++col) {
  //     real__t sum = 0;
  //     for(i = 0; i < n; ++i) {
  //       sum += L0[row*n + i]*U0[i*n + col];
  //     }
  //     LU0[row*n + col] = sum;
  //   }
  // }

  //   check_error(LU0, A0, n, n, relativeTolerance, 1, 1);



    // int dim;
    // dim = nicslu->n;
    



    // uint__t col, col_i;
    // for (col = 0; col < nicslu->n; col++)
    //     for (col_i = 0; col_i < ap[col+1]-ap[col]; col_i++)
    //         A0[ai[ap[col]+col_i] * n + col] = ax[ap[col]+col_i];
    // printf("A0:\n");
    // print_matrix(A0, n);


    // for (col_i = 0; col_i < nicslu->l_nnz; ++col_i)
    // {
    //     printf("%d: %f\n", col_i, lx[col_i]);
    // }


    // for (col_i = 0; col_i < nicslu->u_nnz; ++col_i)
    // {
    //     printf("%d: %f\n", col_i, ux[col_i]);
    // }

    

    // printf("A:\n");
    // for (i = 0; i < nicslu-> nnz; i ++)
    // {
    //   printf("no. %d: %f, %f\n", i, ax[i], ax0[i]);
    // }

    // printf("ai:\n");
    // for (i = 0; i < nicslu-> nnz; i ++)
    // {
    //   printf("no. %d: %d\n", i, ai[i]);
    // }

    // printf("ap:\n");
    // for (i = 0; i < nicslu->n + 1; i ++)
    // {
    //   printf("no. %d: %d\n", i, ap[i]);
    // }

    // printf("L:\n");
    // for (i = 0; i < nicslu-> l_nnz; i ++)
    // {
    //   printf("no. %d: %f\n", i, lx[i]);
    // }

    // printf("li:\n");
    // for (i = 0; i < nicslu-> l_nnz; i ++)
    // {
    //   printf("no. %d: %d\n", i, li[i]);
    // }

    // printf("lp:\n");
    // for (i = 0; i < nicslu->n + 1; i ++)
    // {
    //   printf("no. %d: %d\n", i, lp[i]);
    // }

    // printf("U:\n");
    // for (i = 0; i < nicslu-> u_nnz; i ++)
    // {
    //   printf("no. %d: %f\n", i, ux[i]);
    // }

    // printf("ui:\n");
    // for (i = 0; i < nicslu-> u_nnz; i ++)
    // {
    //   printf("no. %d: %d\n", i, ui[i]);
    // }

    // printf("up:\n");
    // for (i = 0; i < nicslu->n + 1; i ++)
    // {
    //   printf("no. %d: %d\n", i, up[i]);
    // }










    // free(ax);
    // free(ai);
    // free(ap);

    // free(lx);
    // free(li);
    // free(lp);
    // free(ux);
    // free(ui);
    // free(up);

    // int t_c;
    // int e_num = 0;
    // int lu_nnz = 0;

    // real__t *lu;
    // lu = nicslu->lu_array;
    // lu_nnz = nicslu->lu_nnz;

    // for(t_c = 0; t_c < lu_nnz; t_c++)
    //     if(isnan(lu[t_c]) || isinf(lu[t_c])) 
    //     // if (lu[t_c] != lu[t_c])
    //     {
    //         // printf("Pivoting failed\n" );
    //         printf("%d: %f\n", t_c, lu[t_c]);
    //         e_num++;
    //         // break;
    //     //  exit(-1);
    //     }
    // printf("lu_nnz is %d, error num is %d\n", lu_nnz, e_num);

    // for(t_c = 0; t_c < nnz; t_c++)
    // {
    //     printf("%d: %f \n" ,t_c, lu[t_c]);
    // }

    // NicsLU_ReFactorize(nicslu, ax);
    // printf("re-factorization time: %.8g\n", nicslu->stat[2]);

    // NicsLU_Solve(nicslu, x);
    // printf("substitution time: %.8g\n", nicslu->stat[3]);

    // NicsLU_Residual(n, ax, ai, ap, x, b, &err, 1, 0);
    // printf("Ax-b (1-norm): %.8g\n", err);
    // NicsLU_Residual(n, ax, ai, ap, x, b, &err, 2, 0);
    // printf("Ax-b (2-norm): %.8g\n", err);
    // NicsLU_Residual(n, ax, ai, ap, x, b, &err, 0, 0);
    // printf("Ax-b (infinite-norm): %.8g\n", err);

    // printf("NNZ(L+U-I): %ld\n", nicslu->lu_nnz);

    // NicsLU_Flops(nicslu, NULL);
    // NicsLU_Throughput(nicslu, NULL);
    // NicsLU_ConditionNumber(nicslu, NULL);
    // printf("flops: %.8g\n", nicslu->stat[5]);
    // printf("throughput (bytes): %.8g\n", nicslu->stat[12]);
    // printf("condition number: %.8g\n", nicslu->stat[6]);
    // NicsLU_MemoryUsage(nicslu, NULL);
    // printf("memory (Mbytes): %.8g\n", nicslu->stat[21]/1024./1024.);
    
EXIT:
    NicsLU_Destroy(nicslu);
    free(ax);
    free(ai);
    free(ap);
    free(nicslu);
    //free(x);
    return 0;
}
