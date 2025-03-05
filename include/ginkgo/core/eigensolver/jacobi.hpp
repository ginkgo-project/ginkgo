// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/matrix/dense.hpp>

namespace gko {
namespace experimental {
namespace eigensolver {

template <typename ValueType>
void rotation(std::shared_ptr<gko::matrix::Dense<ValueType>>
                  matrix, ) template <typename ValueType>
std::shared_ptr<gko::matrix::Dense<ValueType>> jacobi(
    std::shared_ptr<gko::matrix::Dense<ValueType>> matrix, int block_size,
    double tol, int max_iter)
{
    auto exec = matrix->get_executor();
    auto small_eigen_size = 2 * block_size;
    GKO_THROW_IF_INVALID(
        matrix->get_size(0) % small_eigen_size == 0,
        "2 * blocksize must be devided by the matrix size now");

    // int i, j, ii, jj,        // counters
    //     k = 2 * block_size,  // size of the local systems
    //     m, r, s,             // indices for blocks to be considered
    //     mythreadnum,         // my thread ID
    //     max_iter = *iter,    // maximum number of sweeps
    //     lwork = MAX(k * k - k, 3 * k - 1), info;  // variables for dsyev

    // double *pA, *pV,              // pointers for A and V
    //     alpha = 1.0, beta = 0.0,  // scalars for dgemm
    //     **dbuff, *pdbuff;         // buffer and pointer for dgemm


    // // auxiliary buffer for GEMM, required for every thread
    // dbuff = (double**)malloc((size_t)m * sizeof(double*));
    // // index buffers for qsort
    // int **ind = (int**)malloc((size_t)m * sizeof(int*)), *pind;
    // int **stack = (int**)malloc((size_t)m * sizeof(int*)), *pstack;
    // for (i = 0; i < m; i++) {
    //     // each buffer must have size n x k
    //     dbuff[i] = (double*)malloc((size_t)n * k * sizeof(double));
    //     // each integer array must have size k
    //     ind[i] = (int*)malloc((size_t)k * sizeof(int));
    //     stack[i] = (int*)malloc((size_t)k * sizeof(int));
    // }  // end for i

    // number of parallel problems to be solved simultaneously
    auto num_prob_per_stage = matrix->get_size()[0] / small_eigen_size;
    // // initial partitioning (0,1), (2,3), (4,5), ...
    std::vector<std::pair<int>> coord(num_prob_per_stage);
    for (size_t i = 0; i < num_prob_per_stage; i++) {
        coord.at(i) = std::tie(2 * i, 2 * i + 1);
    }

    // initial eigenvector matrix = I
    auto eigenvecotr = gko::share(
        gko::matrix::Dense<ValueType>::create(exec, matrix->get_size()));
    matrix->fill(0.0);
    for (size_t i = 0; i < matrix->get_size()[0]; i++) {
        matrix->at(i, i) = one<ValueType>();
    }

    double norm_diagonal = 0.0;
    double norm_off_diagonal = 0.0;
    for (size_t row = 0; row < matrix->get_size()[0]; row++) {
        for (size_t col = 0; col < matrix->get_size()[1]; col++) {
            if (row == col) {
                norm_diagonal += matrix->at(row, col) * matrix->at(row, col);
            } else {
                norm_off_diagonal +=
                    matrix->at(row, col) * matrix->at(row, col);
            }
        }
    }


    // // auxiliary space for m local eigenpair computations
    // // local eigenvectors after dsyev, later eigenvalues
    // double **DD = (double**)malloc((size_t)m * sizeof(double*)), *pD;
    // // copy of the local diagonal entries, stored because their order
    // // imposes the order of the eigenvalues
    // double** AA = (double**)malloc((size_t)m * sizeof(double*));
    // // local eigenvectors after dsyev and reordering
    // double** VV = (double**)malloc((size_t)m * sizeof(double*));
    // for (i = 0; i < m; i++) {
    //     DD[i] = (double*)malloc((size_t)k * k * sizeof(double));
    //     AA[i] = (double*)malloc((size_t)k * sizeof(double));
    //     VV[i] = (double*)malloc((size_t)MAX(k * k, 4 * k - 1) *
    //     sizeof(double));
    // }  // end for i

    int iter = 0;
    auto local_problem = gko::share(gko::matrix::Dense<ValueType>::create(
        exec, dim<2>{small_eigen_size, small_eigen_size}));
    while (norm_off_diagonal > tol * norm_diagonal && iter < max_iter) {
        // hist[*iter] = nrm_offdgl / nrm_dgl;
        iter = iter + 1;

        // compute yet another sweep
        for (size_t i = 0; i < 2 * num_prob_per_stage - 1; i++) {
            // this loop could in principle be performed in parallel
            // but multiplication of D must be synchronized, at first
            // multiplication from the right with VV is done in common
            // after that, when all multiplications D(:,[I J])*VV are
            // finished, then all multiplications VV'*D([I J],:) can
            // be computed
            // first stage: solve small eigenvalue problem and update norms
            double mynrm_diagonal = 0.0;
            double mynrm_off_diagonal = 0.0;
            // first sub block  I=r*block_size,...,(r+1)*block_size-1
            // second sub block J=s*block_size,...,(s+1)*block_size-1
            // we only deal with block_size = 1 now


#pragma omp parallel for default(none) shared(                                 \
        m, ind, stack, coord1, coord2, A, block_size, DD, n, k, AA, VV, lwork, \
            stdout) private(mythreadnum, pind, pstack, r, s, pA, pD, jj, ii,   \
                                nrm_dgl_loc, nrm_offdgl_loc, info, pV)         \
    reduction(+ : mynrm_dgl, mynrm_offdgl)
            for (size_t j = 0; j < m; j++) {
                mythreadnum = omp_get_thread_num();
                // local index arrays used for qsort
                pind = ind[mythreadnum];
                pstack = stack[mythreadnum];
                // extract block number (r,s)
                r = coord1[j];
                s = coord2[j];
                // solve eigenvalue problem of block (r,s)
                // first sub block  I=r*block_size,...,(r+1)*block_size-1
                // second sub block J=s*block_size,...,(s+1)*block_size-1
                // local symmetric matrix block DD=A([I J],[I J]);

                auto block_coord = coord.at(j);
                // 1. local(0:bs-1,0:bs-1)=A(I,I)
                // 2. local(0:bs-1,bs:end)=A(I,J)
                // 3. local(bs:end,0:bs-1)=A(J,I)
                // 4. local(bs:end,bs:end)=A(J,J)
                for (int row_block = 0; row_block < 2; row_block++) {
                    auto row_block_idx =
                        row_block == 0 ? block_coord.first : block_coord.second;
                    for (int col_block = 0; col_block < 2; col_block++) {
                        auto col_block_idx = col_block == 0
                                                 ? block_coord.first
                                                 : block_coord.second;
                        // TODO: replaced by submatrix copy_from or maybe we can
                        // operate on the original matrix directly?
                        for (size_t row = 0; row < block_size; row++) {
                            for (size_t col = 0; col < block_size; col++) {
                                local_problem->at(
                                    row + row_block * block_size,
                                    col + col_block * block_size) =
                                    matrix->at(
                                        row_block_idx * block_size + row,
                                        col_block_idx * block_size + col);
                            }
                        }
                    }
                }
                double norm_diagonal_local = 0.0;
                double norm_off_diagonal_local = 0.0;
                // local square sum of the diagonal entries
                // local square sum of the off-diagonal entries
                for (size_t row = 0; row < local_problem->get_size()[0];
                     row++) {
                    for (size_t col = 0; col < local_problem->get_size()[1];
                         col++) {
                        if (row == col) {
                            norm_diagonal += local_problem->at(row, col) *
                                             local_problem->at(row, col);
                        } else {
                            norm_off_diagonal += local_problem->at(row, col) *
                                                 local_problem->at(row, col);
                        }
                    }
                }

                // locally downdate squared norm of the diagonal entries
                mynrm_diagonal -= norm_diagonal_local;
                // locally downdate squared norm of the off-diagonal entries
                mynrm_off_diagonal -= norm_off_diagonal_local;

                // TODO? any reason? copy diagonal entries of A([I J],[I J])
                // pA = AA[j];
                // pD = DD[j];
                // for (jj = 0; jj < k; jj++, pA++, pD += k + 1) *pA = *pD;

                // solve local eigenvalue problem
                // [VV{j},DD{j}]=schur(AA);
                pV = VV[j];
                pD = DD[j];
                dsyev_("V", "L", &k, pD, &k, pV, pV + k, &lwork, &info, 1, 1);
                if (info < 0) {
                    printf("DSYEV: the %d-th argument had an illegal value\n",
                           -info);
                    exit(1);
                } else if (info > 0) {
                    printf(
                        "DSYEV: the algorithm failed to converge; %d "
                        "off-diagonal elements of an intermediate tridiagonal "
                        "form did not converge to zero.\n",
                        info);
                    exit(1);
                }  // end if-else if

                // find out which eigenvalues are closest to the diagonal
                // entries of AA. This is necessary to make sure that VV tends
                // to I when AA tends to be diagonal natural ordering
                for (jj = 0; jj < k; jj++) pind[jj] = jj;
                // sort diagonal entries in ascending order
                qsort_(AA[j], pind, pstack, &k);
                // now pind refers to permuting the original diagonal entries
                // in ascending order
                // use pstack for inverse permutation
                for (jj = 0; jj < k; jj++) pstack[pind[jj]] = jj;
                // reorder eigenpairs and eigenvalues according to inverse
                // permutation
                // copy reordered eigenvalues to AA[j]
                pV = VV[j];
                pA = AA[j];
                for (jj = 0; jj < k; jj++) pA[jj] = pV[pstack[jj]];
                // copy reordered eigenvectors to VV[j]
                pV = VV[j];
                pD = DD[j];
                for (jj = 0; jj < k; jj++, pV += k)
                    memcpy(pV, pD + pstack[jj] * k, k * sizeof(double));

                // copy eigenvalues back to DD[j]
                memcpy(pD, pA, k * sizeof(double));

                // locally update squared norm of the diagonal entries
                nrm_dgl_loc = 0.0;
                for (jj = 0; jj < k; jj++, pD++) nrm_dgl_loc += *pD * *pD;
                mynrm_dgl += nrm_dgl_loc;
            }  // end for j
            // end omp parallel for
            // globally downdate squared norm of the diagonal entries
            nrm_dgl += mynrm_dgl;
            // globally downdate squared norm of the off-diagonal entries
            nrm_offdgl += mynrm_offdgl;

            // second stage: multiplication from the right
#pragma omp parallel for default(none)                                     \
    shared(m, dbuff, coord1, coord2, V, block_size, n, k, alpha, beta, VV, \
           A) private(mythreadnum, pdbuff, r, s, pV, jj, pA)
            for (j = 0; j < m; j++) {
                mythreadnum = omp_get_thread_num();
                pdbuff = dbuff[mythreadnum];
                // extract block number (r,s)
                r = coord1[j];
                s = coord2[j];
                // first sub block  I=r*block_size,...,(r+1)*block_size-1
                // second sub block J=s*block_size,...,(s+1)*block_size-1

                // update eigenvector matrix
                // V(:,[I J])=V(:,[I J])*VV{j};
                // copy V(:,[I J]) to dbuff
                pV = *V + r * block_size * n;
                for (jj = 0; jj < block_size; jj++)
                    memcpy(pdbuff + jj * n, pV + jj * n, n * sizeof(double));
                pV = *V + s * block_size * n;
                for (jj = 0; jj < block_size; jj++)
                    memcpy(pdbuff + (jj + block_size) * n, pV + jj * n,
                           n * sizeof(double));

                // V(:,I) <- 1 * dbuff * VV{j}(:,1:bs) + 0 * V(:,I)
                dgemm_("n", "n", &n, &block_size, &k, &alpha, pdbuff, &n, VV[j],
                       &k, &beta, *V + r * block_size * n, &n, 1, 1);
                // V(:,J) <- 1 * dbuff * VV{j}(:,bs+1:2bs) + 0 * V(:,J)
                dgemm_("n", "n", &n, &block_size, &k, &alpha, pdbuff, &n,
                       VV[j] + block_size * k, &k, &beta,
                       *V + s * block_size * n, &n, 1, 1);
                // update eigenvalue matrix
                // A(:,[I J])=A(:,[I J])*VV{j};
                // copy A(:,[I J]) to dbuff
                pA = *A + r * block_size * n;
                for (jj = 0; jj < block_size; jj++)
                    memcpy(pdbuff + jj * n, pA + jj * n, n * sizeof(double));
                pA = *A + s * block_size * n;
                for (jj = 0; jj < block_size; jj++)
                    memcpy(pdbuff + (jj + block_size) * n, pA + jj * n,
                           n * sizeof(double));

                // A(:,I) <- 1 * dbuff * VV{j}(:,1:bs) + 0 * A(:,I)
                dgemm_("n", "n", &n, &block_size, &k, &alpha, pdbuff, &n, VV[j],
                       &k, &beta, *A + r * block_size * n, &n, 1, 1);
                // A(:,J) <- 1 * dbuff * VV{j}(:,bs+1:2bs) + 0 * A(:,J)
                dgemm_("n", "n", &n, &block_size, &k, &alpha, pdbuff, &n,
                       VV[j] + block_size * k, &k, &beta,
                       *A + s * block_size * n, &n, 1, 1);
            }  // end for j
               // end omp parallel for

            // third stage: multiplication of A from the left
#pragma omp parallel for default(none)                               \
    shared(m, dbuff, coord1, coord2, A, block_size, k, n, alpha, VV, \
           beta) private(mythreadnum, pdbuff, r, s, pA, ii, jj)
            for (j = 0; j < m; j++) {
                mythreadnum = omp_get_thread_num();
                pdbuff = dbuff[mythreadnum];
                // extract block number (r,s)
                r = coord1[j];
                s = coord2[j];
                // first sub block  I=r*block_size,...,(r+1)*block_size-1
                // second sub block J=s*block_size,...,(s+1)*block_size-1
                // update eigenvalue matrix
                // A([I J],:)=VV{j}'*A([I J],:);
                // copy A([I J],:) to dbuff'
                pA = *A + r * block_size;
                ii = 1;
                for (jj = 0; jj < block_size; jj++)
                    dcopy_(&n, pA + jj, &n, pdbuff + jj * n, &ii);
                pA = *A + s * block_size;
                for (jj = 0; jj < block_size; jj++)
                    dcopy_(&n, pA + jj, &n, pdbuff + (jj + block_size) * n,
                           &ii);

                // A(I,:) <- 1 * VV{j}(:,1:bs)' * dbuff'  + 0 * A(I,:)
                dgemm_("T", "T", &block_size, &n, &k, &alpha, VV[j], &k, pdbuff,
                       &n, &beta, *A + r * block_size, &n, 1, 1);
                // A(J,:) <- 1 * VV{j}(:,bs+1:2bs)'* dbuff' + 0 * A(J,:)
                dgemm_("T", "T", &block_size, &n, &k, &alpha,
                       VV[j] + block_size * k, &k, pdbuff, &n, &beta,
                       *A + s * block_size, &n, 1, 1);
            }  // end for j
               // end omp parallel for

            // fourth stage: diagonal blocks and new coordinates
#pragma omp parallel for default(none) shared( \
    m, coord1, coord2, A, block_size, n, DD) private(r, s, pA, pD, jj, ii)
            for (j = 0; j < m; j++) {
                // extract block number (r,s)
                r = coord1[j];
                s = coord2[j];
                // first sub block  I=r*block_size,...,(r+1)*block_size-1

                // update eigenvalue matrix from local to matrix (diagoanl =
                // eigenvalue, others are zero)
                auto block_coord = coord.at(j);
                for (int row_block = 0; row_block < 2; row_block++) {
                    auto row_block_idx =
                        row_block == 0 ? block_coord.first : block_coord.second;
                    for (int col_block = 0; col_block < 2; col_block++) {
                        auto col_block_idx = col_block == 0
                                                 ? block_coord.first
                                                 : block_coord.second;
                        // TODO: replaced by submatrix copy_from or maybe we can
                        // operate on the original matrix directly?
                        for (size_t row = 0; row < block_size; row++) {
                            for (size_t col = 0; col < block_size; col++) {
                                if (row == col) {
                                    matrix->at(
                                        row_block_idx * block_size + row,
                                        col_block_idx * block_size + col) =
                                        local_problem->at(
                                            row + row_block * block_size,
                                            col + col_block * block_size);
                                } else {
                                    matrix->at(row_block_idx * block_size + row,
                                               col_block_idx * block_size +
                                                   col) = zero<ValueType>();
                                }
                            }
                        }
                    }
                }

                auto update_coord = [small_eigen_size](auto idx) {
                    // consider idx + 1
                    // 1 -> 1
                    // even -> try +2, if not, idx+1 = 2 * small_eigen_size - 1
                    // odd -> try -2, if it is less than 3, idx + 1 = 2
                    if (idx + 1 != 1) {
                        // even case
                        if ((idx + 1) % 2 == 0) {
                            // increase by 2 if possible
                            if (idx + 1 < 2 * small_eigen_size) {
                                idx += 2;
                            } else {  // => idx+1=2*small_eigen_size
                                idx = 2 * m - 2;
                            }
                        } else {  // odd case
                            // decrease by 2 if possible
                            if (idx + 1 > 3) {
                                idx -= 2;
                            } else {  // => idx+1=3
                                idx = 1;
                            }
                        }
                    }
                };
                // new coord for next sweep
                coord.at(j) = std::tie(
                    update_coord(block_coord.first, block_coord.second));
            }  // end for j
        }      // end for i
    }          // end while
    hist[*iter] = nrm_offdgl / nrm_dgl;


    free(coord1);
    free(coord2);
    for (i = 0; i < m; i++) {
        free(DD[i]);
        free(VV[i]);
        free(AA[i]);
    }  // end for i
    free(DD);
    free(VV);
    free(AA);
    m = omp_get_max_threads();
    for (i = 0; i < m; i++) {
        free(dbuff[i]);
        free(ind[i]);
        free(stack[i]);
    }
    free(dbuff);
    free(ind);
    free(stack);

    // remove potential additional entries
    l = nn % (2 * block_size);
    if (l) {
        m = 2 * block_size - l;
        double nrm2;
        r = 0;
        s = n;
        j = 0;
        pV = *V;
        pA = *A;
        ii = 1;
        while (j < s) {
            // nrm2 <-||V(1:nn,j)||_2
            nrm2 = dnrm2_(&nn, pV + j * n, &ii);
            // this is an additional trivial eigenvector caused by augmentation
            if (nrm2 <= 1.0e-1) {
                // shuffle last column to column j
                s--;
                r++;
                memcpy(pV + j * n, pV + s * n, n * sizeof(double));
                pA[j + j * n] = pA[s + s * n];
                if (r == m) break;
            }  // end if
            else
                j++;
        }  // end while
        // now compress V and A to size nn x nn
        ii = m;
        pV = *V + n;
        pA = *A + n;
        for (j = 1; j < nn; j++, ii += m, pA += n, pV += n) {
            memmove(pA - ii, pA, nn * sizeof(double));
            memmove(pV - ii, pV, nn * sizeof(double));
        }  // end for j
        // restore correct n
        n = nn;
    }  // end if l


    // for clearness, sort eigenvalues and eigenvectors such that the
    // eigenvalues are ascending
    double* D = (double*)malloc((size_t)n * sizeof(double));
    int* perm = (int*)malloc((size_t)n * sizeof(int));
    int* ibuffer = (int*)malloc((size_t)n * sizeof(int));
    pA = *A;
    for (j = 0; j < n; j++, pA += n + 1) {
        D[j] = *pA;
        perm[j] = j;
    }  // end for j
    // sort eigenvalues in ascending order
    qsort_(D, perm, ibuffer, &n);
    // transfer sorted eigenvalues
    pA = *A;
    for (j = 0; j < n; j++, pA += n + 1) *pA = D[j];
    // copy associated permuted eigenvector to a buffer
    double* dbuffer = (double*)malloc((size_t)n * n * sizeof(double));
    pV = *V;
    for (j = 0; j < n; j++)
        memcpy(dbuffer + j * n, pV + perm[j] * n, n * sizeof(double));
    // rewrite buffer
    memcpy(pV, dbuffer, n * n * sizeof(double));

    free(D);
    free(perm);
    free(ibuffer);
    free(dbuffer);


    if (*iter >= max_iter)
        return 1;
    else
        return 0;
}
}  // namespace eigensolver
}  // namespace experimental
}  // namespace gko
