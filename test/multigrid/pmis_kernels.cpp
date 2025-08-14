// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <core/multigrid/pmis_kernels.hpp>

using ValueType = double;
using IndexType = int;

TEST(PmisKernels, ComputeStrongDepRowWorksOnSmallMatrix)
{
    auto exec = gko::ReferenceExecutor::create();
    // Matrice 3x3:
    // [ 2  -1   0 ]
    // [ 4   3   5 ]
    // [ 0  -2   1 ]

    gko::matrix_data<ValueType, IndexType> mdata(
        gko::dim<2>{3, 3},
        {
            {0, 0, 2.0}, {0, 1, -1.0},
            {1, 0, 4.0}, {1, 1, 3.0}, {1, 2, 5.0},
            {2, 1, -2.0}, {2, 2, 1.0}
        }
    );
    auto csr = gko::matrix::Csr<ValueType, IndexType>::create(exec);
    csr->read(mdata);

    std::vector<IndexType> sparsity_rows(3, 0);
    gko::kernels::reference::pmis::compute_strong_dep_row(
        exec, csr.get(), 0.5, sparsity_rows.data());

    // La funzione attuale calcola il numero di dipendenze forti per riga (non il massimo per colonna)
    // Per la matrice di esempio:
    // Riga 0: ha 1 elemento fuori diagonale (col 1)
    // Riga 1: ha 2 elementi fuori diagonale (col 0 e col 2)
    // Riga 2: ha 1 elemento fuori diagonale (col 1)
    // Dopo la prefix sum:
    // sparsity_rows = [1, 3, 4]

    std::cout << "sparsity_rowsAAAAAAAAAAAAAAAA: ";
    for(int i=0; i < csr->get_size()[0] + 1; i++) {
        std::cout << "sparsity_rows[" << i << "] " << sparsity_rows[i] << " ";
    }
    std::cout << std::endl;

    EXPECT_EQ(sparsity_rows[0], 0);
    EXPECT_EQ(sparsity_rows[1], 1);
    EXPECT_EQ(sparsity_rows[2], 3);
    EXPECT_EQ(sparsity_rows[3], 4);
}
