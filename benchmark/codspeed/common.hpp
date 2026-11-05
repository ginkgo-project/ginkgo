// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_BENCHMARK_CODSPEED_COMMON_HPP_
#define GKO_BENCHMARK_CODSPEED_COMMON_HPP_


#include <memory>

#include <benchmark/benchmark.h>

#include <ginkgo/ginkgo.hpp>


namespace gko_codspeed {


using value_type = double;
using index_type = gko::int32;
using csr = gko::matrix::Csr<value_type, index_type>;
using vec = gko::matrix::Dense<value_type>;


inline std::shared_ptr<gko::Executor> exec()
{
    static auto ref = gko::ReferenceExecutor::create();
    return ref;
}


/**
 * Assembles the 5-point finite difference Laplacian on a grid x grid domain.
 * The result is symmetric positive definite with grid^2 rows.
 */
inline gko::matrix_data<value_type, index_type> laplacian_2d(index_type grid)
{
    const auto n = static_cast<gko::size_type>(grid) * grid;
    gko::matrix_data<value_type, index_type> data{gko::dim<2>{n, n}};
    for (index_type i = 0; i < grid; ++i) {
        for (index_type j = 0; j < grid; ++j) {
            const auto row = i * grid + j;
            if (i > 0) {
                data.nonzeros.emplace_back(row, row - grid, -1.0);
            }
            if (j > 0) {
                data.nonzeros.emplace_back(row, row - 1, -1.0);
            }
            data.nonzeros.emplace_back(row, row, 4.0);
            if (j < grid - 1) {
                data.nonzeros.emplace_back(row, row + 1, -1.0);
            }
            if (i < grid - 1) {
                data.nonzeros.emplace_back(row, row + grid, -1.0);
            }
        }
    }
    return data;
}


template <typename MatrixType = csr>
std::shared_ptr<MatrixType> laplacian_matrix(index_type grid)
{
    auto mtx = gko::share(MatrixType::create(exec()));
    mtx->read(laplacian_2d(grid));
    return mtx;
}


/** Deterministic dense vector with rows x cols entries. */
inline std::unique_ptr<vec> make_vector(gko::size_type rows,
                                        gko::size_type cols = 1)
{
    auto v = vec::create(exec(), gko::dim<2>{rows, cols});
    for (gko::size_type i = 0; i < rows; ++i) {
        for (gko::size_type j = 0; j < cols; ++j) {
            v->at(i, j) =
                1.0 + static_cast<value_type>((i * 7 + j * 13) % 17) / 17.0;
        }
    }
    return v;
}


}  // namespace gko_codspeed


#endif  // GKO_BENCHMARK_CODSPEED_COMMON_HPP_
