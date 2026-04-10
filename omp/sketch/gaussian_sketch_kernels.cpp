// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/sketch/gaussian_sketch_kernels.hpp"

#include <random>

#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace kernels {
namespace omp {
namespace gaussian_sketch {


template <typename ValueType>
void generate(std::shared_ptr<const DefaultExecutor> exec,
              matrix::view::dense<ValueType> sketch_matrix, uint64 seed)
{
    auto k = sketch_matrix.size[0];
    auto m = sketch_matrix.size[1];
    auto scale = 1.0 / sqrt(static_cast<double>(k));
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> dist(0.0, scale);
    for (size_type row = 0; row < k; ++row) {
        for (size_type col = 0; col < m; ++col) {
            sketch_matrix(row, col) =
                static_cast<remove_complex<ValueType>>(dist(rng));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GAUSSIAN_SKETCH_GENERATE);


}  // namespace gaussian_sketch
}  // namespace omp
}  // namespace kernels
}  // namespace gko
