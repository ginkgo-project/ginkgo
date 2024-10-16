// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ginkgo/core/stop/batch_stop_enum.hpp>


namespace gko {
namespace kernels {

// TODO: update when splitting compilation
constexpr bool cg_no_shared_vecs = true;


namespace batch_cg {


template <typename RealType>
struct settings {
    static_assert(std::is_same<RealType, remove_complex<RealType>>::value,
                  "Template parameter must be a real type");
    int max_iterations;
    RealType residual_tol;
    batch::stop::tolerance_type tol_type;
};

template <typename ValueType>
inline int local_memory_requirement(const int num_rows, const int num_rhs)
{
    return (5 * num_rows * num_rhs + 3 * num_rhs) * sizeof(ValueType) +
           2 * num_rhs * sizeof(typename gko::remove_complex<ValueType>);
}


}  // namespace batch_cg
}  // namespace kernels
}  // namespace gko
