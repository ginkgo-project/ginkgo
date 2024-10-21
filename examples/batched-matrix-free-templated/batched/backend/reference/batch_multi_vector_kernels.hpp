// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_REFERENCE_BASE_BATCH_MULTI_VECTOR_KERNELS_HPP_
#define GKO_REFERENCE_BASE_BATCH_MULTI_VECTOR_KERNELS_HPP_


#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>

#include "reference/base/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace batch_template {
namespace batch_single_kernels {


template <typename ValueType>
inline void scale_kernel(
    const batch::multi_vector::batch_item<const ValueType>& alpha,
    const batch::multi_vector::batch_item<ValueType>& x)
{
    if (alpha.num_rhs == 1) {
        for (int i = 0; i < x.num_rows; ++i) {
            for (int j = 0; j < x.num_rhs; ++j) {
                x.values[i * x.stride + j] *= alpha.values[0];
            }
        }
    } else if (alpha.num_rows == x.num_rows) {
        for (int i = 0; i < x.num_rows; ++i) {
            for (int j = 0; j < x.num_rhs; ++j) {
                x.values[i * x.stride + j] *=
                    alpha.values[i * alpha.stride + j];
            }
        }
    } else {
        for (int i = 0; i < x.num_rows; ++i) {
            for (int j = 0; j < x.num_rhs; ++j) {
                x.values[i * x.stride + j] *= alpha.values[j];
            }
        }
    }
}


template <typename ValueType>
inline void add_scaled_kernel(
    const batch::multi_vector::batch_item<const ValueType>& alpha,
    const batch::multi_vector::batch_item<const ValueType>& x,
    const batch::multi_vector::batch_item<ValueType>& y)
{
    if (alpha.num_rhs == 1) {
        for (int i = 0; i < x.num_rows; ++i) {
            for (int j = 0; j < x.num_rhs; ++j) {
                y.values[i * y.stride + j] +=
                    alpha.values[0] * x.values[i * x.stride + j];
            }
        }
    } else {
        for (int i = 0; i < x.num_rows; ++i) {
            for (int j = 0; j < x.num_rhs; ++j) {
                y.values[i * y.stride + j] +=
                    alpha.values[j] * x.values[i * x.stride + j];
            }
        }
    }
}


template <typename ValueType>
inline void compute_dot_product_kernel(
    const batch::multi_vector::batch_item<const ValueType>& x,
    const batch::multi_vector::batch_item<const ValueType>& y,
    const batch::multi_vector::batch_item<ValueType>& result)
{
    for (int c = 0; c < result.num_rhs; c++) {
        result.values[c] = zero<ValueType>();
    }

    for (int r = 0; r < x.num_rows; r++) {
        for (int c = 0; c < x.num_rhs; c++) {
            result.values[c] +=
                x.values[r * x.stride + c] * y.values[r * y.stride + c];
        }
    }
}


template <typename ValueType>
inline void compute_conj_dot_product_kernel(
    const batch::multi_vector::batch_item<const ValueType>& x,
    const batch::multi_vector::batch_item<const ValueType>& y,
    const batch::multi_vector::batch_item<ValueType>& result)
{
    for (int c = 0; c < result.num_rhs; c++) {
        result.values[c] = zero<ValueType>();
    }

    for (int r = 0; r < x.num_rows; r++) {
        for (int c = 0; c < x.num_rhs; c++) {
            result.values[c] +=
                conj(x.values[r * x.stride + c]) * y.values[r * y.stride + c];
        }
    }
}


template <typename ValueType>
inline void compute_norm2_kernel(
    const batch::multi_vector::batch_item<const ValueType>& x,
    const batch::multi_vector::batch_item<remove_complex<ValueType>>& result)
{
    for (int j = 0; j < x.num_rhs; ++j) {
        result.values[j] = zero<remove_complex<ValueType>>();
    }
    for (int i = 0; i < x.num_rows; ++i) {
        for (int j = 0; j < x.num_rhs; ++j) {
            result.values[j] += squared_norm(x.values[i * x.stride + j]);
        }
    }
    for (int j = 0; j < x.num_rhs; ++j) {
        result.values[j] = sqrt(result.values[j]);
    }
}


/**
 * Copies the values of one multi-vector into another.
 *
 * Note that the output multi-vector should already have memory allocated
 * and stride set.
 */
template <typename ValueType>
inline void copy_kernel(
    const batch::multi_vector::batch_item<const ValueType>& in,
    const batch::multi_vector::batch_item<ValueType>& out)
{
    for (int iz = 0; iz < in.num_rows * in.num_rhs; iz++) {
        const int i = iz / in.num_rhs;
        const int j = iz % in.num_rhs;
        out.values[i * out.stride + j] = in.values[i * in.stride + j];
    }
}


}  // namespace batch_single_kernels
}  // namespace batch_template
}  // namespace reference
}  // namespace kernels
}  // namespace gko


#endif
