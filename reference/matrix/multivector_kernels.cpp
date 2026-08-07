// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/multivector_kernels.hpp"

#include <algorithm>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>

#include "accessor/block_col_major.hpp"
#include "accessor/range.hpp"
#include "core/base/mixed_precision_types.hpp"
#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The MultiVector matrix format namespace.
 * @ref MultiVector
 * @ingroup dense
 */
namespace multivector {


template <typename InValueType, typename OutValueType>
void copy(std::shared_ptr<const DefaultExecutor> exec,
          matrix::view::dense<const InValueType> input,
          matrix::view::dense<OutValueType> output)
{
    for (size_type row = 0; row < input.size[0]; ++row) {
        for (size_type col = 0; col < input.size[1]; ++col) {
            output(row, col) = static_cast<OutValueType>(input(row, col));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_CONVERSION_OR_COPY(
    GKO_DECLARE_MULTIVECTOR_COPY_KERNEL);


template <typename ValueType>
void fill(std::shared_ptr<const DefaultExecutor> exec,
          matrix::view::dense<ValueType> mat, ValueType value)
{
    for (size_type row = 0; row < mat.size[0]; ++row) {
        for (size_type col = 0; col < mat.size[1]; ++col) {
            mat(row, col) = value;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MULTIVECTOR_FILL_KERNEL);


template <typename ValueType, typename ScalarType>
void scale(std::shared_ptr<const ReferenceExecutor> exec,
           matrix::view::dense<const ScalarType> alpha,
           matrix::view::dense<ValueType> x)
{
    if (alpha.size[1] == 1) {
        for (size_type i = 0; i < x.size[0]; ++i) {
            for (size_type j = 0; j < x.size[1]; ++j) {
                if (is_zero(alpha(0, 0))) {
                    x(i, j) = zero<ValueType>();
                } else {
                    x(i, j) *= alpha(0, 0);
                }
            }
        }
    } else {
        for (size_type i = 0; i < x.size[0]; ++i) {
            for (size_type j = 0; j < x.size[1]; ++j) {
                x(i, j) *= alpha(0, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_SCALAR_TYPE(
    GKO_DECLARE_MULTIVECTOR_SCALE_KERNEL);


template <typename ValueType, typename ScalarType>
void inv_scale(std::shared_ptr<const ReferenceExecutor> exec,
               matrix::view::dense<const ScalarType> alpha,
               matrix::view::dense<ValueType> x)
{
    if (alpha.size[1] == 1) {
        for (size_type i = 0; i < x.size[0]; ++i) {
            for (size_type j = 0; j < x.size[1]; ++j) {
                x(i, j) /= alpha(0, 0);
            }
        }
    } else {
        for (size_type i = 0; i < x.size[0]; ++i) {
            for (size_type j = 0; j < x.size[1]; ++j) {
                x(i, j) /= alpha(0, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_SCALAR_TYPE(
    GKO_DECLARE_MULTIVECTOR_INV_SCALE_KERNEL);


template <typename ValueType, typename ScalarType>
void add_scaled(std::shared_ptr<const ReferenceExecutor> exec,
                matrix::view::dense<const ScalarType> alpha,
                matrix::view::dense<const ValueType> x,
                matrix::view::dense<ValueType> y)
{
    if (alpha.size[1] == 1) {
        if (is_nonzero(alpha(0, 0))) {
            for (size_type i = 0; i < x.size[0]; ++i) {
                for (size_type j = 0; j < x.size[1]; ++j) {
                    y(i, j) += alpha(0, 0) * x(i, j);
                }
            }
        }
    } else {
        for (size_type i = 0; i < x.size[0]; ++i) {
            for (size_type j = 0; j < x.size[1]; ++j) {
                y(i, j) += alpha(0, j) * x(i, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_SCALAR_TYPE(
    GKO_DECLARE_MULTIVECTOR_ADD_SCALED_KERNEL);


template <typename ValueType, typename ScalarType>
void sub_scaled(std::shared_ptr<const ReferenceExecutor> exec,
                matrix::view::dense<const ScalarType> alpha,
                matrix::view::dense<const ValueType> x,
                matrix::view::dense<ValueType> y)
{
    if (alpha.size[1] == 1) {
        if (is_nonzero(alpha(0, 0))) {
            for (size_type i = 0; i < x.size[0]; ++i) {
                for (size_type j = 0; j < x.size[1]; ++j) {
                    y(i, j) -= alpha(0, 0) * x(i, j);
                }
            }
        }
    } else {
        for (size_type i = 0; i < x.size[0]; ++i) {
            for (size_type j = 0; j < x.size[1]; ++j) {
                y(i, j) -= alpha(0, j) * x(i, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_SCALAR_TYPE(
    GKO_DECLARE_MULTIVECTOR_SUB_SCALED_KERNEL);


template <typename ValueType>
void compute_dot(std::shared_ptr<const ReferenceExecutor> exec,
                 matrix::view::dense<const ValueType> x,
                 matrix::view::dense<const ValueType> y,
                 matrix::view::dense<ValueType> result, array<char>&)
{
    for (size_type j = 0; j < x.size[1]; ++j) {
        result(0, j) = zero<ValueType>();
    }
    for (size_type i = 0; i < x.size[0]; ++i) {
        for (size_type j = 0; j < x.size[1]; ++j) {
            result(0, j) += x(i, j) * y(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MULTIVECTOR_COMPUTE_DOT_KERNEL);


template <typename ValueType>
void compute_dot_dispatch(std::shared_ptr<const ReferenceExecutor> exec,
                          matrix::view::dense<const ValueType> x,
                          matrix::view::dense<const ValueType> y,
                          matrix::view::dense<ValueType> result,
                          array<char>& tmp)
{
    compute_dot(exec, x, y, result, tmp);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_MULTIVECTOR_COMPUTE_DOT_DISPATCH_KERNEL);


template <typename ValueType>
void compute_conj_dot(std::shared_ptr<const ReferenceExecutor> exec,
                      matrix::view::dense<const ValueType> x,
                      matrix::view::dense<const ValueType> y,
                      matrix::view::dense<ValueType> result, array<char>&)
{
    for (size_type j = 0; j < x.size[1]; ++j) {
        result(0, j) = zero<ValueType>();
    }
    for (size_type i = 0; i < x.size[0]; ++i) {
        for (size_type j = 0; j < x.size[1]; ++j) {
            result(0, j) += conj(x(i, j)) * y(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_MULTIVECTOR_COMPUTE_CONJ_DOT_KERNEL);


template <typename ValueType>
void compute_conj_dot_dispatch(std::shared_ptr<const DefaultExecutor> exec,
                               matrix::view::dense<const ValueType> x,
                               matrix::view::dense<const ValueType> y,
                               matrix::view::dense<ValueType> result,
                               array<char>& tmp)
{
    compute_conj_dot(exec, x, y, result, tmp);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_MULTIVECTOR_COMPUTE_CONJ_DOT_DISPATCH_KERNEL);


template <typename ValueType>
void compute_norm2(std::shared_ptr<const ReferenceExecutor> exec,
                   matrix::view::dense<const ValueType> x,
                   matrix::view::dense<remove_complex<ValueType>> result,
                   array<char>&)
{
    for (size_type j = 0; j < x.size[1]; ++j) {
        result(0, j) = zero<remove_complex<ValueType>>();
    }
    for (size_type i = 0; i < x.size[0]; ++i) {
        for (size_type j = 0; j < x.size[1]; ++j) {
            result(0, j) += squared_norm(x(i, j));
        }
    }
    for (size_type j = 0; j < x.size[1]; ++j) {
        result(0, j) = sqrt(result(0, j));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_MULTIVECTOR_COMPUTE_NORM2_KERNEL);


template <typename ValueType>
void compute_norm2_dispatch(
    std::shared_ptr<const DefaultExecutor> exec,
    matrix::view::dense<const ValueType> x,
    matrix::view::dense<remove_complex<ValueType>> result, array<char>& tmp)
{
    compute_norm2(exec, x, result, tmp);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_MULTIVECTOR_COMPUTE_NORM2_DISPATCH_KERNEL);


template <typename ValueType>
void compute_norm1(std::shared_ptr<const ReferenceExecutor> exec,
                   matrix::view::dense<const ValueType> x,
                   matrix::view::dense<remove_complex<ValueType>> result,
                   array<char>&)
{
    for (size_type j = 0; j < x.size[1]; ++j) {
        result(0, j) = zero<remove_complex<ValueType>>();
    }
    for (size_type i = 0; i < x.size[0]; ++i) {
        for (size_type j = 0; j < x.size[1]; ++j) {
            result(0, j) += abs(x(i, j));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_MULTIVECTOR_COMPUTE_NORM1_KERNEL);


template <typename ValueType>
void compute_mean(std::shared_ptr<const ReferenceExecutor> exec,
                  matrix::view::dense<const ValueType> x,
                  matrix::view::dense<ValueType> result, array<char>&)
{
    using ValueType_nc = gko::remove_complex<ValueType>;
    for (size_type j = 0; j < x.size[1]; ++j) {
        result(0, j) = zero<ValueType>();
    }

    if (x.size[0] == 0) return;

    for (size_type i = 0; i < x.size[1]; ++i) {
        for (size_type j = 0; j < x.size[0]; ++j) {
            result(0, i) += x(j, i);
        }
        result(0, i) /= static_cast<ValueType_nc>(x.size[0]);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_MULTIVECTOR_COMPUTE_MEAN_KERNEL);


template <typename ValueType, typename IndexType>
void fill_in_matrix_data(std::shared_ptr<const ReferenceExecutor> exec,
                         const device_matrix_data<ValueType, IndexType>& data,
                         matrix::view::dense<ValueType> output)
{
    for (size_type i = 0; i < data.get_num_stored_elements(); i++) {
        output(data.get_const_row_idxs()[i], data.get_const_col_idxs()[i]) =
            data.get_const_values()[i];
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_MULTIVECTOR_FILL_IN_MATRIX_DATA_KERNEL);


template <typename ValueType>
void compute_squared_norm2(
    std::shared_ptr<const ReferenceExecutor> exec,
    matrix::view::dense<const ValueType> x,
    matrix::view::dense<remove_complex<ValueType>> result, array<char>&)
{
    for (size_type j = 0; j < x.size[1]; ++j) {
        result(0, j) = zero<remove_complex<ValueType>>();
    }
    for (size_type i = 0; i < x.size[0]; ++i) {
        for (size_type j = 0; j < x.size[1]; ++j) {
            result(0, j) += squared_norm(x(i, j));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_MULTIVECTOR_COMPUTE_SQUARED_NORM2_KERNEL);


template <typename ValueType>
void compute_sqrt(std::shared_ptr<const ReferenceExecutor> exec,
                  matrix::view::dense<ValueType> data)
{
    for (size_type i = 0; i < data.size[0]; ++i) {
        for (size_type j = 0; j < data.size[1]; ++j) {
            data(i, j) = sqrt(data(i, j));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(
    GKO_DECLARE_MULTIVECTOR_COMPUTE_SQRT_KERNEL);


template <typename ValueType>
void transpose(std::shared_ptr<const ReferenceExecutor> exec,
               matrix::view::dense<const ValueType> orig,
               matrix::view::dense<ValueType> trans)
{
    for (size_type i = 0; i < orig.size[0]; ++i) {
        for (size_type j = 0; j < orig.size[1]; ++j) {
            trans(j, i) = orig(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MULTIVECTOR_TRANSPOSE_KERNEL);


template <typename ValueType>
void conj_transpose(std::shared_ptr<const ReferenceExecutor> exec,
                    matrix::view::dense<const ValueType> orig,
                    matrix::view::dense<ValueType> trans)
{
    for (size_type i = 0; i < orig.size[0]; ++i) {
        for (size_type j = 0; j < orig.size[1]; ++j) {
            trans(j, i) = conj(orig(i, j));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_MULTIVECTOR_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void symm_permute(std::shared_ptr<const ReferenceExecutor> exec,
                  const IndexType* perm,
                  matrix::view::dense<const ValueType> orig,
                  matrix::view::dense<ValueType> permuted)
{
    auto size = orig.size[0];
    for (size_type i = 0; i < size; ++i) {
        for (size_type j = 0; j < size; ++j) {
            permuted(i, j) = orig(perm[i], perm[j]);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_MULTIVECTOR_SYMM_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inv_symm_permute(std::shared_ptr<const ReferenceExecutor> exec,
                      const IndexType* perm,
                      matrix::view::dense<const ValueType> orig,
                      matrix::view::dense<ValueType> permuted)
{
    auto size = orig.size[0];
    for (size_type i = 0; i < size; ++i) {
        for (size_type j = 0; j < size; ++j) {
            permuted(perm[i], perm[j]) = orig(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_MULTIVECTOR_INV_SYMM_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void nonsymm_permute(std::shared_ptr<const ReferenceExecutor> exec,
                     const IndexType* row_perm, const IndexType* col_perm,
                     matrix::view::dense<const ValueType> orig,
                     matrix::view::dense<ValueType> permuted)
{
    for (size_type i = 0; i < orig.size[0]; ++i) {
        for (size_type j = 0; j < orig.size[1]; ++j) {
            permuted(i, j) = orig(row_perm[i], col_perm[j]);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_MULTIVECTOR_NONSYMM_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inv_nonsymm_permute(std::shared_ptr<const ReferenceExecutor> exec,
                         const IndexType* row_perm, const IndexType* col_perm,
                         matrix::view::dense<const ValueType> orig,
                         matrix::view::dense<ValueType> permuted)
{
    for (size_type i = 0; i < orig.size[0]; ++i) {
        for (size_type j = 0; j < orig.size[1]; ++j) {
            permuted(row_perm[i], col_perm[j]) = orig(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_MULTIVECTOR_INV_NONSYMM_PERMUTE_KERNEL);


template <typename ValueType, typename OutputType, typename IndexType>
void row_gather(std::shared_ptr<const ReferenceExecutor> exec,
                const IndexType* rows,
                matrix::view::dense<const ValueType> orig,
                matrix::view::dense<OutputType> row_collection)
{
    for (size_type i = 0; i < row_collection.size[0]; ++i) {
        for (size_type j = 0; j < orig.size[1]; ++j) {
            row_collection(i, j) = orig(rows[i], j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE_2(
    GKO_DECLARE_MULTIVECTOR_ROW_GATHER_KERNEL);


template <typename ValueType, typename OutputType, typename IndexType>
void advanced_row_gather(std::shared_ptr<const ReferenceExecutor> exec,
                         matrix::view::dense<const ValueType> alpha,
                         const IndexType* rows,
                         matrix::view::dense<const ValueType> orig,
                         matrix::view::dense<const ValueType> beta,
                         matrix::view::dense<OutputType> row_collection)
{
    using type = highest_precision<ValueType, OutputType>;
    auto scalar_alpha = alpha(0, 0);
    auto scalar_beta = beta(0, 0);
    for (size_type i = 0; i < row_collection.size[0]; ++i) {
        for (size_type j = 0; j < orig.size[1]; ++j) {
            row_collection(i, j) =
                static_cast<type>(scalar_alpha * orig(rows[i], j)) +
                static_cast<type>(scalar_beta) *
                    static_cast<type>(row_collection(i, j));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE_2(
    GKO_DECLARE_MULTIVECTOR_ADVANCED_ROW_GATHER_KERNEL);


template <typename ValueType, typename IndexType>
void col_permute(std::shared_ptr<const ReferenceExecutor> exec,
                 const IndexType* perm,
                 matrix::view::dense<const ValueType> orig,
                 matrix::view::dense<ValueType> col_permuted)
{
    for (size_type i = 0; i < orig.size[0]; ++i) {
        for (size_type j = 0; j < orig.size[1]; ++j) {
            col_permuted(i, j) = orig(i, perm[j]);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_MULTIVECTOR_COL_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inv_row_permute(std::shared_ptr<const ReferenceExecutor> exec,
                     const IndexType* perm,
                     matrix::view::dense<const ValueType> orig,
                     matrix::view::dense<ValueType> row_permuted)
{
    for (size_type i = 0; i < orig.size[0]; ++i) {
        for (size_type j = 0; j < orig.size[1]; ++j) {
            row_permuted(perm[i], j) = orig(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_MULTIVECTOR_INV_ROW_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inv_col_permute(std::shared_ptr<const ReferenceExecutor> exec,
                     const IndexType* perm,
                     matrix::view::dense<const ValueType> orig,
                     matrix::view::dense<ValueType> col_permuted)
{
    for (size_type i = 0; i < orig.size[0]; ++i) {
        for (size_type j = 0; j < orig.size[1]; ++j) {
            col_permuted(i, perm[j]) = orig(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_MULTIVECTOR_INV_COL_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void symm_scale_permute(std::shared_ptr<const ReferenceExecutor> exec,
                        const ValueType* scale, const IndexType* perm,
                        matrix::view::dense<const ValueType> orig,
                        matrix::view::dense<ValueType> permuted)
{
    for (size_type i = 0; i < orig.size[0]; ++i) {
        for (size_type j = 0; j < orig.size[1]; ++j) {
            const auto row = perm[i];
            const auto col = perm[j];
            permuted(i, j) = scale[row] * scale[col] * orig(row, col);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_MULTIVECTOR_SYMM_SCALE_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inv_symm_scale_permute(std::shared_ptr<const ReferenceExecutor> exec,
                            const ValueType* scale, const IndexType* perm,
                            matrix::view::dense<const ValueType> orig,
                            matrix::view::dense<ValueType> permuted)
{
    for (size_type i = 0; i < orig.size[0]; ++i) {
        for (size_type j = 0; j < orig.size[1]; ++j) {
            const auto row = perm[i];
            const auto col = perm[j];
            permuted(row, col) = orig(i, j) / (scale[row] * scale[col]);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_MULTIVECTOR_INV_SYMM_SCALE_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void nonsymm_scale_permute(std::shared_ptr<const ReferenceExecutor> exec,
                           const ValueType* row_scale,
                           const IndexType* row_perm,
                           const ValueType* col_scale,
                           const IndexType* col_perm,
                           matrix::view::dense<const ValueType> orig,
                           matrix::view::dense<ValueType> permuted)
{
    for (size_type i = 0; i < orig.size[0]; ++i) {
        for (size_type j = 0; j < orig.size[1]; ++j) {
            const auto row = row_perm[i];
            const auto col = col_perm[j];
            permuted(i, j) = row_scale[row] * col_scale[col] * orig(row, col);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_MULTIVECTOR_NONSYMM_SCALE_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inv_nonsymm_scale_permute(std::shared_ptr<const ReferenceExecutor> exec,
                               const ValueType* row_scale,
                               const IndexType* row_perm,
                               const ValueType* col_scale,
                               const IndexType* col_perm,
                               matrix::view::dense<const ValueType> orig,
                               matrix::view::dense<ValueType> permuted)
{
    for (size_type i = 0; i < orig.size[0]; ++i) {
        for (size_type j = 0; j < orig.size[1]; ++j) {
            const auto row = row_perm[i];
            const auto col = col_perm[j];
            permuted(row, col) = orig(i, j) / (row_scale[row] * col_scale[col]);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_MULTIVECTOR_INV_NONSYMM_SCALE_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void row_scale_permute(std::shared_ptr<const ReferenceExecutor> exec,
                       const ValueType* scale, const IndexType* perm,
                       matrix::view::dense<const ValueType> orig,
                       matrix::view::dense<ValueType> permuted)
{
    for (size_type i = 0; i < orig.size[0]; ++i) {
        for (size_type j = 0; j < orig.size[1]; ++j) {
            const auto row = perm[i];
            permuted(i, j) = scale[row] * orig(row, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_MULTIVECTOR_ROW_SCALE_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inv_row_scale_permute(std::shared_ptr<const ReferenceExecutor> exec,
                           const ValueType* scale, const IndexType* perm,
                           matrix::view::dense<const ValueType> orig,
                           matrix::view::dense<ValueType> permuted)
{
    for (size_type i = 0; i < orig.size[0]; ++i) {
        for (size_type j = 0; j < orig.size[1]; ++j) {
            const auto row = perm[i];
            permuted(row, j) = orig(i, j) / scale[row];
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_MULTIVECTOR_INV_ROW_SCALE_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void col_scale_permute(std::shared_ptr<const ReferenceExecutor> exec,
                       const ValueType* scale, const IndexType* perm,
                       matrix::view::dense<const ValueType> orig,
                       matrix::view::dense<ValueType> permuted)
{
    for (size_type i = 0; i < orig.size[0]; ++i) {
        for (size_type j = 0; j < orig.size[1]; ++j) {
            const auto col = perm[j];
            permuted(i, j) = scale[col] * orig(i, col);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_MULTIVECTOR_COL_SCALE_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inv_col_scale_permute(std::shared_ptr<const ReferenceExecutor> exec,
                           const ValueType* scale, const IndexType* perm,
                           matrix::view::dense<const ValueType> orig,
                           matrix::view::dense<ValueType> permuted)
{
    for (size_type i = 0; i < orig.size[0]; ++i) {
        for (size_type j = 0; j < orig.size[1]; ++j) {
            const auto col = perm[j];
            permuted(i, col) = orig(i, j) / scale[col];
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_MULTIVECTOR_INV_COL_SCALE_PERMUTE_KERNEL);


template <typename ValueType>
void inplace_absolute_dense(std::shared_ptr<const ReferenceExecutor> exec,
                            matrix::view::dense<ValueType> source)
{
    auto dim = source.size;
    for (size_type row = 0; row < dim[0]; row++) {
        for (size_type col = 0; col < dim[1]; col++) {
            source(row, col) = abs(source(row, col));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_INPLACE_ABSOLUTE_DENSE_KERNEL);


template <typename ValueType>
void outplace_absolute_dense(
    std::shared_ptr<const ReferenceExecutor> exec,
    matrix::view::dense<const ValueType> source,
    matrix::view::dense<remove_complex<ValueType>> result)
{
    auto dim = source.size;
    for (size_type row = 0; row < dim[0]; row++) {
        for (size_type col = 0; col < dim[1]; col++) {
            result(row, col) = abs(source(row, col));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_OUTPLACE_ABSOLUTE_DENSE_KERNEL);


template <typename ValueType>
void make_complex(std::shared_ptr<const ReferenceExecutor> exec,
                  matrix::view::dense<const ValueType> source,
                  matrix::view::dense<to_complex<ValueType>> result)
{
    auto dim = source.size;
    for (size_type row = 0; row < dim[0]; row++) {
        for (size_type col = 0; col < dim[1]; col++) {
            result(row, col) = to_complex<ValueType>{source(row, col)};
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MAKE_COMPLEX_KERNEL);


template <typename ValueType>
void get_real(std::shared_ptr<const ReferenceExecutor> exec,
              matrix::view::dense<const ValueType> source,
              matrix::view::dense<remove_complex<ValueType>> result)
{
    auto dim = source.size;
    for (size_type row = 0; row < dim[0]; row++) {
        for (size_type col = 0; col < dim[1]; col++) {
            result(row, col) = real(source(row, col));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GET_REAL_KERNEL);


template <typename ValueType>
void get_imag(std::shared_ptr<const ReferenceExecutor> exec,
              matrix::view::dense<const ValueType> source,
              matrix::view::dense<remove_complex<ValueType>> result)
{
    auto dim = source.size;
    for (size_type row = 0; row < dim[0]; row++) {
        for (size_type col = 0; col < dim[1]; col++) {
            result(row, col) = imag(source(row, col));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GET_IMAG_KERNEL);


}  // namespace multivector
}  // namespace reference
}  // namespace kernels
}  // namespace gko
