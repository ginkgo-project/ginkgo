// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_MATRIX_AMP_HPP_
#define GKO_PUBLIC_CORE_MATRIX_AMP_HPP_


#include <limits>

#include <ginkgo/core/base/amp_types.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>


namespace gko {
namespace matrix {


template <typename ValueType>
class Dense;


/**
 * AMP is an adaptive mixed precision matrix class.
 *
 * It takes any sparse matrix and sorts the nonzeros into 'bins' of different
 * precisions, where each bin is a sparse matrix with a specific value type.
 *
 * @tparam ValueType  Highest precision of matrix elements
 * @tparam IndexType  Integer type of matrix indexes
 *
 * @ingroup amp
 * @ingroup mat_formats
 * @ingroup LinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class AMP : public EnableLinOp<AMP<ValueType, IndexType>>,
            public ConvertibleTo<Dense<ValueType>>,
            public DiagonalExtractable<ValueType> {
    friend class EnablePolymorphicObject<AMP, LinOp>;
    friend class Dense<ValueType>;
    friend class AMP<to_complex<ValueType>, IndexType>;

    GKO_ASSERT_SUPPORTED_INDEX_TYPE;
    static_assert(
        std::is_same<remove_complex<ValueType>, double>::value ||
            std::is_same<remove_complex<ValueType>, float>::value,
        "AMP is currently only supported for real types double and float!");

public:
    using EnableLinOp<AMP<ValueType, IndexType>>::convert_to;
    using EnableLinOp<AMP<ValueType, IndexType>>::move_to;
    using ConvertibleTo<Dense<ValueType>>::convert_to;
    using ConvertibleTo<Dense<ValueType>>::move_to;

    using value_type = ValueType;
    using index_type = IndexType;
    using real_type = remove_complex<ValueType>;

    // Maximum number of supported precisions.
    static constexpr int num_precisions =
        gko::amp::num_amp_precisions -
        gko::amp::precision_index<real_type>::index;

    void convert_to(Dense<ValueType>* other) const override;

    void move_to(Dense<ValueType>* other) override;

    std::unique_ptr<Diagonal<ValueType>> extract_diagonal() const override;

    /**
     * Returns a pointer to the i-th bin matrix.
     *
     * The zeroth entry always refers to FP64, the 1st entry to FP32,
     * if supported, entry 2 to BF16, and if supported, entry 3 to FP16
     * (and so on).
     * If particular precision is not supported or if the corredponding bin
     * is not necessary for this matrix, its slot is left un-allocated and
     * should return `nullptr`.
     *
     * @param i  bin index (0 to num_precisions-1)
     * @return pointer to the bin matrix, or nullptr if index out of range
     */
    const LinOp* get_bin_matrix(int i) const
    {
        return i >= 0 && i < num_precisions ? mat_bins_[i].get() : nullptr;
    }

    /// Type of tolerance for adaptive precision
    enum class tolerance_type { normwise, componentwise };

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * The tolerance "epsilon" for adaptive mixed precision generation.
         */
        float GKO_FACTORY_PARAMETER_SCALAR(
            tolerance, std::numeric_limits<real_type>::epsilon() * 100);

        /**
         * Meaning of the tolerance - componentwise or normwise tolerance.
         */
        tolerance_type GKO_FACTORY_PARAMETER_SCALAR(
            strategy, tolerance_type::componentwise);
    };
    GKO_ENABLE_LIN_OP_FACTORY(AMP, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    /**
     * Copy-assigns an AMP matrix. Preserves the executor while copying each
     * precision bin, using its `copy_from` function, as well as the size.
     */
    AMP& operator=(const AMP&);

    /**
     * Move-assigns an AMP matrix. Preserves the executor, moves the data over
     * Leaves the moved-from object in an empty state (0x0 with empty array).
     */
    AMP& operator=(AMP&&);

    /**
     * Copy-constructs an AMP matrix. Inherits executor and dimensions, but
     * copies data without padding.
     */
    AMP(const AMP&);

    /**
     * Move-constructs an AMP matrix. Inherits executor, dimensions and data
     * with padding. The moved-from object is empty (0x0 with empty Array).
     */
    AMP(AMP&&);

protected:
    /// Creates an empty matrix.
    explicit AMP(std::shared_ptr<const Executor>);

    /**
     * Constructs an AMP matrix from a given (high-precision) matrix.
     * Inherits the executor and size; runs an analysis step.
     */
    explicit AMP(const Factory* factory, std::shared_ptr<const LinOp> lin_op)
        : EnableLinOp<AMP>(factory->get_executor(), lin_op->get_size()),
          parameters_{factory->get_parameters()},
          mat_bins_(generate_amp(lin_op.get()))
    {}

    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

    /* Array of bins of the different precisions.
     */
    std::array<std::unique_ptr<const LinOp>, num_precisions> mat_bins_;

    /**
     * Generate binned adaptive precision matrix from given (fixed precision)
     * matrix.
     */
    std::array<std::unique_ptr<const LinOp>, num_precisions> generate_amp(
        const LinOp* matrix) const;
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_AMP_HPP_
