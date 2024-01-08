// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_REORDER_NESTED_DISSECTION_HPP_
#define GKO_PUBLIC_CORE_REORDER_NESTED_DISSECTION_HPP_


#include <ginkgo/config.hpp>


#if GKO_HAVE_METIS


#include <memory>
#include <unordered_map>


#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/permutation.hpp>


namespace gko {
namespace experimental {
/**
 * @brief The Reorder namespace.
 *
 * @ingroup reorder
 */
namespace reorder {


/**
 * Computes a Nested Dissection (ND) reordering of an input matrix using the
 * METIS library.
 *
 * @tparam ValueType  the type used to store values of the system matrix
 * @tparam IndexType  the type used to store sparsity pattern indices of the
 *                    system matrix
 */
template <typename ValueType, typename IndexType>
class NestedDissection
    : public EnablePolymorphicObject<NestedDissection<ValueType, IndexType>,
                                     LinOpFactory>,
      public EnablePolymorphicAssignment<
          NestedDissection<ValueType, IndexType>> {
public:
    struct parameters_type;
    friend class EnablePolymorphicObject<NestedDissection<ValueType, IndexType>,
                                         LinOpFactory>;
    friend class enable_parameters_type<parameters_type,
                                        NestedDissection<ValueType, IndexType>>;

    using value_type = ValueType;
    using index_type = IndexType;
    using matrix_type = matrix::Csr<value_type, index_type>;
    using permutation_type = matrix::Permutation<index_type>;

    struct parameters_type
        : public enable_parameters_type<
              parameters_type, NestedDissection<ValueType, IndexType>> {
        /**
         * The options to be passed on to METIS, stored as key-value pairs.
         * Any options that are not set here use their default value.
         */
        std::unordered_map<int, int> options;

        /**
         * @copydoc options
         * @return `*this` for chaining
         */
        parameters_type& with_options(std::unordered_map<int, int> options)
        {
            this->options = std::move(options);
            return *this;
        }
    };

    /**
     * Returns the parameters used to construct the factory.
     *
     * @return the parameters used to construct the factory.
     */
    const parameters_type& get_parameters() { return parameters_; }

    /**
     * @copydoc LinOpFactory::generate
     * @note This function overrides the default LinOpFactory::generate to
     *       return a Permutation instead of a generic LinOp, which would need
     *       to be cast to Permutation again to access its indices.
     *       It is only necessary because smart pointers aren't covariant.
     */
    std::unique_ptr<permutation_type> generate(
        std::shared_ptr<const LinOp> system_matrix) const;

    /** Creates a new parameter_type to set up the factory. */
    static parameters_type build() { return {}; }

protected:
    explicit NestedDissection(std::shared_ptr<const Executor> exec,
                              const parameters_type& params = {});

    std::unique_ptr<LinOp> generate_impl(
        std::shared_ptr<const LinOp> system_matrix) const override;

private:
    parameters_type parameters_;
};


}  // namespace reorder
}  // namespace experimental
}  // namespace gko


#endif  // GKO_HAVE_METIS


#endif  // GKO_PUBLIC_CORE_REORDER_NESTED_DISSECTION_HPP_
