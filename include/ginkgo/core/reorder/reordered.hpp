// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_REORDER_REORDERED_HPP_
#define GKO_PUBLIC_CORE_REORDER_REORDERED_HPP_


#include <ginkgo/core/base/dense_cache.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/permutation.hpp>


namespace gko {
namespace experimental {
namespace reorder {


/**
 * Reordered is a
 *
 * @ingroup reorder
 * @ingroup LinOp
 */
template <typename ValueType, typename IndexType>
class Reordered final : public EnableLinOp<Reordered<ValueType, IndexType>> {
    friend class EnableLinOp<Reordered>;
    friend class EnablePolymorphicObject<Reordered, LinOp>;
    friend class Factory;

public:
    using value_type = ValueType;
    using index_type = IndexType;
    using matrix_type = matrix::Csr<value_type, index_type>;
    using vector_type = matrix::Dense<value_type>;
    using permutation_type = matrix::Permutation<index_type>;

    class Factory;

    struct parameters_type : enable_parameters_type<parameters_type, Factory> {
        /**
         * The factory that is used to generate the inner operator on the
         * reordered system matrix.
         */
        std::shared_ptr<const LinOpFactory> GKO_DEFERRED_FACTORY_PARAMETER(
            inner_operator);

        /**
         * The reordering that is to be applied to the system matrix.
         */
        std::shared_ptr<const LinOpFactory> GKO_DEFERRED_FACTORY_PARAMETER(
            reordering);
    };

    class Factory final : public EnableDefaultLinOpFactory<Factory, Reordered,
                                                           parameters_type> {
        friend class EnablePolymorphicObject<Factory, LinOpFactory>;
        friend class enable_parameters_type<parameters_type, Factory>;
        explicit Factory(std::shared_ptr<const Executor> exec)
            : EnableDefaultLinOpFactory<Factory, Reordered, parameters_type>(
                  std::move(exec))
        {}
        explicit Factory(std::shared_ptr<const Executor> exec,
                         const parameters_type& parameters)
            : EnableDefaultLinOpFactory<Factory, Reordered, parameters_type>(
                  std::move(exec), parameters)
        {}

    public:
        using BaseReuseData = LinOpFactory::ReuseData;

        class ReorderedReuseData : public BaseReuseData {
            friend class Factory;
            friend class Reordered;

            bool is_empty() const;

        protected:
            std::shared_ptr<const permutation_type> permutation_;
            typename matrix_type::permuting_reuse_info permute_reuse_;
            std::unique_ptr<const matrix_type> permuted_;
            std::unique_ptr<LinOpFactory::ReuseData> inner_reuse_;
        };

        std::unique_ptr<BaseReuseData> create_empty_reuse_data() const override;

        /// TODO document
        std::unique_ptr<Reordered> generate_reuse(
            std::shared_ptr<const LinOp> input,
            BaseReuseData& reuse_data) const;

    protected:
        void check_reuse_consistent(const LinOp* input,
                                    BaseReuseData& reuse_data) const override;

        std::unique_ptr<LinOp> generate_reuse_impl(
            std::shared_ptr<const LinOp> input,
            BaseReuseData& reuse_data) const override;
    };

    friend EnableDefaultLinOpFactory<Factory, Reordered, parameters_type>;

    [[nodiscard]] const parameters_type& get_parameters() const;

    [[nodiscard]] static parameters_type build();

    [[nodiscard]] std::shared_ptr<const permutation_type> get_permutation()
        const;

    [[nodiscard]] std::shared_ptr<const LinOp> get_inner_operator() const;

private:
    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

    explicit Reordered(std::shared_ptr<const Executor> exec);

    explicit Reordered(const Factory* factory,
                       std::shared_ptr<const LinOp> system_matrix);

    explicit Reordered(const Factory* factory,
                       std::shared_ptr<const LinOp> system_matrix,
                       typename Factory::ReorderedReuseData& reuse_data);

    std::shared_ptr<const matrix::Permutation<IndexType>> permutation_;
    std::shared_ptr<const LinOp> inner_op_;
    detail::DenseCache<value_type> cache_in_;
    detail::DenseCache<value_type> cache_out_;
    parameters_type parameters_;
};


}  // namespace reorder
}  // namespace experimental
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_REORDER_REORDERED_HPP_
