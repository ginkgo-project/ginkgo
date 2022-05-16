/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_PUBLIC_CORE_FACTORIZATION_GLU_HPP_
#define GKO_PUBLIC_CORE_FACTORIZATION_GLU_HPP_


#include <memory>


#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/reorder/mc64.hpp>
#include <ginkgo/core/reorder/reordering_base.hpp>


namespace gko {
/**
 * @brief The Factorization namespace.
 *
 * @ingroup factor
 */
namespace factorization {


/**
 * Represents an incomplete LU factorization -- GLU(0) -- of a sparse matrix.
 *
 * More specifically, it consists of a lower unitriangular factor $L$ and
 * an upper triangular factor $U$ with sparsity pattern
 * $\mathcal S(L + U)$ = $\mathcal S(A)$
 * fulfilling $LU = A$ at every non-zero location of $A$.
 *
 * @tparam ValueType  Type of the values of all matrices used in this class
 * @tparam IndexType  Type of the indices of all matrices used in this class
 *
 * @ingroup factor
 * @ingroup LinOp
 */
template <typename ValueType = gko::default_precision,
          typename IndexType = gko::int32>
class Glu : public Composition<ValueType> {
public:
    using value_type = ValueType;
    using index_type = IndexType;
    using matrix_type = matrix::Csr<ValueType, IndexType>;
    using diag = matrix::Diagonal<ValueType>;
    using index_array = Array<IndexType>;

    std::shared_ptr<const matrix_type> get_l_factor() const
    {
        // Can be `static_cast` since the type is guaranteed in this class
        return std::static_pointer_cast<const matrix_type>(
            this->get_operators()[0]);
    }

    std::shared_ptr<const matrix_type> get_u_factor() const
    {
        // Can be `static_cast` since the type is guaranteed in this class
        return std::static_pointer_cast<const matrix_type>(
            this->get_operators()[1]);
    }

    const int get_status() const { return status_; }

    // Remove the possibility of calling `create`, which was enabled by
    // `Composition`
    template <typename... Args>
    static std::unique_ptr<Composition<ValueType>> create(Args&&... args) =
        delete;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Strategy which will be used by the L matrix. The default value
         * `nullptr` will result in the strategy `classical`.
         */
        std::shared_ptr<typename matrix_type::strategy_type>
            GKO_FACTORY_PARAMETER_SCALAR(l_strategy, nullptr);

        /**
         * Strategy which will be used by the U matrix. The default value
         * `nullptr` will result in the strategy `classical`.
         */
        std::shared_ptr<typename matrix_type::strategy_type>
            GKO_FACTORY_PARAMETER_SCALAR(u_strategy, nullptr);

        /**
         * The `system_matrix`, which will be given to this factory, must be
         * sorted (first by row, then by column) in order for the algorithm
         * to work. If it is known that the matrix will be sorted, this
         * parameter can be set to `true` to skip the sorting (therefore,
         * shortening the runtime).
         * However, if it is unknown or if the matrix is known to be not sorted,
         * it must remain `false`, otherwise, this factorization might be
         * incorrect.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(skip_sorting, false);
    };
    GKO_ENABLE_LIN_OP_FACTORY(Glu, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    class ReusableFactory;
    class ReusableFactoryParameters : public parameters_type {
    public:
        std::unique_ptr<ReusableFactory> on(
            std::shared_ptr<const Executor> exec) const = delete;
        std::unique_ptr<ReusableFactory> on(
            std::shared_ptr<const Executor> exec, const LinOp* A) const
        {
            return std::unique_ptr<ReusableFactory>(
                new ReusableFactory(exec, A, *self()));
        }
        std::unique_ptr<ReusableFactory> on(
            std::shared_ptr<const Executor> exec, const LinOp* A,
            const reorder::ReorderingBase* reordering) const
        {
            return std::unique_ptr<ReusableFactory>(
                new ReusableFactory(exec, A, reordering, *self()));
        }

    protected:
        GKO_ENABLE_SELF(ReusableFactoryParameters);
    };

    class ReusableFactory
        : public EnableDefaultFactory<ReusableFactory, Glu,
                                      ReusableFactoryParameters, LinOpFactory> {
        friend class EnablePolymorphicObject<ReusableFactory, LinOpFactory>;
        friend class ReusableFactoryParameters;
        ReusableFactoryParameters reusable_parameters_;
        explicit ReusableFactory(std::shared_ptr<const Executor> exec)
            : EnableDefaultFactory<ReusableFactory, Glu,
                                   ReusableFactoryParameters, LinOpFactory>(
                  std::move(exec))
        {}
        explicit ReusableFactory(std::shared_ptr<const Executor> exec,
                                 const LinOp* A,
                                 const ReusableFactoryParameters& parameters)
            : EnableDefaultFactory<ReusableFactory, Glu,
                                   ReusableFactoryParameters, LinOpFactory>(
                  std::move(exec), parameters)
        {
            symbolic_factorization(A);
        }
        explicit ReusableFactory(std::shared_ptr<const Executor> exec,
                                 const LinOp* A,
                                 const reorder::ReorderingBase* reordering,
                                 const ReusableFactoryParameters& parameters)
            : EnableDefaultFactory<ReusableFactory, Glu,
                                   ReusableFactoryParameters, LinOpFactory>(
                  std::move(exec), parameters)
        {
            auto mc64 =
                dynamic_cast<const reorder::Mc64<ValueType, IndexType>*>(
                    reordering);
            if (mc64) {
                auto PA = as<matrix_type>(as<matrix_type>(A)->row_permute(
                    mc64->get_permutation().get()));
                PA = as<matrix_type>(PA->inverse_column_permute(
                    mc64->get_inverse_permutation().get()));
                symbolic_factorization(PA.get());
            } else {
                symbolic_factorization(A);
            }
        }

        void symbolic_factorization(const LinOp* system_matrix);

    public:
        std::vector<unsigned> row_ptrs_;
        std::vector<unsigned> col_idxs_;
        std::vector<int> level_idx_;
        std::vector<int> level_ptr_;
        unsigned num_lev_;
        unsigned sym_nnz_;
        std::vector<unsigned> csr_r_ptr;
        std::vector<unsigned> csr_c_idx;
        std::vector<unsigned> csr_diag_ptr;
        std::vector<unsigned> l_col_ptr;
        std::vector<value_type> csr_val;
    };

    friend EnableDefaultFactory<ReusableFactory, Glu, ReusableFactoryParameters,
                                LinOpFactory>;

    static auto build_reusable() -> decltype(ReusableFactory::create())
    {
        return ReusableFactory::create();
    }

protected:
    Glu(const Factory* factory, std::shared_ptr<const gko::LinOp> system_matrix)
        : Composition<ValueType>{factory->get_executor()},
          parameters_{factory->get_parameters()}
    {
        if (parameters_.l_strategy == nullptr) {
            parameters_.l_strategy =
                std::make_shared<typename matrix_type::classical>();
        }
        if (parameters_.u_strategy == nullptr) {
            parameters_.u_strategy =
                std::make_shared<typename matrix_type::classical>();
        }
        auto reusable = Glu::build_reusable().on(factory->get_executor(),
                                                 system_matrix.get());
        generate_l_u(system_matrix, reusable->row_ptrs_, reusable->col_idxs_,
                     reusable->level_idx_, reusable->level_ptr_,
                     reusable->csr_r_ptr, reusable->csr_c_idx,
                     reusable->csr_diag_ptr, reusable->l_col_ptr,
                     reusable->csr_val, reusable->sym_nnz_, reusable->num_lev_,
                     parameters_.skip_sorting)
            ->move_to(this);
    }

    Glu(const ReusableFactory* factory,
        std::shared_ptr<const gko::LinOp> system_matrix)
        : Composition<ValueType>{factory->get_executor()},
          parameters_{factory->get_parameters()}
    {
        if (parameters_.l_strategy == nullptr) {
            parameters_.l_strategy =
                std::make_shared<typename matrix_type::classical>();
        }
        if (parameters_.u_strategy == nullptr) {
            parameters_.u_strategy =
                std::make_shared<typename matrix_type::classical>();
        }
        generate_l_u(system_matrix, factory->row_ptrs_, factory->col_idxs_,
                     factory->level_idx_, factory->level_ptr_,
                     factory->csr_r_ptr, factory->csr_c_idx,
                     factory->csr_diag_ptr, factory->l_col_ptr,
                     factory->csr_val, factory->sym_nnz_, factory->num_lev_,
                     parameters_.skip_sorting)
            ->move_to(this);
    }

    /**
     * Generates the incomplete LU factors, which will be returned as a
     * composition of the lower (first element of the composition) and the
     * upper factor (second element). The dynamic type of L is l_matrix_type,
     * while the dynamic type of U is u_matrix_type.
     *
     * @param system_matrix  the source matrix used to generate the factors.
     *                       @note: system_matrix must be convertible to a Csr
     *                              Matrix, otherwise, an exception is thrown.
     * @param skip_sorting  determines if the sorting of system_matrix can be
     *                      skipped (therefore, marking that it is already
     *                      sorted)
     * @return  A Composition, containing the incomplete LU factors for the
     *          given system_matrix (first element is L, then U)
     */
    std::unique_ptr<Composition<ValueType>> generate_l_u(
        const std::shared_ptr<const LinOp>& system_matrix,
        const std::vector<unsigned>& row_ptrs,
        const std::vector<unsigned>& col_idxs,
        const std::vector<int>& level_idx, const std::vector<int>& level_ptr,
        const std::vector<unsigned>& csr_r_ptr,
        const std::vector<unsigned>& csr_c_idx,
        const std::vector<unsigned>& csr_diag_ptr,
        const std::vector<unsigned>& l_col_ptr,
        const std::vector<ValueType>& csr_val, unsigned sym_nnz,
        unsigned num_lev, bool skip_sorting);

    int status_;
};


}  // namespace factorization
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_FACTORIZATION_GLU_HPP_
