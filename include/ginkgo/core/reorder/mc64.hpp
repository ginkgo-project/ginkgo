/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#ifndef GKO_PUBLIC_CORE_REORDER_MC64_HPP_
#define GKO_PUBLIC_CORE_REORDER_MC64_HPP_


#include <memory>


#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>
#include <ginkgo/core/reorder/reordering_base.hpp>


namespace gko {
/**
 * @brief The Reorder namespace.
 *
 * @ingroup reorder
 */
namespace reorder {


template <typename ValueType = default_precision, typename IndexType = int32>
class Mc64 : public EnablePolymorphicObject<Mc64<ValueType, IndexType>,
                                            ReorderingBase>,
             public EnablePolymorphicAssignment<Mc64<ValueType, IndexType>> {
    friend class EnablePolymorphicObject<Mc64, ReorderingBase>;

public:
    using matrix_type = matrix::Csr<ValueType, IndexType>;
    using PermutationMatrix = matrix::Permutation<IndexType>;
    using value_type = ValueType;
    using index_type = IndexType;

    enum class reordering_strategy { max_diagonal_product, max_diagonal_sum };

    /**
     * Gets the permutation (permutation matrix, output of the algorithm) of the
     * linear operator.
     *
     * @return the permutation (permutation matrix)
     */
    std::shared_ptr<const PermutationMatrix> get_permutation() const
    {
        return permutation_;
    }

    /**
     * Gets the inverse permutation (permutation matrix, output of the
     * algorithm) of the linear operator.
     *
     * @return the inverse permutation (permutation matrix)
     */
    std::shared_ptr<const PermutationMatrix> get_inverse_permutation() const
    {
        return inv_permutation_;
    }

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * This parameter controls the goal of the permutation.
         */
        reordering_strategy GKO_FACTORY_PARAMETER_SCALAR(
            strategy, reordering_strategy::max_diagonal_product);
    };
    GKO_ENABLE_REORDERING_BASE_FACTORY(Mc64, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    /**
     * Generates the permutation matrix and the inverse permutation
     * matrix.
     */
    void generate(std::shared_ptr<const Executor>& exec,
                  std::shared_ptr<LinOp> system_matrix) const;

    explicit Mc64(std::shared_ptr<const Executor> exec)
        : EnablePolymorphicObject<Mc64, ReorderingBase>(std::move(exec))
    {}

    explicit Mc64(const Factory* factory, const ReorderingBaseArgs& args)
        : EnablePolymorphicObject<Mc64, ReorderingBase>(
              factory->get_executor()),
          parameters_{factory->get_parameters()}
    {
        // Always execute the reordering on the cpu.
        const auto is_gpu_executor =
            this->get_executor() != this->get_executor()->get_master();
        auto cpu_exec = is_gpu_executor ? this->get_executor()->get_master()
                                        : this->get_executor();

        // The system matrix has to be square.
        GKO_ASSERT_IS_SQUARE_MATRIX(args.system_matrix);

        auto const dim = args.system_matrix->get_size();
        permutation_ = PermutationMatrix::create(cpu_exec, dim);
        inv_permutation_ = PermutationMatrix::create(cpu_exec, dim);

        this->generate(cpu_exec, args.system_matrix);

        // Copy back results to gpu if necessary.
        if (is_gpu_executor) {
            const auto gpu_exec = this->get_executor();
            auto gpu_perm = PermutationMatrix::create(gpu_exec, dim);
            gpu_perm->copy_from(permutation_.get());
            permutation_ = gko::share(gpu_perm);
            auto gpu_inv_perm = PermutationMatrix::create(gpu_exec, dim);
            gpu_inv_perm->copy_from(inv_permutation_.get());
            inv_permutation_ = gko::share(gpu_inv_perm);
        }
    }

private:
    std::shared_ptr<PermutationMatrix> permutation_;
    std::shared_ptr<PermutationMatrix> inv_permutation_;
};


}  // namespace reorder
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_REORDER_MC64_HPP_
