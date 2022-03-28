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

#ifndef GKO_PUBLIC_CORE_SOLVER_BATCH_SOLVER_HPP_
#define GKO_PUBLIC_CORE_SOLVER_BATCH_SOLVER_HPP_


#include <ginkgo/core/matrix/batch_dense.hpp>


namespace gko {
namespace solver {
namespace detail {


struct common_batch_params {
    std::shared_ptr<const BatchLinOpFactory> prec_factory;
    std::shared_ptr<const BatchLinOp> generated_prec;
    std::shared_ptr<const BatchLinOp> left_scaling_op;
    std::shared_ptr<const BatchLinOp> right_scaling_op;
};


template <typename ParamsType>
common_batch_params extract_common_batch_params(ParamsType& params)
{
    return {params.preconditioner, params.generated_preconditioner,
            params.left_scaling_op, params.right_scaling_op};
}


}  // namespace detail


struct BatchInfo;


/**
 * @tparam PolymorphicBase  The base class; must be a subclass of BatchLinOp.
 */
template <typename ConcreteSolver, typename PolymorphicBase = BatchLinOp>
class EnableBatchSolver
    : public EnableBatchLinOp<ConcreteSolver, PolymorphicBase> {
public:
    /**
     * Returns the system operator (matrix) of the linear system.
     *
     * @return the system operator (matrix)
     */
    std::shared_ptr<const BatchLinOp> get_system_matrix() const
    {
        return system_matrix_;
    }

protected:
    explicit EnableBatchSolver(std::shared_ptr<const Executor> exec)
        : EnableBatchLinOp<ConcreteSolver, PolymorphicBase>(std::move(exec))
    {}

    explicit EnableBatchSolver(std::shared_ptr<const Executor> exec,
                               std::shared_ptr<const BatchLinOp> system_matrix,
                               detail::common_batch_params common_params);

    void apply_impl(const BatchLinOp* b, BatchLinOp* x) const override;

    void apply_impl(const BatchLinOp* alpha, const BatchLinOp* b,
                    const BatchLinOp* beta, BatchLinOp* x) const override;

    std::shared_ptr<const BatchLinOp> system_matrix_{};
    std::shared_ptr<const BatchLinOp> preconditioner_{};
    std::shared_ptr<const BatchLinOp> left_scaling_{};
    std::shared_ptr<const BatchLinOp> right_scaling_{};

private:
    /**
     * Calls the concrete solver on the given system (not necessarily on
     * system_matrix_).
     *
     * @param mtx  Left-hand side matrix for the linear solve.
     * @param b  Right-hand side vector.
     * @param x  Solution vector and initial guess.
     * @param info  Batch logging information.
     */
    virtual void solver_apply(const BatchLinOp* b, BatchLinOp* x,
                              BatchInfo* const info) const = 0;
};


}  // namespace solver
}  // namespace gko

#endif  // GKO_PUBLIC_CORE_SOLVER_BATCH_SOLVER_HPP_
