/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#ifndef GKO_CORE_SOLVER_MULTIGRID_HPP_
#define GKO_CORE_SOLVER_MULTIGRID_HPP_


#include <memory>
#include <utility>


#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/multigrid/restrict_prolong.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/criterion.hpp>


namespace gko {
namespace solver {


/**
 * Multigrid
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @ingroup solvers
 * @ingroup LinOp
 */
template <typename ValueType = default_precision>
class Multigrid : public EnableLinOp<Multigrid<ValueType>> {
    friend class EnableLinOp<Multigrid>;
    friend class EnablePolymorphicObject<Multigrid, LinOp>;

public:
    using value_type = ValueType;
    using vector_type = matrix::Dense<ValueType>;

    /**
     * Return true as iterative solvers use the data in x as an initial guess.
     *
     * @return true as iterative solvers use the data in x as an initial guess.
     */
    bool apply_uses_initial_guess() const override { return true; }

    /**
     * Gets the system operator of the linear system.
     *
     * @return the system operator
     */
    std::shared_ptr<const LinOp> get_system_matrix() const
    {
        return system_matrix_;
    }

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Criterion factories.
         */
        std::vector<std::shared_ptr<const stop::CriterionFactory>>
            GKO_FACTORY_PARAMETER(criteria, nullptr);

        std::vector<std::shared_ptr<const multigrid::RestrictProlongFactory>>
            GKO_FACTORY_PARAMETER(rstr_prlg, nullptr);

        std::function<size_type(const size_type, const size_type)>
            GKO_FACTORY_PARAMETER(rstr_prlg_index, nullptr);

        /**
         * Pre-smooth factory.
         */
        std::shared_ptr<const LinOpFactory> GKO_FACTORY_PARAMETER(pre_smoother,
                                                                  nullptr);

        /**
         * Post-smooth factory.
         */
        std::shared_ptr<const LinOpFactory> GKO_FACTORY_PARAMETER(post_smoother,
                                                                  nullptr);

        size_type GKO_FACTORY_PARAMETER(max_levels, 10);

        size_type GKO_FACTORY_PARAMETER(min_coarse_rows, 2);

        /**
         * Coarsest factory.
         */
        std::shared_ptr<const LinOpFactory> GKO_FACTORY_PARAMETER(
            coarsest_solver, nullptr);
    };
    GKO_ENABLE_LIN_OP_FACTORY(Multigrid, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;

    /**
     * Generates the analysis structure from the system matrix and the right
     * hand side needed for the level solver.
     */
    void generate();

    void v_cycle(
        size_type level, std::shared_ptr<const LinOp> matrix,
        const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *x,
        std::vector<std::shared_ptr<matrix::Dense<ValueType>>> &r_list,
        std::vector<std::shared_ptr<matrix::Dense<ValueType>>> &g_list,
        std::vector<std::shared_ptr<matrix::Dense<ValueType>>> &e_list) const;

    void prepare_vcycle(
        const size_type nrhs,
        std::vector<std::shared_ptr<matrix::Dense<ValueType>>> &r,
        std::vector<std::shared_ptr<matrix::Dense<ValueType>>> &g,
        std::vector<std::shared_ptr<matrix::Dense<ValueType>>> &e) const;

    explicit Multigrid(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Multigrid>(std::move(exec))
    {}

    explicit Multigrid(const Factory *factory,
                       std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<Multigrid>(factory->get_executor(),
                                 transpose(system_matrix->get_size())),
          parameters_{factory->get_parameters()},
          system_matrix_{system_matrix},
          one_op_{std::move(initialize<vector_type>({one<ValueType>()},
                                                    factory->get_executor()))},
          neg_one_op_{std::move(initialize<vector_type>(
              {-one<ValueType>()}, factory->get_executor()))}
    {
        GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);

        pre_smoother_is_identity_ =
            validate_identity_factory(parameters_.pre_smoother);
        post_smoother_is_identity_ =
            validate_identity_factory(parameters_.post_smoother);
        stop_criterion_factory_ =
            stop::combine(std::move(parameters_.criteria));
        if (!parameters_.rstr_prlg_index) {
            if (parameters_.rstr_prlg.size() == 1) {
                rstr_prlg_index_ = [](const size_type, const size_type) {
                    return size_type{0};
                };
            } else if (parameters_.rstr_prlg.size() > 1) {
                rstr_prlg_index_ = [](const size_type, const size_type level) {
                    return level;
                };
            }
        }
        this->generate();
    }

    bool validate_identity_factory(std::shared_ptr<const LinOpFactory> factory)
    {
        return !factory ||
               std::dynamic_pointer_cast<
                   const matrix::IdentityFactory<ValueType>>(factory);
    }

private:
    std::shared_ptr<const LinOp> system_matrix_{};
    std::shared_ptr<const stop::CriterionFactory> stop_criterion_factory_{};
    std::vector<std::shared_ptr<multigrid::RestrictProlong>> rstr_prlg_list_{};
    std::vector<std::shared_ptr<LinOp>> pre_smoother_list_{};
    std::vector<std::shared_ptr<LinOp>> post_smoother_list_{};
    std::shared_ptr<LinOp> coarsest_solver_{};
    bool pre_smoother_is_identity_;
    bool post_smoother_is_identity_;
    std::function<size_type(const size_type, const size_type)> rstr_prlg_index_;
    std::shared_ptr<matrix::Dense<ValueType>> one_op_;
    std::shared_ptr<matrix::Dense<ValueType>> neg_one_op_;
};


}  // namespace solver
}  // namespace gko


#endif  // GKO_CORE_SOLVER_MULTIGRID_HPP_
