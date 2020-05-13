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

#ifndef GKO_CORE_MULTIGRID_AMGX_PGM_HPP_
#define GKO_CORE_MULTIGRID_AMGX_PGM_HPP_


#include <vector>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/multigrid/restrict_prolong.hpp>


namespace gko {
namespace multigrid {


/**
 * Amgx parallel graph match (AmgxPgm) is the aggregate method introduced in the
 * paper M. Naumov et al., "AmgX: A Library for GPU Accelerated Algebraic
 * Multigrid and Preconditioned Iterative Methods"
 *
 * AmgxPgm creates the aggreagate group according to the matrix value not the
 * structure.
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  precision of matrix indexes
 *
 * @ingroup RestrictProlong
 * @ingroup LinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class AmgxPgm : public EnableRestrictProlong<AmgxPgm<ValueType, IndexType>> {
    friend class EnableRestrictProlong<AmgxPgm>;
    friend class EnablePolymorphicObject<AmgxPgm, RestrictProlong>;

public:
    using value_type = ValueType;
    using index_type = IndexType;

    /**
     * Returns the system operator (matrix) of the linear system.
     *
     * @return the system operator (matrix)
     */
    std::shared_ptr<const LinOp> get_system_matrix() const
    {
        return system_matrix_;
    }

    /**
     * Returns the aggregate group.
     *
     * @return the aggregate group.
     */
    IndexType *get_agg() noexcept { return agg_.get_data(); }

    /**
     * @copydoc AmgxPgm::get_agg()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const IndexType *get_const_agg() const noexcept
    {
        return agg_.get_const_data();
    }

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * The maximum number of iteration.
         */
        unsigned GKO_FACTORY_PARAMETER(max_iterations, 15);

        /**
         * The maximum ratio of unassigned number.
         */
        double GKO_FACTORY_PARAMETER(max_unassigned_percentage, 0.05);

        /**
         * Use the deterministic assign_to_exist_agg method or not.
         */
        bool GKO_FACTORY_PARAMETER(deterministic, false);
    };
    GKO_ENABLE_RESTRICT_PROLONG_FACTORY(AmgxPgm, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    void restrict_apply_impl(const LinOp *b, LinOp *x) const override;

    void prolong_applyadd_impl(const LinOp *b, LinOp *x) const override;
    explicit AmgxPgm(std::shared_ptr<const Executor> exec)
        : EnableRestrictProlong<AmgxPgm>(std::move(exec))
    {}
    explicit AmgxPgm(const Factory *factory,
                     std::shared_ptr<const LinOp> system_matrix)
        : EnableRestrictProlong<AmgxPgm>(factory->get_executor()),
          parameters_{factory->get_parameters()},
          system_matrix_{std::move(system_matrix)},
          agg_(factory->get_executor(), system_matrix_->get_size()[0])
    {
        this->generate();
    }

    void generate();

private:
    std::shared_ptr<const LinOp> system_matrix_{};
    Array<IndexType> agg_;
};


}  // namespace multigrid
}  // namespace gko


#endif  // GKO_CORE_MULTIGRID_AMGX_PGM_HPP_
