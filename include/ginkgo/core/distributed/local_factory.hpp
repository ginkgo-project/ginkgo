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

#ifndef GKO_PUBLIC_CORE_DISTRIBUTED_LOCAL_FACTORY_HPP_
#define GKO_PUBLIC_CORE_DISTRIBUTED_LOCAL_FACTORY_HPP_


#include <ginkgo/config.hpp>


#if GINKGO_BUILD_MPI


#include <ginkgo/core/base/dense_cache.hpp>
#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/distributed/base.hpp>
#include <ginkgo/core/distributed/lin_op.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/vector.hpp>

namespace gko {
namespace experimental {
namespace distributed {


/**
 * The LocalFactory class defines a local factory for (MPI-)distributed system
 *
 * It will generate individual factory on the local matrix, so there's no
 * communication in the local factory or corresponding operation.
 */
class LocalFactory : public EnableLinOp<LocalFactory> {
    friend class EnablePolymorphicObject<LocalFactory, LinOp>;

public:
    using EnableLinOp<LocalFactory>::convert_to;
    using EnableLinOp<LocalFactory>::move_to;

    /**
     * Get read access to the stored local matrix.
     *
     * @return  Shared pointer to the stored local matrix
     */
    std::shared_ptr<const LinOp> get_local_op() const { return local_op_; }

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Already generated preconditioner. If one is provided, the factory
         * `preconditioner` will be ignored.
         */
        std::shared_ptr<const LinOpFactory> GKO_FACTORY_PARAMETER_SCALAR(
            local_factory, nullptr);
    };
    GKO_ENABLE_LIN_OP_FACTORY(LocalFactory, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    explicit LocalFactory(std::shared_ptr<const Executor> exec)
        : EnableLinOp<LocalFactory>(std::move(exec))
    {}
    explicit LocalFactory(const Factory* factory,
                          std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<LocalFactory>(factory->get_executor(),
                                    gko::transpose(system_matrix->get_size())),
          parameters_{factory->get_parameters()}
    {
        if (parameters_.local_factory != nullptr) {
            local_op_ = parameters_.local_factory->generate(
                as<GetLocalShared>(system_matrix)->get_const_local());
        }
    }

    void apply_impl(const LinOp* b, LinOp* x) const override
    {
        local_op_->apply(as<GetLocal>(b)->get_const_local(),
                         as<GetLocal>(x)->get_local());
    }

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override
    {
        local_op_->apply(alpha, as<GetLocal>(b)->get_const_local(), beta,
                         as<GetLocal>(x)->get_local());
    }

private:
    std::shared_ptr<const LinOp> local_op_;
};


}  // namespace distributed
}  // namespace experimental
}  // namespace gko


#endif


#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_LOCAL_FACTORY_HPP_
