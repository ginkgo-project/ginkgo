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

#ifndef GKO_PUBLIC_CORE_BASE_TRANSPOSITION_HPP_
#define GKO_PUBLIC_CORE_BASE_TRANSPOSITION_HPP_


#include <vector>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>


namespace gko {
namespace transposition {

/**
 * Behavior defines when to trigger the transpose.
 */
enum behavior {
    now,  // construct intermediately in construction
    lazy  // construct when first apply
    // impilict // use transposed_apply (leave for feature)
};


}  // namespace transposition


/**
 * The Transposition class can be used to wrap of transpose apply such that it
 * can be constructed in lazy fashion.
 *
 * @ingroup LinOp
 */
class Transposition : public EnableLinOp<Transposition>,
                      public EnableCreateMethod<Transposition> {
    friend class EnablePolymorphicObject<Transposition, LinOp>;
    friend class EnableCreateMethod<Transposition>;

public:
    /**
     * Returns the operator
     *
     * @return the operator
     */
    std::shared_ptr<const LinOp> get_operator() const noexcept
    {
        return operator_;
    }

    /**
     * Returns transposition of operator
     *
     * @return transposition of operator
     */
    std::shared_ptr<const LinOp> get_transposition() const noexcept
    {
        this->prepare_transposition();
        return trans_;
    }

    /**
     * Construct the transposition no matter what the behavior is.
     *
     * @note this is const member function such that use this in other const
     *       function.
     */
    void prepare_transposition() const
    {
        if (trans_ == nullptr && operator_ != nullptr) {
            trans_ = std::dynamic_pointer_cast<const Transposable>(operator_)
                         ->transpose();
        }
    }

protected:
    /**
     * Creates an empty operator transposition (0x0 operator).
     *
     * @param exec  Executor associated to the transposition
     */
    explicit Transposition(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Transposition>(exec)
    {}

    /**
     * Creates a transposition of operator with behavior
     *
     * @param oper  the operator
     * @param behavior  the transposition::behavior
     */
    explicit Transposition(
        std::shared_ptr<const LinOp> oper,
        transposition::behavior behavior = transposition::behavior::lazy)
        : EnableLinOp<Transposition>(oper->get_executor(),
                                     transpose(oper->get_size())),
          operator_{oper},
          behavior_{behavior}
    {
        auto ptr = std::dynamic_pointer_cast<const Transposable>(operator_);
        if (ptr == nullptr) {
            GKO_NOT_SUPPORTED(oper);
        }
        if (behavior_ == transposition::behavior::now) {
            trans_ = ptr->transpose();
        }
    }

    void apply_impl(const LinOp* b, LinOp* x) const override
    {
        this->prepare_transposition();
        trans_->apply(b, x);
    }

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override
    {
        this->prepare_transposition();
        trans_->apply(alpha, b, beta, x);
    }

private:
    std::shared_ptr<const LinOp> operator_;
    mutable std::shared_ptr<const LinOp> trans_;
    transposition::behavior behavior_;
};


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_TRANSPOSITION_HPP_
