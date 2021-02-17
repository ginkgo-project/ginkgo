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

#ifndef GKO_PUBLIC_CORE_MULTIGRID_MULTIGRID_BASE_HPP_
#define GKO_PUBLIC_CORE_MULTIGRID_MULTIGRID_BASE_HPP_


#include <functional>
#include <memory>


#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace multigrid {


/**
 * This interface allows to implement the restrict operation of
 * a multigrid method.
 *
 * Implementers of this interface only need to overload the
 * RestrictOp::restrict_apply_impl() method.
 *
 * @ingroup Multigrid
 */
class RestrictOp {
public:
    /**
     * Applies a restrict operator to a vector (or a sequence of vectors).
     *
     * Performs the operation x = op(b), where op is this restrict operator.
     *
     * @param b  the input vector(s) on which the operator is applied
     * @param x  the output vector(s) where the result is stored
     */
    void restrict_apply(const LinOp *b, LinOp *x) const
    {
        this->validate_parameters(b, x);
        this->restrict_apply_impl(make_temporary_clone(exec_, b).get(),
                                  make_temporary_clone(exec_, x).get());
    }

protected:
    /**
     * Sets the size of the restrict operator.
     *
     * @param value  the new size of the restrict operator
     */
    void set_restrict_size(const dim<2> &value) noexcept { size_ = value; }

    virtual void restrict_apply_impl(const LinOp *b, LinOp *x) const = 0;

    /**
     * Creates a RestrictOp with settings
     *
     * @param exec  Executor associated to the RestrictOp
     * @param size  the size of RestrictOp.
     *
     * @note the multigrid is generated in the generation not the construction,
     *       so calling `set_restrict_size` is needed to set the restrict
     *       size after generation.
     */
    explicit RestrictOp(std::shared_ptr<const Executor> exec,
                        const gko::dim<2> &size = gko::dim<2>{})
        : exec_(exec), size_(size)
    {}

private:
    /**
     * Throws a DimensionMismatch exception if the parameters to
     * `restrict_apply` are of the wrong size.
     *
     * @param b  vector(s) on which the operator is applied
     * @param x  output vector(s)
     */
    void validate_parameters(const LinOp *b, LinOp *x) const
    {
        GKO_ASSERT_CONFORMANT(size_, b);
        GKO_ASSERT_EQUAL_ROWS(size_, x);
        GKO_ASSERT_EQUAL_COLS(b, x);
    }

    std::shared_ptr<const Executor> exec_;
    gko::dim<2> size_;
};


/**
 * This interface allows to implement the prolongation operation
 * of a multigrid method.
 *
 * Implementers of this interface need to overload both the
 * ProlongOp::prolong_apply_impl() and
 * ProlongOp::prolong_applyadd_impl() methods.
 *
 * @ingroup Multigrid
 */
class ProlongOp {
public:
    /**
     * Applies a prolong operator to a vector (or a sequence of vectors).
     *
     * Performs the operation x = op(b), where op is this prolong operator.
     *
     * @param b  the input vector(s) on which the operator is applied
     * @param x  the output vector(s) where the result is stored
     */
    void prolong_apply(const LinOp *b, LinOp *x) const
    {
        this->validate_parameters(b, x);
        this->prolong_apply_impl(make_temporary_clone(exec_, b).get(),
                                 make_temporary_clone(exec_, x).get());
    }

    /**
     * Applies a prolong operator onto a vector (or a sequence of vectors).
     *
     * Performs the operation x += op(b), where op is this prolong operator.
     *
     * @param b  the input vector(s) on which the operator is applied
     * @param x  the output vector(s) where the result is stored
     */
    void prolong_applyadd(const LinOp *b, LinOp *x) const
    {
        this->validate_parameters(b, x);
        this->prolong_applyadd_impl(make_temporary_clone(exec_, b).get(),
                                    make_temporary_clone(exec_, x).get());
    }

protected:
    /**
     * Sets the size of the prolong operator.
     *
     * @param value  the new size of the prolong operator
     */
    void set_prolong_size(const dim<2> &value) noexcept { size_ = value; }

    virtual void prolong_apply_impl(const LinOp *b, LinOp *x) const = 0;

    virtual void prolong_applyadd_impl(const LinOp *b, LinOp *x) const = 0;

    /**
     * Creates a ProlongOp with settings
     *
     * @param exec  Executor associated to the ProlongOp
     * @param native_applyadd  the prolong applyadd is natively implemented in
     *                         one kernel or not. This argument does not has
     *                         effect yet.
     * @param size  the size of ProlongOp.
     *
     * @note the multigrid is generated in the generation not the construction,
     *       so needs to call `set_prolong_size` to set prolong size after
     *       generation.
     */
    explicit ProlongOp(std::shared_ptr<const Executor> exec,
                       bool native_applyadd = false,
                       const gko::dim<2> &size = gko::dim<2>{})
        : exec_(exec), native_applyadd_(native_applyadd), size_(size)
    {}

private:
    /**
     * Throws a DimensionMismatch exception if the parameters to
     * `prolong_applyadd` are of the wrong size.
     *
     * @param b  vector(s) on which the operator is applied
     * @param x  output vector(s)
     */
    void validate_parameters(const LinOp *b, LinOp *x) const
    {
        GKO_ASSERT_CONFORMANT(size_, b);
        GKO_ASSERT_EQUAL_ROWS(size_, x);
        GKO_ASSERT_EQUAL_COLS(b, x);
    }

    std::shared_ptr<const Executor> exec_;
    gko::dim<2> size_;
    bool native_applyadd_;
};


/**
 * This class provides functionality for managing the coarse and fine operator
 * for a multigrid method.
 *
 * @ingroup Multigrid
 */
class CoarseOp {
public:
    /**
     * Gets the fine operator (matrix)
     *
     * @return the fine operator (matrix)
     */
    std::shared_ptr<const LinOp> get_fine_matrix() const { return fine_; }

    /**
     * Gets the coarse operator (matrix)
     *
     * @return the coarse operator (matrix)
     */
    std::shared_ptr<const LinOp> get_coarse_matrix() const { return coarse_; }

protected:
    /**
     * Sets the fine and coarse information.
     *
     * @param fine  the matrix on fine level
     * @param coarse  the matrix on coarse level
     */
    void set_fine_coarse(std::shared_ptr<const LinOp> fine,
                         std::shared_ptr<const LinOp> coarse)
    {
        fine_ = fine;
        coarse_ = coarse;
    }

private:
    std::shared_ptr<const LinOp> coarse_;
    std::shared_ptr<const LinOp> fine_;
};


/**
 * A class implementing this interface can contain restrict, prolong, coarse
 * generation operation. MultigridLevel provides multigrid_level_default_apply
 * to represent op(b) = prolong(coarse(restrict(b))). Those classes implementing
 * all MultigridLevel operation must inherit this class.
 *
 * Implementers of this interface need to overloadthe
 * MultigridLevel::prolong_apply_impl(),
 * MultigridLevel::prolong_applyadd_impl(), and
 * MultigridLevel::restrict_apply_impl() methods.
 *
 * ```c++
 * class MyMultigridLevel : public EnableLinOp<MyMultigridLevel>,
 *                          public MultigridLevel {
 *     GKO_CREATE_FACTORY_PARAMETERS(MyMultigridLevel, Factory) {
 *         // some parameter settings as LinOpFactory
 *     };
 *     GKO_ENABLE_LIN_OP_FACTORY(MyMultigridLevel, parameters, Factory);
 *     GKO_ENABLE_BUILD_METHOD(Factory);
 *
 *     // Need to implement the following override function:
 *     // prolong_apply_impl, prolong_applyadd_impl, restict_apply_impl
 *
 *     // could use the default apply implementation of MultigridLevel
 *     void apply_impl(const LinOp* b, LinOp* x) const override {
 *         this->template multigrid_level_default_apply<ValueType>(b, x);
 *     }
 *
 *     // could use the default apply implementation of MultigridLevel
 *     void apply_impl(const LinOp *alpha, const LinOp *b,
 *                     const LinOp *beta, LinOp *x) const override {
 *         this->template multigrid_level_default_apply<ValueType>(
 *             alpha, b, beta, x);
 *     }
 *
 *     // constructor needed by EnableLinOp
 *     explicit MyMultigridLevel(std::shared_ptr<const Executor> exec) {
 *         : EnableLinOp<MyMultigridLevel>(exec) {}
 *
 *     // constructor needed by the factory
 *     explicit MyMultigridLevel(const Factory *factory,
 *                      std::shared_ptr<const LinOp> matrix)
 *         : EnableLinOp<MyMultigridLevel>(factory->get_executor()),
 *                                         matrix->get_size()),
 *           // store factory's parameters locally
 *           my_parameters_{factory->get_parameters()},
 *     {
 *          // do something with parameter to generate fine/coarse matrices
 *          // set the information such that the checks work properly
 *          this->set_fine_coarse(fine, corase);
 *     }
 * ```
 *
 * @ingroup Multigrid
 */
template <typename ValueType>
class MultigridLevel : public UseComposition<ValueType> {
public:
    using value_type = ValueType;

    std::shared_ptr<const LinOp> get_fine_op() const { return fine_op_; }

    std::shared_ptr<const LinOp> get_restrict_op() const
    {
        return this->get_composition_ithop(2);
    }

    std::shared_ptr<const LinOp> get_coarse_op() const
    {
        return this->get_composition_ithop(1);
    }

    std::shared_ptr<const LinOp> get_prolong_op() const
    {
        return this->get_composition_ithop(0);
    }

protected:
    /**
     * Sets the fine and coarse information.
     *
     * @param fine  the matrix on fine level
     * @param coarse  the matrix on coarse level
     */
    void set_multigrid_level(std::shared_ptr<const LinOp> prolong_op,
                             std::shared_ptr<const LinOp> coarse_op,
                             std::shared_ptr<const LinOp> restrict_op)
    {
        gko::dim<2> mg_size{prolong_op->get_size()[0],
                            restrict_op->get_size()[1]};
        GKO_ASSERT_EQUAL_DIMENSIONS(fine_op_->get_size(), mg_size);
        // check mg_size is the same as fine_size
        this->set_composition(prolong_op, coarse_op, restrict_op);
    }

    explicit MultigridLevel() {}
    /**
     * Creates a MultigridLevel with settings
     *
     * @param exec  Executor associated to the multigrid_level
     * @param native_applyadd  the prolong applyadd is natively implemented in
     *                         one kernel or not. This argument does not has
     *                         effect yet.
     *
     * @note the multigrid is generated in the generation not the construction,
     *       so needs to call `set_multigrid_level` to set corresponding
     *       information after generation.
     */
    explicit MultigridLevel(std::shared_ptr<const LinOp> fine_op)
        : fine_op_(fine_op)
    {}

private:
    std::shared_ptr<const LinOp> fine_op_;
};


}  // namespace multigrid
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MULTIGRID_MULTIGRID_BASE_HPP_
