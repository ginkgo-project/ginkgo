// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_MULTIGRID_MULTIGRID_LEVEL_HPP_
#define GKO_PUBLIC_CORE_MULTIGRID_MULTIGRID_LEVEL_HPP_


#include <functional>
#include <memory>


#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/utils.hpp>


namespace gko {
/**
 * @brief The multigrid components namespace.
 *
 * @ingroup gko
 */
namespace multigrid {


/**
 * This class represents two levels in a multigrid hierarchy.
 *
 * The MultigridLevel is an interface that allows to get the individual
 * components of multigrid level. Each implementation of a multigrid level
 * should inherit from this interface. Use EnableMultigridLevel<ValueType> to
 * implement this interface with composition by default.
 *
 * @ingroup Multigrid
 */
class MultigridLevel {
public:
    /**
     * Returns the operator on fine level.
     *
     * @return  the operator on fine level.
     */
    virtual std::shared_ptr<const LinOp> get_fine_op() const = 0;

    /**
     * Returns the restrict operator.
     *
     * @return  the restrict operator.
     */
    virtual std::shared_ptr<const LinOp> get_restrict_op() const = 0;

    /**
     * Returns the operator on coarse level.
     *
     * @return  the operator on coarse level.
     */
    virtual std::shared_ptr<const LinOp> get_coarse_op() const = 0;

    /**
     * Returns the prolong operator.
     *
     * @return  the prolong operator.
     */
    virtual std::shared_ptr<const LinOp> get_prolong_op() const = 0;
};


/**
 * The EnableMultigridLevel gives the default implementation of MultigridLevel
 * with composition and provides `set_multigrid_level` function.
 *
 * A class inherit from EnableMultigridLevel should use the
 * this->get_compositions()->apply(...) as its own apply, which represents
 * op(b) = prolong(coarse(restrict(b))).
 *
 * @ingroup Multigrid
 */
template <typename ValueType>
class EnableMultigridLevel : public MultigridLevel,
                             public UseComposition<ValueType> {
public:
    using value_type = ValueType;

    std::shared_ptr<const LinOp> get_fine_op() const override
    {
        return fine_op_;
    }

    std::shared_ptr<const LinOp> get_restrict_op() const override
    {
        return this->get_operator_at(2);
    }

    std::shared_ptr<const LinOp> get_coarse_op() const override
    {
        return this->get_operator_at(1);
    }

    std::shared_ptr<const LinOp> get_prolong_op() const override
    {
        return this->get_operator_at(0);
    }

protected:
    /**
     * Sets the multigrid level information. The stored composition will be
     * prolong_op x coarse_op x restrict_op.
     *
     * @param prolong_op  the prolong operator
     * @param coarse_op  the coarse operator
     * @param restrict_op  the restrict operator
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

    /**
     * Sets the multigrid level fine operator, which is used to update fine
     * operator when the MultigridLevel changes the precision of the operator.
     *
     * @param fine_op  the fine operator
     */
    void set_fine_op(std::shared_ptr<const LinOp> fine_op)
    {
        GKO_ASSERT_EQUAL_DIMENSIONS(fine_op_->get_size(), fine_op->get_size());
        fine_op_ = fine_op;
    }

    explicit EnableMultigridLevel() {}

    /**
     * Creates a EnableMultigridLevel with the given fine operator
     *
     * @param fine_op  The fine operator associated to the multigrid_level
     *
     * @note the multigrid is generated in the generation not the construction,
     *       so the user needs to call `set_multigrid_level` to set the
     *       corresponding information after generation.
     */
    explicit EnableMultigridLevel(std::shared_ptr<const LinOp> fine_op)
        : fine_op_(fine_op)
    {}

private:
    std::shared_ptr<const LinOp> fine_op_;
};


}  // namespace multigrid
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MULTIGRID_MULTIGRID_LEVEL_HPP_
