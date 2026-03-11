// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_TEST_UTILS_FACTORIES_HPP_
#define GKO_CORE_TEST_UTILS_FACTORIES_HPP_

#include <ginkgo/core/base/lin_op.hpp>


/** LinOpFactory that always returns a clone of the same object. */
class ConstantLinOpFactory
    : public gko::EnablePolymorphicObject<ConstantLinOpFactory,
                                          gko::LinOpFactory>,
      public gko::EnablePolymorphicAssignment<ConstantLinOpFactory> {
public:
    friend class EnablePolymorphicObject<ConstantLinOpFactory,
                                         gko::LinOpFactory>;

    static std::unique_ptr<ConstantLinOpFactory> create(
        std::shared_ptr<const gko::LinOp> op)
    {
        return std::unique_ptr<ConstantLinOpFactory>(
            new ConstantLinOpFactory{std::move(op)});
    }

protected:
    explicit ConstantLinOpFactory(std::shared_ptr<const gko::Executor> exec)
        : EnablePolymorphicObject<ConstantLinOpFactory, gko::LinOpFactory>{exec}
    {}

    explicit ConstantLinOpFactory(std::shared_ptr<const gko::LinOp> op)
        : EnablePolymorphicObject<ConstantLinOpFactory,
                                  gko::LinOpFactory>{op->get_executor()},
          op_{std::move(op)}
    {}

    std::unique_ptr<gko::LinOp> generate_impl(
        std::shared_ptr<const gko::LinOp>) const override
    {
        return op_->clone();
    }

    std::shared_ptr<const gko::LinOp> op_;
};


/** LinOpFactory that always returns a clone of the input object. */
class PassthruLinOpFactory
    : public gko::EnablePolymorphicObject<PassthruLinOpFactory,
                                          gko::LinOpFactory>,
      public gko::EnablePolymorphicAssignment<PassthruLinOpFactory> {
public:
    friend class EnablePolymorphicObject<PassthruLinOpFactory,
                                         gko::LinOpFactory>;

    static std::unique_ptr<PassthruLinOpFactory> create(
        std::shared_ptr<const gko::Executor> exec)
    {
        return std::unique_ptr<PassthruLinOpFactory>(
            new PassthruLinOpFactory{exec});
    }

    std::shared_ptr<const gko::LinOp> get_last_op() const { return last_op_; }

protected:
    explicit PassthruLinOpFactory(std::shared_ptr<const gko::Executor> exec)
        : EnablePolymorphicObject<PassthruLinOpFactory, gko::LinOpFactory>{exec}
    {}

    std::unique_ptr<gko::LinOp> generate_impl(
        std::shared_ptr<const gko::LinOp> op) const override
    {
        last_op_ = op;
        return op->clone();
    }

    // last operator we generated on
    mutable std::shared_ptr<const gko::LinOp> last_op_;
};


#endif  // GKO_CORE_TEST_UTILS_FACTORIES_HPP_
