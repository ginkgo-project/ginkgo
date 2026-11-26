// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/base/temporary_conversion.hpp"

#include <memory>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/multivector.hpp>

#include "core/test/utils.hpp"


class LinOpA : public gko::LinOp {
public:
    LinOpA(const std::shared_ptr<const gko::Executor>& exec, double value = 0.0)
        : LinOp(exec, {}), value(value)
    {}

    static std::unique_ptr<LinOpA> create(
        const std::shared_ptr<const gko::Executor>& exec, double value = 0.0)
    {
        return std::make_unique<LinOpA>(exec, value);
    }

    virtual double get_value() const { return value; }

    // unused but required to compile
    template <typename Other>
    void convert_to(Other* ptr)
    {}

protected:
    void apply_impl(const LinOp* b, LinOp* x) const override {}
    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override
    {}

private:
    double value;
};

class LinOpB : public LinOpA {
public:
    LinOpB(const std::shared_ptr<const gko::Executor>& exec, double value = 0.0)
        : LinOpA(exec, 0.0), value(value)
    {}

    static std::unique_ptr<LinOpB> create(
        const std::shared_ptr<const gko::Executor>& exec, double value = 0.0)
    {
        return std::make_unique<LinOpB>(exec, value);
    }


    double get_value() const override { return value; }

private:
    double value;
};


class alloc : public gko::log::Logger {
public:
    mutable int count = 0;

protected:
    void on_allocation_started(const gko::Executor* exec,
                               const gko::size_type& num_bytes) const override
    {
        count++;
    }
};


class TemporaryConversion : public ::testing::Test {
protected:
    using value_type = double;
    using Vec = gko::matrix::MultiVector<float>;
    using MixedVec = gko::matrix::MultiVector<double>;

    void SetUp() override
    {
        log->count = 0;
        exec->add_logger(log);
    }

    std::shared_ptr<alloc> log = std::make_shared<alloc>();
    std::shared_ptr<gko::ReferenceExecutor> exec =
        gko::ReferenceExecutor::create();
    std::unique_ptr<Vec> vec = gko::initialize<Vec>({2, 3}, exec);
    std::unique_ptr<MixedVec> mixed_vec =
        gko::initialize<MixedVec>({2, 3}, exec);
    std::unique_ptr<LinOpA> lA = std::make_unique<LinOpA>(exec, 3);
    std::unique_ptr<LinOpB> lB = std::make_unique<LinOpB>(exec, 4);
};


TEST_F(TemporaryConversion, CreateFromNullptrThrows)
{
    EXPECT_THROW(
        gko::temporary_conversion<Vec>::create(static_cast<Vec*>(nullptr)),
        gko::NotSupported);
}


TEST_F(TemporaryConversion, ConstCreateFromSameType)
{
    auto tmp = gko::temporary_conversion<const Vec>::create(vec.get());

    EXPECT_EQ(typeid(tmp.get()), typeid(const Vec*));
    EXPECT_EQ(tmp.get(), vec.get());
    EXPECT_EQ(log->count, 0);
    GKO_ASSERT_EQUAL_DIMENSIONS(tmp, vec);
}


TEST_F(TemporaryConversion, ConstCreateFromDerivedType)
{
    auto tmp = gko::temporary_conversion<const LinOpA>::create(lB.get());

    EXPECT_EQ(typeid(tmp.get()), typeid(const LinOpA*));
    EXPECT_EQ(tmp->get_value(), lB->get_value());
    EXPECT_EQ(log->count, 0);
}


TEST_F(TemporaryConversion, ConstCreateFromBaseType)
{
    auto tmp = gko::temporary_conversion<const LinOpB>::create(
        gko::as<LinOpA>(lB.get()));

    EXPECT_EQ(typeid(tmp.get()), typeid(const LinOpB*));
    EXPECT_EQ(tmp->get_value(), lB->get_value());
    EXPECT_EQ(log->count, 0);
}


TEST_F(TemporaryConversion, ConstCreateFromConvertibleType)
{
    using NewVec = gko::matrix::MultiVector<double>;
    auto tmp = gko::temporary_conversion<const NewVec>::create(vec.get());

    EXPECT_EQ(typeid(tmp.get()), typeid(const NewVec*));
    EXPECT_NE(reinterpret_cast<std::uintptr_t>(tmp->get_const_values()),
              reinterpret_cast<std::uintptr_t>(vec->get_values()));
    EXPECT_GT(log->count, 0);
    GKO_ASSERT_MTX_NEAR(tmp.get(), vec, 0);
}


TEST_F(TemporaryConversion, ConstCreateFromSameTypeUniquePtr)
{
    auto orig = vec.get();

    auto tmp = gko::temporary_conversion<const Vec>::create(std::move(vec));

    EXPECT_EQ(typeid(tmp.get()), typeid(const Vec*));
    EXPECT_EQ(tmp.get(), orig);
    EXPECT_EQ(log->count, 0);
    EXPECT_EQ(vec, nullptr);
}


TEST_F(TemporaryConversion, ConstCreateFromDerivedTypeUniquePtr)
{
    auto orig = lB.get();

    auto tmp = gko::temporary_conversion<const LinOpA>::create(std::move(lB));

    EXPECT_EQ(typeid(tmp.get()), typeid(const LinOpA*));
    EXPECT_EQ(tmp.get(), orig);
    EXPECT_EQ(tmp->get_value(), 4);
    EXPECT_EQ(log->count, 0);
    EXPECT_EQ(lB, nullptr);
}


TEST_F(TemporaryConversion, ConstCreateFromBaseTypeUniquePtr)
{
    auto orig = lB.get();

    auto tmp = gko::temporary_conversion<const LinOpB>::create(
        std::unique_ptr<LinOpA>(std::move(lB)));

    EXPECT_EQ(typeid(tmp.get()), typeid(const LinOpB*));
    EXPECT_EQ(tmp.get(), orig);
    EXPECT_EQ(tmp->get_value(), 4);
    EXPECT_EQ(log->count, 0);
    EXPECT_EQ(lB, nullptr);
}


TEST_F(TemporaryConversion, ConstCreateFromConvertibleTypeUniquePtr)
{
    using NewVec = gko::matrix::MultiVector<double>;
    auto orig_values = vec->get_values();
    auto ref = gko::clone(vec);
    log->count = 0;

    auto tmp = gko::temporary_conversion<const NewVec>::create(std::move(vec));

    EXPECT_EQ(typeid(tmp.get()), typeid(const NewVec*));
    EXPECT_NE(reinterpret_cast<std::uintptr_t>(tmp->get_const_values()),
              reinterpret_cast<std::uintptr_t>(orig_values));
    EXPECT_GT(log->count, 0);
    EXPECT_EQ(vec, nullptr);
    GKO_ASSERT_MTX_NEAR(tmp.get(), ref, 0);
}


TEST_F(TemporaryConversion, CreateFromSameType)
{
    auto tmp = gko::temporary_conversion<Vec>::create(vec.get());

    EXPECT_EQ(typeid(tmp.get()), typeid(Vec*));
    EXPECT_EQ(tmp->get_values(), vec->get_values());
    EXPECT_EQ(log->count, 0);
    GKO_ASSERT_EQUAL_DIMENSIONS(tmp, vec);
}


TEST_F(TemporaryConversion, CreateFromDerivedType)
{
    auto tmp = gko::temporary_conversion<LinOpA>::create(lB.get());

    EXPECT_EQ(typeid(tmp.get()), typeid(LinOpA*));
    EXPECT_EQ(tmp->get_value(), lB->get_value());
    EXPECT_EQ(log->count, 0);
}


TEST_F(TemporaryConversion, CreateFromBaseType)
{
    auto tmp =
        gko::temporary_conversion<LinOpB>::create(gko::as<LinOpA>(lB.get()));

    EXPECT_EQ(typeid(tmp.get()), typeid(LinOpB*));
    EXPECT_EQ(tmp->get_value(), lB->get_value());
    EXPECT_EQ(log->count, 0);
}


TEST_F(TemporaryConversion, CreateFromConvertibleType)
{
    using NewVec = gko::matrix::MultiVector<double>;
    auto tmp = gko::temporary_conversion<NewVec>::create(vec.get());

    EXPECT_EQ(typeid(tmp.get()), typeid(NewVec*));
    EXPECT_NE(reinterpret_cast<std::uintptr_t>(tmp->get_const_values()),
              reinterpret_cast<std::uintptr_t>(vec->get_values()));
    EXPECT_GT(log->count, 0);
    GKO_ASSERT_MTX_NEAR(tmp.get(), vec, 0);
}


TEST_F(TemporaryConversion, CreateFromNullptrUniquePtrThrows)
{
    EXPECT_THROW(gko::temporary_conversion<Vec>::create(std::unique_ptr<Vec>{}),
                 gko::NotSupported);
}


TEST_F(TemporaryConversion, CreateFromSameTypeUniquePtr)
{
    auto orig = vec.get();

    auto tmp = gko::temporary_conversion<Vec>::create(std::move(vec));

    EXPECT_EQ(typeid(tmp.get()), typeid(Vec*));
    EXPECT_EQ(tmp.get(), orig);
    EXPECT_EQ(log->count, 0);
    EXPECT_EQ(vec, nullptr);
}


TEST_F(TemporaryConversion, CreateFromDerivedTypeUniquePtr)
{
    auto orig = lB.get();

    auto tmp = gko::temporary_conversion<LinOpA>::create(std::move(lB));

    EXPECT_EQ(typeid(tmp.get()), typeid(LinOpA*));
    EXPECT_EQ(tmp.get(), orig);
    EXPECT_EQ(tmp->get_value(), 4);
    EXPECT_EQ(log->count, 0);
    EXPECT_EQ(lB, nullptr);
}


TEST_F(TemporaryConversion, CreateFromBaseTypeUniquePtr)
{
    auto orig = lB.get();

    auto tmp = gko::temporary_conversion<LinOpB>::create(
        std::unique_ptr<LinOpA>(std::move(lB)));

    EXPECT_EQ(typeid(tmp.get()), typeid(LinOpB*));
    EXPECT_EQ(tmp.get(), orig);
    EXPECT_EQ(tmp->get_value(), 4);
    EXPECT_EQ(log->count, 0);
    EXPECT_EQ(lB, nullptr);
}


TEST_F(TemporaryConversion, CreateFromConvertibleTypeUniquePtr)
{
    using NewVec = gko::matrix::MultiVector<double>;
    auto orig_values = vec->get_values();
    auto ref = gko::clone(vec);
    log->count = 0;

    auto tmp = gko::temporary_conversion<NewVec>::create(std::move(vec));

    EXPECT_EQ(typeid(tmp.get()), typeid(NewVec*));
    EXPECT_NE(reinterpret_cast<std::uintptr_t>(tmp->get_const_values()),
              reinterpret_cast<std::uintptr_t>(orig_values));
    EXPECT_GT(log->count, 0);
    EXPECT_EQ(vec, nullptr);
    GKO_ASSERT_MTX_NEAR(tmp.get(), ref, 0);
}


TEST_F(TemporaryConversion, ConstCreateFromBaseTypeNonConvertibleThrows)
{
    EXPECT_THROW(gko::temporary_conversion<const LinOpA>::create(
                     static_cast<gko::PolymorphicObject*>(vec.get())),
                 gko::NotSupported);
}


TEST_F(TemporaryConversion, CreateFromBaseTypeNonConvertibleThrows)
{
    EXPECT_THROW(gko::temporary_conversion<LinOpA>::create(
                     static_cast<gko::PolymorphicObject*>(vec.get())),
                 gko::NotSupported);
}


TEST_F(TemporaryConversion,
       ConstCreateFromBaseTypeNonConvertibleUniquePtrThrows)
{
    EXPECT_THROW(gko::temporary_conversion<const LinOpA>::create(
                     std::unique_ptr<gko::PolymorphicObject>(std::move(vec))),
                 gko::NotSupported);
}


TEST_F(TemporaryConversion, CreateFromBaseTypeNonConvertibleUniquePtrThrows)
{
    EXPECT_THROW(gko::temporary_conversion<LinOpA>::create(
                     std::unique_ptr<gko::PolymorphicObject>(std::move(vec))),
                 gko::NotSupported);
}


TEST_F(TemporaryConversion, CreateNonConstFromSameTypeCopiesBack)
{
    {
        auto tmp = gko::temporary_conversion<MixedVec>::create(vec.get());
        tmp->at(0, 0) = -1.0;
        tmp->at(0, 1) = -2.0;

        EXPECT_EQ(vec->at(0, 0), 2.0);
        EXPECT_EQ(vec->at(0, 1), 3.0);
    }

    EXPECT_EQ(vec->at(0, 0), -1.0);
    EXPECT_EQ(vec->at(0, 1), -2.0);
}


TEST_F(TemporaryConversion, CreateNonConstFromConvertibleTypeCopiesBackOnDelete)
{
    {
        auto tmp = gko::temporary_conversion<MixedVec>::create(vec.get());
        tmp->at(0, 0) = -1.0;
        tmp->at(1, 0) = -2.0;

        EXPECT_EQ(vec->at(0, 0), 2.0);
        EXPECT_EQ(vec->at(0, 1), 3.0);
    }

    EXPECT_EQ(vec->at(0, 0), -1.0);
    EXPECT_EQ(vec->at(1, 0), -2.0);
}


TEST_F(TemporaryConversion, CreateNonConstFromSameTypeModifiesOriginal)
{
    {
        auto tmp = gko::temporary_conversion<Vec>::create(vec.get());
        tmp->at(0, 0) = -1.0;
        tmp->at(1, 0) = -2.0;
    }

    EXPECT_EQ(vec->at(0, 0), -1.0);
    EXPECT_EQ(vec->at(1, 0), -2.0);
}


TEST_F(TemporaryConversion, CreateNonConstFromBaseTypeModifiesOriginal)
{
    {
        auto tmp = gko::temporary_conversion<Vec>::create(
            static_cast<gko::PolymorphicObject*>(vec.get()));
        tmp->at(0, 0) = -1.0;
        tmp->at(1, 0) = -2.0;
    }

    EXPECT_EQ(vec->at(0, 0), -1.0);
    EXPECT_EQ(vec->at(1, 0), -2.0);
}


TEST_F(TemporaryConversion, ConstCreateFromConvertibleTypeDoesNotCopyBack)
{
    {
        auto tmp = gko::temporary_conversion<const MixedVec>::create(vec.get());
        // the converted object is a non-const temporary, so this is valid
        const_cast<MixedVec*>(tmp.get())->at(0, 0) = -1.0;
        const_cast<MixedVec*>(tmp.get())->at(1, 0) = -2.0;
    }

    EXPECT_EQ(vec->at(0, 0), 2.0);
    EXPECT_EQ(vec->at(1, 0), 3.0);
}


TEST_F(TemporaryConversion, CreateNonConstChainFromDerivedTypeCopiesBack)
{
    {
        auto tmp = gko::temporary_conversion<MixedVec>::create(vec.get());
        auto tmp_base = gko::temporary_conversion<
            gko::AbstractMultiVector>::create_from_derived(std::move(tmp));
        auto mixed = dynamic_cast<MixedVec*>(tmp_base.get());
        ASSERT_NE(mixed, nullptr);
        mixed->at(0, 0) = -1.0;
        mixed->at(1, 0) = -2.0;
    }

    EXPECT_EQ(vec->at(0, 0), -1.0);
    EXPECT_EQ(vec->at(1, 0), -2.0);
}


TEST_F(TemporaryConversion, CreateNonConstChainFromBaseTypeCopiesBack)
{
    {
        auto tmp = gko::temporary_conversion<MixedVec>::create(vec.get());
        auto tmp_base = gko::temporary_conversion<
            gko::AbstractMultiVector>::create_from_derived(std::move(tmp));
        auto tmp_derived =
            gko::temporary_conversion<MixedVec>::create_from_base(
                std::move(tmp_base));
        tmp_derived->at(0, 0) = -1.0;
        tmp_derived->at(1, 0) = -2.0;
    }

    EXPECT_EQ(vec->at(0, 0), -1.0);
    EXPECT_EQ(vec->at(1, 0), -2.0);
}


TEST_F(TemporaryConversion, ConstCreateChainFromDerivedType)
{
    auto tmp = gko::temporary_conversion<const Vec>::create(vec.get());

    auto tmp_linop =
        gko::temporary_conversion<const gko::LinOp>::create_from_derived(
            std::move(tmp));

    EXPECT_EQ(typeid(tmp_linop.get()), typeid(const gko::LinOp*));
    EXPECT_NE(dynamic_cast<const Vec*>(tmp_linop.get()), nullptr);
    EXPECT_EQ(dynamic_cast<const Vec*>(tmp_linop.get())->get_const_values(),
              vec->get_values());
    EXPECT_EQ(log->count, 0);
}


TEST_F(TemporaryConversion, ConstCreateChainFromBaseType)
{
    auto tmp = gko::temporary_conversion<const LinOpA>::create(lB.get());

    auto tmp_linop = gko::temporary_conversion<const LinOpB>::create_from_base(
        std::move(tmp));

    EXPECT_EQ(typeid(tmp_linop.get()), typeid(const LinOpB*));
    EXPECT_EQ(tmp_linop->get_value(), lB->get_value());
    EXPECT_EQ(log->count, 0);
}


TEST_F(TemporaryConversion, CreateChainFromDerivedType)
{
    auto tmp = gko::temporary_conversion<Vec>::create(vec.get());

    auto tmp_linop = gko::temporary_conversion<gko::LinOp>::create_from_derived(
        std::move(tmp));

    EXPECT_EQ(typeid(tmp_linop.get()), typeid(gko::LinOp*));
    EXPECT_NE(dynamic_cast<Vec*>(tmp_linop.get()), nullptr);
    EXPECT_EQ(dynamic_cast<Vec*>(tmp_linop.get())->get_values(),
              vec->get_values());
    EXPECT_EQ(log->count, 0);
}


TEST_F(TemporaryConversion, CreateChainFromBaseType)
{
    auto tmp = gko::temporary_conversion<LinOpA>::create(lB.get());

    auto tmp_linop =
        gko::temporary_conversion<LinOpB>::create_from_base(std::move(tmp));

    EXPECT_EQ(typeid(tmp_linop.get()), typeid(LinOpB*));
    EXPECT_EQ(tmp_linop->get_value(), lB->get_value());
    EXPECT_EQ(log->count, 0);
}


TEST_F(TemporaryConversion, ConstCreateChainFromWrongBaseTypeThrows)
{
    auto tmp = gko::temporary_conversion<const LinOpA>::create(lA.get());

    EXPECT_THROW(gko::temporary_conversion<const LinOpB>::create_from_base(
                     std::move(tmp)),
                 gko::NotSupported);
}


TEST_F(TemporaryConversion, CreateChainFromWrongBaseTypeThrows)
{
    auto tmp = gko::temporary_conversion<LinOpA>::create(lA.get());

    EXPECT_THROW(
        gko::temporary_conversion<LinOpB>::create_from_base(std::move(tmp)),
        gko::NotSupported);
}


TEST_F(TemporaryConversion, ConstAsWrongTypeThrows)
{
    auto tmp = gko::temporary_conversion<const LinOpA>::create(lA.get());

    EXPECT_THROW(gko::as<LinOpB>(std::move(tmp)), gko::NotSupported);
}


TEST_F(TemporaryConversion, AsWrongTypeThrows)
{
    auto tmp = gko::temporary_conversion<LinOpA>::create(lA.get());

    EXPECT_THROW(gko::as<LinOpB>(std::move(tmp)), gko::NotSupported);
}
