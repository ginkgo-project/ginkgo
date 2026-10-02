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
    void apply_impl(const gko::AbstractMultiVector* b,
                    gko::AbstractMultiVector* x) const override
    {}
    void apply_impl(const gko::AbstractMultiVector* alpha,
                    const gko::AbstractMultiVector* b,
                    const gko::AbstractMultiVector* beta,
                    gko::AbstractMultiVector* x) const override
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

    void SetUp() override
    {
        log->count = 0;
        exec->add_logger(log);
    }

    std::shared_ptr<alloc> log = std::make_shared<alloc>();
    std::shared_ptr<gko::ReferenceExecutor> exec =
        gko::ReferenceExecutor::create();
    std::unique_ptr<Vec> vec = gko::initialize<Vec>({2, 3}, exec);
    std::unique_ptr<LinOpA> lA = std::make_unique<LinOpA>(exec, 3);
    std::unique_ptr<LinOpB> lB = std::make_unique<LinOpB>(exec, 4);
};

TEST_F(TemporaryConversion, CreateFromNullptr)
{
    auto tmp =
        gko::temporary_conversion<Vec>::create(static_cast<Vec*>(nullptr));

    EXPECT_EQ(typeid(tmp.get()), typeid(Vec*));
    EXPECT_EQ(tmp.get(), nullptr);
}


TEST_F(TemporaryConversion, ConstCreateFromSameType)
{
    auto tmp = gko::temporary_conversion<const Vec>::create(vec.get());

    EXPECT_EQ(typeid(tmp.get()), typeid(const Vec*));
    EXPECT_EQ(tmp->get_const_values(), vec->get_values());
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


TEST_F(TemporaryConversion, CreateNonConstCopiesBack)
{
    {
        auto tmp = gko::temporary_conversion<Vec>::create(vec.get());
        tmp->at(0, 0) = -1.0;
        tmp->at(0, 1) = -2.0;
    }

    EXPECT_EQ(vec->at(0, 0), -1.0);
    EXPECT_EQ(vec->at(0, 1), -2.0);
}


TEST_F(TemporaryConversion, ConstCreateChainFromDerivedType)
{
    auto tmp = gko::temporary_conversion<const Vec>::create(vec.get());

    auto tmp_linop = gko::temporary_conversion<
        const gko::AbstractMultiVector>::create_from_derived(std::move(tmp));

    EXPECT_EQ(typeid(tmp_linop.get()), typeid(const gko::AbstractMultiVector*));
    ASSERT_NE(dynamic_cast<const Vec*>(tmp_linop.get()), nullptr);
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
    EXPECT_EQ(lB->get_value(), lB->get_value());
    EXPECT_EQ(log->count, 0);
}


TEST_F(TemporaryConversion, CreateChainFromDerivedType)
{
    auto tmp = gko::temporary_conversion<Vec>::create(vec.get());

    auto tmp_linop = gko::temporary_conversion<
        gko::AbstractMultiVector>::create_from_derived(std::move(tmp));

    EXPECT_EQ(typeid(tmp_linop.get()), typeid(gko::AbstractMultiVector*));
    ASSERT_NE(dynamic_cast<Vec*>(tmp_linop.get()), nullptr);
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
    EXPECT_EQ(lB->get_value(), lB->get_value());
    EXPECT_EQ(log->count, 0);
}
