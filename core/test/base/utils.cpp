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

#include <ginkgo/core/base/utils.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/polymorphic_object.hpp>


namespace {


struct Base {
    virtual ~Base() = default;
};


struct Derived : Base {};


struct NonRelated : Base {};


struct Base2 {
    virtual ~Base2() = default;
};


struct MultipleDerived : Base, Base2 {};


struct ClonableDerived : Base {
    ClonableDerived(std::shared_ptr<const gko::Executor> exec = nullptr)
        : executor(exec)
    {}

    std::unique_ptr<Base> clone()
    {
        return std::unique_ptr<Base>(new ClonableDerived());
    }

    std::unique_ptr<Base> clone(std::shared_ptr<const gko::Executor> exec)
    {
        return std::unique_ptr<Base>(new ClonableDerived{exec});
    }

    std::shared_ptr<const gko::Executor> executor;
};


TEST(Clone, ClonesUniquePointer)
{
    std::unique_ptr<ClonableDerived> p(new ClonableDerived());

    auto clone = gko::clone(p);

    ::testing::StaticAssertTypeEq<decltype(clone),
                                  std::unique_ptr<ClonableDerived>>();
    ASSERT_NE(p.get(), clone.get());
}


TEST(Clone, ClonesSharedPointer)
{
    std::shared_ptr<ClonableDerived> p(new ClonableDerived());

    auto clone = gko::clone(p);

    ::testing::StaticAssertTypeEq<decltype(clone),
                                  std::unique_ptr<ClonableDerived>>();
    ASSERT_NE(p.get(), clone.get());
}


TEST(Clone, ClonesPlainPointer)
{
    std::unique_ptr<ClonableDerived> p(new ClonableDerived());

    auto clone = gko::clone(p.get());

    ::testing::StaticAssertTypeEq<decltype(clone),
                                  std::unique_ptr<ClonableDerived>>();
    ASSERT_NE(p.get(), clone.get());
}


TEST(CloneTo, ClonesUniquePointer)
{
    auto exec = gko::ReferenceExecutor::create();
    std::unique_ptr<ClonableDerived> p(new ClonableDerived());

    auto clone = gko::clone(exec, p);

    ::testing::StaticAssertTypeEq<decltype(clone),
                                  std::unique_ptr<ClonableDerived>>();
    ASSERT_NE(p.get(), clone.get());
    ASSERT_EQ(clone->executor, exec);
}


TEST(CloneTo, ClonesSharedPointer)
{
    auto exec = gko::ReferenceExecutor::create();
    std::shared_ptr<ClonableDerived> p(new ClonableDerived());

    auto clone = gko::clone(exec, p);

    ::testing::StaticAssertTypeEq<decltype(clone),
                                  std::unique_ptr<ClonableDerived>>();
    ASSERT_NE(p.get(), clone.get());
    ASSERT_EQ(clone->executor, exec);
}


TEST(CloneTo, ClonesPlainPointer)
{
    auto exec = gko::ReferenceExecutor::create();
    std::unique_ptr<ClonableDerived> p(new ClonableDerived());

    auto clone = gko::clone(exec, p.get());

    ::testing::StaticAssertTypeEq<decltype(clone),
                                  std::unique_ptr<ClonableDerived>>();
    ASSERT_NE(p.get(), clone.get());
    ASSERT_EQ(clone->executor, exec);
}


TEST(Share, SharesSharedPointer)
{
    std::shared_ptr<Derived> p(new Derived());
    auto plain = p.get();

    auto shared = gko::share(p);

    ::testing::StaticAssertTypeEq<decltype(shared), std::shared_ptr<Derived>>();
    ASSERT_EQ(plain, shared.get());
}


TEST(Share, SharesUniquePointer)
{
    std::unique_ptr<Derived> p(new Derived());
    auto plain = p.get();

    auto shared = gko::share(p);

    ::testing::StaticAssertTypeEq<decltype(shared), std::shared_ptr<Derived>>();
    ASSERT_EQ(plain, shared.get());
}


TEST(Give, GivesSharedPointer)
{
    std::shared_ptr<Derived> p(new Derived());
    auto plain = p.get();

    auto given = gko::give(p);

    ::testing::StaticAssertTypeEq<decltype(given), std::shared_ptr<Derived>>();
    ASSERT_EQ(plain, given.get());
}


TEST(Give, GivesUniquePointer)
{
    std::unique_ptr<Derived> p(new Derived());
    auto plain = p.get();

    auto given = gko::give(p);

    ::testing::StaticAssertTypeEq<decltype(given), std::unique_ptr<Derived>>();
    ASSERT_EQ(plain, given.get());
}


TEST(Lend, LendsUniquePointer)
{
    std::unique_ptr<Derived> p(new Derived());

    auto lent = gko::lend(p);

    ::testing::StaticAssertTypeEq<decltype(lent), Derived *>();
    ASSERT_EQ(p.get(), lent);
}


TEST(Lend, LendsSharedPointer)
{
    std::shared_ptr<Derived> p(new Derived());

    auto lent = gko::lend(p);

    ::testing::StaticAssertTypeEq<decltype(lent), Derived *>();
    ASSERT_EQ(p.get(), lent);
}


TEST(Lend, LendsPlainPointer)
{
    std::unique_ptr<Derived> p(new Derived());

    auto lent = gko::lend(p.get());

    ::testing::StaticAssertTypeEq<decltype(lent), Derived *>();
    ASSERT_EQ(p.get(), lent);
}


TEST(As, ConvertsPolymorphicType)
{
    Derived d;

    Base *b = &d;

    ASSERT_EQ(gko::as<Derived>(b), &d);
}


TEST(As, FailsToConvertIfNotRelated)
{
    Derived d;
    Base *b = &d;

    try {
        gko::as<NonRelated>(b);
        FAIL();
    } catch (gko::NotSupported &m) {
        std::string msg{m.what()};
        auto expected = gko::name_demangling::get_type_name(typeid(Derived));
        ASSERT_TRUE(
            std::equal(expected.rbegin(), expected.rend(), msg.rbegin()));
    }
}


TEST(As, ConvertsConstantPolymorphicType)
{
    Derived d;
    const Base *b = &d;

    ASSERT_EQ(gko::as<Derived>(b), &d);
}


TEST(As, FailsToConvertConstantIfNotRelated)
{
    Derived d;
    const Base *b = &d;

    try {
        gko::as<NonRelated>(b);
        FAIL();
    } catch (gko::NotSupported &m) {
        std::string msg{m.what()};
        auto expected = gko::name_demangling::get_type_name(typeid(Derived));
        ASSERT_TRUE(
            std::equal(expected.rbegin(), expected.rend(), msg.rbegin()));
    }
}


TEST(As, ConvertsPolymorphicTypeUniquePtr)
{
    auto expected = new Derived{};

    ASSERT_EQ(gko::as<Derived>(std::unique_ptr<Base>{expected}).get(),
              expected);
}


TEST(As, FailsToConvertUniquePtrIfNotRelated)
{
    auto expected = new Derived{};

    ASSERT_THROW(gko::as<NonRelated>(std::unique_ptr<Base>{expected}),
                 gko::NotSupported);
}


TEST(As, ConvertsPolymorphicTypeSharedPtr)
{
    auto expected = new Derived{};

    ASSERT_EQ(gko::as<Derived>(std::shared_ptr<Base>{expected}).get(),
              expected);
}


TEST(As, FailsToConvertSharedPtrIfNotRelated)
{
    auto expected = new Derived{};

    ASSERT_THROW(gko::as<NonRelated>(std::shared_ptr<Base>{expected}),
                 gko::NotSupported);
}


TEST(As, ConvertsConstPolymorphicTypeSharedPtr)
{
    auto expected = new Derived{};

    ASSERT_EQ(gko::as<Derived>(std::shared_ptr<const Base>{expected}).get(),
              expected);
}


TEST(As, FailsToConvertConstSharedPtrIfNotRelated)
{
    auto expected = new Derived{};

    ASSERT_THROW(gko::as<NonRelated>(std::shared_ptr<const Base>{expected}),
                 gko::NotSupported);
}


TEST(As, CanCrossCastUniquePtr)
{
    auto obj = std::unique_ptr<MultipleDerived>(new MultipleDerived{});
    auto ptr = obj.get();
    auto base = gko::as<Base>(std::move(obj));

    ASSERT_EQ(gko::as<MultipleDerived>(gko::as<Base2>(std::move(base))).get(),
              ptr);
}


TEST(As, CanCrossCastSharedPtr)
{
    auto obj = std::make_shared<MultipleDerived>();
    auto base = gko::as<Base>(obj);

    ASSERT_EQ(gko::as<MultipleDerived>(gko::as<Base2>(base)), base);
}


TEST(As, CanCrossCastConstSharedPtr)
{
    auto obj = std::make_shared<const MultipleDerived>();
    auto base = gko::as<const Base>(obj);

    ASSERT_EQ(gko::as<const MultipleDerived>(gko::as<const Base2>(base)), base);
}


struct DummyObject : gko::EnablePolymorphicObject<DummyObject>,
                     gko::EnablePolymorphicAssignment<DummyObject>,
                     gko::EnableCreateMethod<DummyObject> {
    DummyObject(std::shared_ptr<const gko::Executor> exec, int value = {})
        : gko::EnablePolymorphicObject<DummyObject>(exec), data{value}
    {}

    int data;
};


class TemporaryClone : public ::testing::Test {
protected:
    TemporaryClone()
        : ref{gko::ReferenceExecutor::create()},
          omp{gko::OmpExecutor::create()},
          obj{DummyObject::create(ref)}
    {}

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::OmpExecutor> omp;
    std::unique_ptr<DummyObject> obj;
};


TEST_F(TemporaryClone, DoesNotCopyToSameMemory)
{
    auto other = gko::ReferenceExecutor::create();
    auto clone = make_temporary_clone(other, gko::lend(obj));

    ASSERT_NE(clone.get()->get_executor(), other);
    ASSERT_EQ(obj->get_executor(), ref);
}


TEST_F(TemporaryClone, CopiesBackAfterLeavingScope)
{
    {
        auto clone = make_temporary_clone(omp, gko::lend(obj));
        clone.get()->data = 7;
    }

    ASSERT_EQ(obj->get_executor(), ref);
    ASSERT_EQ(obj->data, 7);
}


TEST_F(TemporaryClone, AvoidsCopyOnSameExecutor)
{
    auto clone = make_temporary_clone(ref, gko::lend(obj));

    ASSERT_EQ(clone.get(), obj.get());
}


}  // namespace
