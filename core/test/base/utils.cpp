/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <core/base/utils.hpp>


#include <gtest/gtest.h>


#include <core/base/polymorphic_object.hpp>


namespace {


struct Base {
    virtual ~Base() = default;
};


struct Derived : Base {};


struct NonRelated {
    virtual ~NonRelated() = default;
};


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

    ASSERT_THROW(gko::as<NonRelated>(b), gko::NotSupported);
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

    ASSERT_THROW(gko::as<NonRelated>(b), gko::NotSupported);
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


TEST_F(TemporaryClone, CopiesToAnotherExecutor)
{
    auto clone = make_temporary_clone(omp, gko::lend(obj));

    ASSERT_EQ(clone.get()->get_executor(), omp);
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
