// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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


TEST(PointerParam, WorksForRawPointers)
{
    auto obj = std::make_unique<Derived>();
    auto ptr = obj.get();

    gko::ptr_param<Base> param(ptr);
    gko::ptr_param<Derived> param2(ptr);

    ASSERT_EQ(param.get(), static_cast<Base*>(ptr));
    ASSERT_EQ(param2.get(), ptr);
}


TEST(PointerParam, WorksForSharedPointers)
{
    auto obj = std::make_shared<Derived>();
    auto ptr = obj.get();

    // no difference whether we use lvalue or rvalue
    gko::ptr_param<Base> param1(obj);
    gko::ptr_param<Base> param2(std::move(obj));
    gko::ptr_param<Derived> param3(obj);
    gko::ptr_param<Derived> param4(std::move(obj));

    ASSERT_EQ(param1.get(), static_cast<Base*>(ptr));
    ASSERT_EQ(param2.get(), static_cast<Base*>(ptr));
    ASSERT_EQ(param3.get(), ptr);
    ASSERT_EQ(param4.get(), ptr);
    // shared_ptr was unmodified
    ASSERT_EQ(obj.get(), ptr);
}


TEST(PointerParam, WorksForUniquePointers)
{
    auto obj = std::make_unique<Derived>();
    auto ptr = obj.get();

    // no difference whether we use lvalue or rvalue
    gko::ptr_param<Base> param1(obj);
    gko::ptr_param<Base> param2(std::move(obj));
    gko::ptr_param<Derived> param3(obj);
    gko::ptr_param<Derived> param4(std::move(obj));

    ASSERT_EQ(param1.get(), static_cast<Base*>(ptr));
    ASSERT_EQ(param2.get(), static_cast<Base*>(ptr));
    ASSERT_EQ(param3.get(), ptr);
    ASSERT_EQ(param4.get(), ptr);
    // shared_ptr was unmodified
    ASSERT_EQ(obj.get(), ptr);
}


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

    auto shared = gko::share(std::move(p));

    ::testing::StaticAssertTypeEq<decltype(shared), std::shared_ptr<Derived>>();
    ASSERT_EQ(plain, shared.get());
}


TEST(Share, SharesTemporarySharedPointer)
{
    auto shared = gko::share(std::make_shared<Derived>());

    ::testing::StaticAssertTypeEq<decltype(shared), std::shared_ptr<Derived>>();
}


TEST(Share, SharesUniquePointer)
{
    std::unique_ptr<Derived> p(new Derived());
    auto plain = p.get();

    auto shared = gko::share(std::move(p));

    ::testing::StaticAssertTypeEq<decltype(shared), std::shared_ptr<Derived>>();
    ASSERT_EQ(plain, shared.get());
}


TEST(Share, SharesTemporaryUniquePointer)
{
    auto shared = gko::share(std::make_unique<Derived>());

    ::testing::StaticAssertTypeEq<decltype(shared), std::shared_ptr<Derived>>();
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


TEST(As, ConvertsPolymorphicType)
{
    Derived d;

    Base* b = &d;

    ASSERT_EQ(gko::as<Derived>(b), &d);
}


TEST(As, FailsToConvertIfNotRelated)
{
    Derived d;
    Base* b = &d;

    try {
        gko::as<NonRelated>(b);
        FAIL();
    } catch (gko::NotSupported& m) {
        std::string msg{m.what()};
        auto expected = gko::name_demangling::get_type_name(typeid(Derived));
        ASSERT_TRUE(
            std::equal(expected.rbegin(), expected.rend(), msg.rbegin()));
    }
}


TEST(As, ConvertsConstantPolymorphicType)
{
    Derived d;
    const Base* b = &d;

    ASSERT_EQ(gko::as<Derived>(b), &d);
}


TEST(As, FailsToConvertConstantIfNotRelated)
{
    Derived d;
    const Base* b = &d;

    try {
        gko::as<NonRelated>(b);
        FAIL();
    } catch (gko::NotSupported& m) {
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
    auto clone = make_temporary_clone(other, obj);

    ASSERT_NE(clone.get()->get_executor(), other);
    ASSERT_EQ(obj->get_executor(), ref);
}


TEST_F(TemporaryClone, OutputDoesNotCopyToSameMemory)
{
    auto other = gko::ReferenceExecutor::create();
    auto clone = make_temporary_output_clone(other, obj);

    ASSERT_NE(clone.get()->get_executor(), other);
    ASSERT_EQ(obj->get_executor(), ref);
}


TEST_F(TemporaryClone, CopiesBackAfterLeavingScope)
{
    obj->data = 4;
    {
        auto clone = make_temporary_clone(omp, obj);
        clone.get()->data = 7;

        ASSERT_EQ(obj->data, 4);
    }
    ASSERT_EQ(obj->get_executor(), ref);
    ASSERT_EQ(obj->data, 7);
}


TEST_F(TemporaryClone, OutputCopiesBackAfterLeavingScope)
{
    obj->data = 4;
    {
        auto clone = make_temporary_output_clone(omp, obj);
        clone.get()->data = 7;

        ASSERT_EQ(obj->data, 4);
    }
    ASSERT_EQ(obj->get_executor(), ref);
    ASSERT_EQ(obj->data, 7);
}


TEST_F(TemporaryClone, DoesntCopyBackConstAfterLeavingScope)
{
    {
        auto clone = make_temporary_clone(
            omp, static_cast<const DummyObject*>(obj.get()));
        obj->data = 7;
    }

    ASSERT_EQ(obj->get_executor(), ref);
    ASSERT_EQ(obj->data, 7);
}


TEST_F(TemporaryClone, AvoidsCopyOnSameExecutor)
{
    auto clone = make_temporary_clone(ref, obj);

    ASSERT_EQ(clone.get(), obj.get());
}


}  // namespace
