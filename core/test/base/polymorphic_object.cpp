// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <ginkgo/core/base/polymorphic_object.hpp>


namespace {


struct DummyObject : gko::PolymorphicObject,
                     gko::EnableCreateMethod<DummyObject>,
                     gko::EnableCloneable<DummyObject> {
    explicit DummyObject(std::shared_ptr<const gko::Executor> exec, int v = {})
        : gko::PolymorphicObject(std::move(exec)), x{v}
    {}

    DummyObject(const DummyObject& other) : DummyObject(other.get_executor())
    {
        *this = other;
    }

    DummyObject(DummyObject&& other) : DummyObject(other.get_executor())
    {
        *this = std::move(other);
    }

    DummyObject& operator=(const DummyObject& other)
    {
        if (this != &other) {
            x = other.x;
        }
        return *this;
    }

    DummyObject& operator=(DummyObject&& other) noexcept
    {
        if (this != &other) {
            x = std::exchange(other.x, 0);
        }
        return *this;
    }


    int x;
};


struct DummyLogger : gko::log::Logger {
    DummyLogger()
        : gko::log::Logger(gko::log::Logger::polymorphic_object_events_mask)
    {}

    void on_polymorphic_object_created(
        const gko::Executor*, const gko::PolymorphicObject*) const override
    {
        create_started++;
    }

    void on_polymorphic_object_copy_started(
        const gko::Executor*, const gko::PolymorphicObject*,
        const gko::PolymorphicObject*) const override
    {
        copy_started++;
    }

    void on_polymorphic_object_copy_completed(
        const gko::Executor*, const gko::PolymorphicObject*,
        const gko::PolymorphicObject*) const override
    {
        copy_completed++;
    }

    void on_polymorphic_object_move_started(
        const gko::Executor*, const gko::PolymorphicObject*,
        const gko::PolymorphicObject*) const override
    {
        move_started++;
    }

    void on_polymorphic_object_move_completed(
        const gko::Executor*, const gko::PolymorphicObject*,
        const gko::PolymorphicObject*) const override
    {
        move_completed++;
    }

    void on_polymorphic_object_deleted(
        const gko::Executor*, const gko::PolymorphicObject*) const override
    {
        deleted++;
    }

    mutable int create_started = 0;
    mutable int create_completed = 0;
    mutable int copy_started = 0;
    mutable int copy_completed = 0;
    mutable int move_started = 0;
    mutable int move_completed = 0;
    mutable int deleted = 0;
};


class PolymorphicObject : public testing::Test {
protected:
    std::shared_ptr<gko::ReferenceExecutor> ref{
        gko::ReferenceExecutor::create()};
    std::shared_ptr<gko::OmpExecutor> omp{gko::OmpExecutor::create()};
    std::unique_ptr<DummyObject> obj{new DummyObject(ref, 5)};
    std::shared_ptr<DummyLogger> logger{std::make_shared<DummyLogger>()};

    void SetUp() override
    {
        if (obj) {
            obj->add_logger(logger);
        }
    }

    void TearDown() override
    {
        if (obj) {
            obj->remove_logger(logger);
        }
    }
};


TEST_F(PolymorphicObject, CreatesConcreteClass)
{
    // this test passes as soon as an instance of `DummyObject` can be created
}


TEST_F(PolymorphicObject, CreatesDefaultObject)
{
    auto def = obj->create_default();

    ASSERT_NE(def, obj);
    ASSERT_EQ(def->get_executor(), ref);
    ASSERT_EQ(def->x, 0);
}


TEST_F(PolymorphicObject, CreatesDefaultObjectOnAnotherExecutor)
{
    auto def = obj->create_default(omp);

    ASSERT_NE(def, obj);
    ASSERT_EQ(def->get_executor(), omp);
    ASSERT_EQ(def->x, 0);
}


TEST_F(PolymorphicObject, CreatesDefaultObjectIsLogged)
{
    auto before_logger = *this->logger;

    auto def = obj->create_default();

    ASSERT_EQ(logger->create_started, before_logger.create_started + 1);
    ASSERT_EQ(logger->create_completed, before_logger.create_completed + 1);
}


TEST(EnableCreateMethod, CreatesObject)
{
    auto ref = gko::ReferenceExecutor::create();
    auto obj = DummyObject::create(ref, 5);

    ASSERT_EQ(obj->get_executor(), ref);
    ASSERT_EQ(obj->x, 5);
}


struct ConvertibleToDummyObject
    : gko::PolymorphicObject,
      gko::EnableCreateMethod<ConvertibleToDummyObject>,
      gko::EnableCloneable<ConvertibleToDummyObject>,
      gko::ConvertibleTo<DummyObject> {
    explicit ConvertibleToDummyObject(std::shared_ptr<const gko::Executor> exec,
                                      int v = {})
        : gko::PolymorphicObject(std::move(exec)), x{v}
    {}

    void convert_to(DummyObject* obj) const override { obj->x = x; }

    void move_to(DummyObject* obj) override { obj->x = x; }

    int x;
};


TEST(CopyAndConvertTo, ConvertsToDummyObj)
{
    auto ref = gko::ReferenceExecutor::create();
    auto convertible = ConvertibleToDummyObject::create(ref, 5);

    auto dummy = gko::copy_and_convert_to<DummyObject>(ref, convertible.get());

    ASSERT_EQ(dummy->x, 5);
}


TEST(CopyAndConvertTo, ConvertsConstToDummyObj)
{
    auto ref = gko::ReferenceExecutor::create();
    std::unique_ptr<const ConvertibleToDummyObject> convertible =
        ConvertibleToDummyObject::create(ref, 5);

    auto dummy = gko::copy_and_convert_to<DummyObject>(ref, convertible.get());

    ASSERT_EQ(dummy->x, 5);
}


TEST(CopyAndConvertTo, AvoidsConversion)
{
    auto ref = gko::ReferenceExecutor::create();
    auto convertible = DummyObject::create(ref, 5);

    auto dummy = gko::copy_and_convert_to<DummyObject>(ref, convertible.get());

    ASSERT_EQ(dummy, convertible);
}


}  // namespace
