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

    void on_polymorphic_object_create_started(
        const gko::Executor*, const gko::PolymorphicObject*) const override
    {
        create_started++;
    }

    void on_polymorphic_object_create_completed(
        const gko::Executor*, const gko::PolymorphicObject*,
        const gko::PolymorphicObject*) const override
    {
        create_completed++;
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


TEST(PolymorphicObject, HoldsExecutor)
{
    auto ref = gko::ReferenceExecutor::create();
    auto obj = DummyObject(ref, 5);

    ASSERT_EQ(obj.get_executor(), ref);
    ASSERT_EQ(obj.x, 5);
}


TEST(PolymorphicObject, LogsObjectDeletion)
{
    auto ref = gko::ReferenceExecutor::create();
    std::shared_ptr<DummyLogger> logger{std::make_shared<DummyLogger>()};
    auto before_count = logger->deleted;

    {
        auto obj = DummyObject(ref, 5);
        obj.add_logger(logger);
    }

    ASSERT_EQ(logger->deleted, before_count + 1);
}


TEST(EnableCreateMethod, CreatesObject)
{
    auto ref = gko::ReferenceExecutor::create();
    auto obj = DummyObject::create(ref, 5);

    ASSERT_EQ(obj->get_executor(), ref);
    ASSERT_EQ(obj->x, 5);
}


class Cloneable : public testing::Test {
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


TEST_F(Cloneable, CreatesDefaultObject)
{
    auto def = obj->create_default();

    ASSERT_NE(def, obj);
    ASSERT_EQ(def->get_executor(), ref);
    ASSERT_EQ(def->x, 0);
}


TEST_F(Cloneable, CreatesDefaultObjectOnAnotherExecutor)
{
    auto def = obj->create_default(omp);

    ASSERT_NE(def, obj);
    ASSERT_EQ(def->get_executor(), omp);
    ASSERT_EQ(def->x, 0);
}


TEST_F(Cloneable, ClonesObject)
{
    auto clone = obj->clone();

    ASSERT_NE(clone.get(), obj.get());
    ASSERT_EQ(clone->get_executor(), obj->get_executor());
    ASSERT_EQ(clone->x, obj->x);
}


TEST_F(Cloneable, ClonesObjectOnAnotherExecutor)
{
    auto clone = obj->clone(omp);

    ASSERT_NE(clone.get(), obj.get());
    ASSERT_EQ(clone->get_executor(), omp);
    ASSERT_EQ(clone->x, obj->x);
}


TEST_F(Cloneable, CopiesFrom)
{
    auto copy = obj->create_default();

    copy->copy_from(obj.get());

    ASSERT_EQ(copy->x, obj->x);
}


TEST_F(Cloneable, CopiesFromLogsEvents)
{
    auto copy_started = logger->copy_started;
    auto copy_completed = logger->copy_completed;
    auto copy = obj->create_default();
    copy->add_logger(logger);

    copy->copy_from(obj.get());

    ASSERT_EQ(logger->copy_started, copy_started + 1);
    ASSERT_EQ(logger->copy_completed, copy_completed + 1);
}


TEST_F(Cloneable, MovesFrom)
{
    auto move = obj->create_default();
    auto expected_x = obj->x;

    move->move_from(obj.get());

    ASSERT_EQ(move->x, expected_x);
    ASSERT_EQ(obj->x, 0);
}


TEST_F(Cloneable, MovesFromLogsEvents)
{
    auto move_started = logger->move_started;
    auto move_completed = logger->move_completed;
    auto move = obj->create_default();
    move->add_logger(logger);

    move->move_from(obj.get());

    ASSERT_EQ(logger->move_started, move_started + 1);
    ASSERT_EQ(logger->move_completed, move_completed + 1);
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
