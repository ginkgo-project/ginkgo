// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/polymorphic_object.hpp>


#include <gtest/gtest.h>


namespace {


struct DummyObject : gko::EnablePolymorphicObject<DummyObject>,
                     gko::EnableCreateMethod<DummyObject>,
                     gko::EnablePolymorphicAssignment<DummyObject> {
    explicit DummyObject(std::shared_ptr<const gko::Executor> exec, int v = {})
        : gko::EnablePolymorphicObject<DummyObject>(std::move(exec)), x{v}
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


class EnablePolymorphicObject : public testing::Test {
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


TEST_F(EnablePolymorphicObject, CreatesConcreteClass)
{
    // this test passes as soon as an instance of `DummyObject` can be created
}


TEST_F(EnablePolymorphicObject, CreatesDefaultObject)
{
    auto def = obj->create_default();

    ASSERT_NE(def, obj);
    ASSERT_EQ(def->get_executor(), ref);
    ASSERT_EQ(def->x, 0);
}


TEST_F(EnablePolymorphicObject, CreatesDefaultObjectOnAnotherExecutor)
{
    auto def = obj->create_default(omp);

    ASSERT_NE(def, obj);
    ASSERT_EQ(def->get_executor(), omp);
    ASSERT_EQ(def->x, 0);
}


TEST_F(EnablePolymorphicObject, CreatesDefaultObjectIsLogged)
{
    auto before_logger = *this->logger;

    auto def = obj->create_default();

    ASSERT_EQ(logger->create_started, before_logger.create_started + 1);
    ASSERT_EQ(logger->create_completed, before_logger.create_completed + 1);
}


TEST_F(EnablePolymorphicObject, ClonesObject)
{
    auto clone = obj->clone();

    ASSERT_NE(clone, obj);
    ASSERT_EQ(clone->get_executor(), ref);
    ASSERT_EQ(clone->x, 5);
}


TEST_F(EnablePolymorphicObject, ClonesObjectToAnotherExecutor)
{
    auto clone = obj->clone(omp);

    ASSERT_NE(clone, obj);
    ASSERT_EQ(clone->get_executor(), omp);
    ASSERT_EQ(clone->x, 5);
}


TEST_F(EnablePolymorphicObject, CopiesObject)
{
    auto copy = DummyObject::create(omp, 7);

    copy->copy_from(obj);

    ASSERT_NE(copy, obj);
    ASSERT_EQ(copy->get_executor(), omp);
    ASSERT_EQ(copy->x, 5);
    ASSERT_EQ(obj->get_executor(), ref);
    ASSERT_EQ(obj->x, 5);
}


TEST_F(EnablePolymorphicObject, CopiesObjectIsLogged)
{
    auto before_logger = *logger;
    auto copy = DummyObject::create(omp, 7);
    copy->add_logger(logger);

    copy->copy_from(obj);

    ASSERT_EQ(logger->copy_started, before_logger.copy_started + 1);
    ASSERT_EQ(logger->copy_completed, before_logger.copy_completed + 1);
}


TEST_F(EnablePolymorphicObject, MovesObject)
{
    auto copy = DummyObject::create(ref, 7);

    copy->move_from(obj);

    ASSERT_NE(copy, obj);
    ASSERT_EQ(copy->get_executor(), ref);
    ASSERT_EQ(copy->x, 5);
    ASSERT_EQ(obj->get_executor(), ref);
    ASSERT_EQ(obj->x, 0);
}


TEST_F(EnablePolymorphicObject, MovesObjectIsLogged)
{
    auto before_logger = *this->logger;
    auto copy = DummyObject::create(ref, 7);
    copy->add_logger(logger);

    copy->move_from(obj);

    ASSERT_EQ(logger->move_started, before_logger.move_started + 1);
    ASSERT_EQ(logger->move_completed, before_logger.move_completed + 1);
}


TEST_F(EnablePolymorphicObject, ClearsObject)
{
    obj->clear();

    ASSERT_EQ(obj->get_executor(), ref);
    ASSERT_EQ(obj->x, 0);
}


TEST(EnableCreateMethod, CreatesObject)
{
    auto ref = gko::ReferenceExecutor::create();
    auto obj = DummyObject::create(ref, 5);

    ASSERT_EQ(obj->get_executor(), ref);
    ASSERT_EQ(obj->x, 5);
}


struct ConvertibleToDummyObject
    : gko::EnablePolymorphicObject<ConvertibleToDummyObject>,
      gko::EnableCreateMethod<ConvertibleToDummyObject>,
      gko::EnablePolymorphicAssignment<ConvertibleToDummyObject>,
      gko::ConvertibleTo<DummyObject> {
    explicit ConvertibleToDummyObject(std::shared_ptr<const gko::Executor> exec,
                                      int v = {})
        : gko::EnablePolymorphicObject<ConvertibleToDummyObject>(
              std::move(exec)),
          x{v}
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
