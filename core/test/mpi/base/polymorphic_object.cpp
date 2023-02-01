/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include <gtest/gtest.h>


#include <ginkgo/core/distributed/polymorphic_object.hpp>


namespace {


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


struct DummyDistributedObject
    : gko::experimental::EnableDistributedPolymorphicObject<
          DummyDistributedObject>,
      gko::EnableCreateMethod<DummyDistributedObject>,
      gko::EnablePolymorphicAssignment<DummyDistributedObject>,
      gko::experimental::distributed::DistributedBase {
    explicit DummyDistributedObject(std::shared_ptr<const gko::Executor> exec,
                                    gko::experimental::mpi::communicator comm,
                                    int v = {})
        : gko::experimental::EnableDistributedPolymorphicObject<
              DummyDistributedObject>(std::move(exec)),
          gko::experimental::distributed::DistributedBase(std::move(comm)),
          x{v}
    {}

    DummyDistributedObject(const DummyDistributedObject& other)
        : DummyDistributedObject(other.get_executor(), other.get_communicator())
    {
        *this = other;
    }

    DummyDistributedObject(DummyDistributedObject&& other)
        : DummyDistributedObject(other.get_executor(), other.get_communicator())
    {
        *this = std::move(other);
    }

    DummyDistributedObject& operator=(const DummyDistributedObject& other)
    {
        if (this != &other) {
            x = other.x;
        }
        return *this;
    }

    DummyDistributedObject& operator=(DummyDistributedObject&& other) noexcept
    {
        if (this != &other) {
            x = std::exchange(other.x, 0);
        }
        return *this;
    }

    int x;
};


class EnableDistributedPolymorphicObject : public testing::Test {
protected:
    std::shared_ptr<gko::ReferenceExecutor> ref{
        gko::ReferenceExecutor::create()};
    // TDOD: We can't rely on Omp module being available in this test!
    std::shared_ptr<gko::OmpExecutor> omp{gko::OmpExecutor::create()};
    gko::experimental::mpi::communicator comm{MPI_COMM_WORLD};
    gko::experimental::mpi::communicator split_comm{comm.get(), comm.rank() < 2,
                                                    comm.rank()};
    std::unique_ptr<DummyDistributedObject> obj{
        new DummyDistributedObject(ref, split_comm, 5)};
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
            obj->remove_logger(logger.get());
        }
    }
};


TEST_F(EnableDistributedPolymorphicObject, CreatesConcreteClass)
{
    // this test passes as soon as an instance of `DummyDistributedObject` can
    // be created
}


TEST_F(EnableDistributedPolymorphicObject, CreatesDefaultObject)
{
    auto def = obj->create_default();

    ASSERT_NE(def, obj);
    ASSERT_EQ(def->get_executor(), ref);
    ASSERT_EQ(def->x, 0);
    ASSERT_EQ(def->get_communicator().get(), split_comm.get());
}


TEST_F(EnableDistributedPolymorphicObject,
       CreatesDefaultObjectOnAnotherExecutor)
{
    auto def = obj->create_default(omp);

    ASSERT_NE(def, obj);
    ASSERT_EQ(def->get_executor(), omp);
    ASSERT_EQ(def->x, 0);
    ASSERT_EQ(def->get_communicator().get(), split_comm.get());
}


TEST_F(EnableDistributedPolymorphicObject, CreatesDefaultObjectIsLogged)
{
    auto before_logger = *this->logger;

    auto def = obj->create_default();

    ASSERT_EQ(logger->create_started, before_logger.create_started + 1);
    ASSERT_EQ(logger->create_completed, before_logger.create_completed + 1);
}


TEST_F(EnableDistributedPolymorphicObject, ClonesObject)
{
    auto clone = obj->clone();

    ASSERT_NE(clone, obj);
    ASSERT_EQ(clone->get_executor(), ref);
    ASSERT_EQ(clone->x, 5);
    ASSERT_EQ(clone->get_communicator().get(), split_comm.get());
}


TEST_F(EnableDistributedPolymorphicObject, ClonesObjectToAnotherExecutor)
{
    auto clone = obj->clone(omp);

    ASSERT_NE(clone, obj);
    ASSERT_EQ(clone->get_executor(), omp);
    ASSERT_EQ(clone->x, 5);
    ASSERT_EQ(clone->get_communicator().get(), split_comm.get());
}


TEST_F(EnableDistributedPolymorphicObject, CopiesObject)
{
    auto copy = DummyDistributedObject::create(omp, comm, 7);

    copy->copy_from(obj);

    ASSERT_NE(copy, obj);
    ASSERT_EQ(copy->get_executor(), omp);
    ASSERT_EQ(copy->x, 5);
    ASSERT_EQ(copy->get_communicator().get(), comm.get());
    ASSERT_EQ(obj->get_executor(), ref);
    ASSERT_EQ(obj->x, 5);
}


TEST_F(EnableDistributedPolymorphicObject, CopiesObjectIsLogged)
{
    auto before_logger = *logger;
    auto copy = DummyDistributedObject::create(omp, comm, 7);
    copy->add_logger(logger);

    copy->copy_from(obj);

    ASSERT_EQ(logger->copy_started, before_logger.copy_started + 1);
    ASSERT_EQ(logger->copy_completed, before_logger.copy_completed + 1);
}


TEST_F(EnableDistributedPolymorphicObject, MovesObjectByCopyFromUniquePtr)
{
    auto copy = DummyDistributedObject::create(ref, comm, 7);

    copy->copy_from(gko::give(obj));

    ASSERT_NE(copy, obj);
    ASSERT_EQ(copy->get_executor(), ref);
    ASSERT_EQ(copy->x, 5);
    ASSERT_EQ(copy->get_communicator().get(), comm.get());
}


TEST_F(EnableDistributedPolymorphicObject,
       MovesObjectByCopyFromUniquePtrIsLogged)
{
    auto before_logger = *this->logger;
    auto copy = DummyDistributedObject::create(ref, comm, 7);
    copy->add_logger(logger);

    copy->copy_from(gko::give(obj));

    ASSERT_EQ(logger->move_started, before_logger.move_started + 1);
    ASSERT_EQ(logger->move_completed, before_logger.move_completed + 1);
}


TEST_F(EnableDistributedPolymorphicObject, MovesObject)
{
    auto copy = DummyDistributedObject::create(ref, comm, 7);

    copy->move_from(obj);

    ASSERT_NE(copy, obj);
    ASSERT_EQ(copy->get_executor(), ref);
    ASSERT_EQ(copy->x, 5);
    ASSERT_EQ(copy->get_communicator().get(), comm.get());
    ASSERT_EQ(obj->get_executor(), ref);
    ASSERT_EQ(obj->x, 0);
}


TEST_F(EnableDistributedPolymorphicObject, MovesObjectIsLogged)
{
    auto before_logger = *this->logger;
    auto copy = DummyDistributedObject::create(ref, comm, 7);
    copy->add_logger(logger);

    copy->move_from(obj);

    ASSERT_EQ(logger->move_started, before_logger.move_started + 1);
    ASSERT_EQ(logger->move_completed, before_logger.move_completed + 1);
}


TEST_F(EnableDistributedPolymorphicObject, MovesFromUniquePtr)
{
    auto copy = DummyDistributedObject::create(ref, comm, 7);

    copy->copy_from(gko::give(obj));

    ASSERT_NE(copy, obj);
    ASSERT_EQ(copy->get_executor(), ref);
    ASSERT_EQ(copy->x, 5);
    ASSERT_EQ(copy->get_communicator().get(), comm.get());
}


TEST_F(EnableDistributedPolymorphicObject, MovesFromUniquePtrIsLogged)
{
    auto before_logger = *this->logger;
    auto copy = DummyDistributedObject::create(ref, comm, 7);
    copy->add_logger(logger);

    copy->copy_from(gko::give(obj));

    ASSERT_EQ(logger->move_started, before_logger.move_started + 1);
    ASSERT_EQ(logger->move_completed, before_logger.move_completed + 1);
}


TEST_F(EnableDistributedPolymorphicObject, ClearsObject)
{
    obj->clear();

    ASSERT_EQ(obj->get_executor(), ref);
    ASSERT_EQ(obj->x, 0);
    ASSERT_EQ(obj->get_communicator().get(), split_comm.get());
}


}  // namespace
