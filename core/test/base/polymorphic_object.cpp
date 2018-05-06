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

#include <core/base/polymorphic_object.hpp>


#include <gtest/gtest.h>


#include <core/base/std_extensions.hpp>


namespace {


struct DummyObject : gko::EnablePolymorphicObject<DummyObject>,
                     gko::EnableCreateMethod<DummyObject>,
                     gko::EnablePolymorphicAssignment<DummyObject> {
    explicit DummyObject(std::shared_ptr<const gko::Executor> exec, int v = {})
        : gko::EnablePolymorphicObject<DummyObject>(std::move(exec)), x{v}
    {}

    int x;
};


class EnablePolymorphicObject : public testing::Test {
protected:
    std::shared_ptr<gko::ReferenceExecutor> ref{
        gko::ReferenceExecutor::create()};
    std::shared_ptr<gko::CpuExecutor> cpu{gko::CpuExecutor::create()};
    std::unique_ptr<DummyObject> obj{new DummyObject(ref, 5)};
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
    auto def = obj->create_default(cpu);

    ASSERT_NE(def, obj);
    ASSERT_EQ(def->get_executor(), cpu);
    ASSERT_EQ(def->x, 0);
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
    auto clone = obj->clone(cpu);

    ASSERT_NE(clone, obj);
    ASSERT_EQ(clone->get_executor(), cpu);
    ASSERT_EQ(clone->x, 5);
}


TEST_F(EnablePolymorphicObject, CopiesObject)
{
    auto copy = gko::xstd::make_unique<DummyObject>(cpu, 7);

    copy->copy_from(gko::lend(obj));

    ASSERT_NE(copy, obj);
    ASSERT_EQ(copy->get_executor(), cpu);
    ASSERT_EQ(copy->x, 5);
}


TEST_F(EnablePolymorphicObject, MovesObject)
{
    auto copy = gko::xstd::make_unique<DummyObject>(ref, 7);

    copy->copy_from(gko::give(obj));

    ASSERT_NE(copy, obj);
    ASSERT_EQ(copy->get_executor(), ref);
    ASSERT_EQ(copy->x, 5);
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


}  // namespace
