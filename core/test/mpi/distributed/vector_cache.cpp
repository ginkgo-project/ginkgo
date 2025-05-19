// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <mpi.h>

#include <gtest/gtest.h>

#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/distributed/vector_cache.hpp>

#include "core/distributed/vector_cache_accessor.hpp"
#include "core/test/utils.hpp"


template <typename ValueType>
class VectorCache : public ::testing::Test {
protected:
    using value_type = ValueType;
    using vector_type = gko::experimental::distributed::Vector<value_type>;

    VectorCache()
        : comm(gko::experimental::mpi::communicator(MPI_COMM_WORLD)),
          ref(gko::ReferenceExecutor::create()),
          rank(comm.rank()),
          num_ranks(comm.size()),
          default_local_size(rank + 1, 3),
          default_global_size((num_ranks + 1) * num_ranks / 2, 3),
          default_vector(vector_type::create(this->ref, this->comm,
                                             this->default_global_size,
                                             this->default_local_size))
    {}

    gko::experimental::distributed::detail::VectorCache<value_type> cache;
    gko::experimental::mpi::communicator comm;
    std::shared_ptr<gko::ReferenceExecutor> ref;
    int rank;
    int num_ranks;
    gko::dim<2> default_local_size;
    gko::dim<2> default_global_size;
    std::unique_ptr<vector_type> default_vector;
};


TYPED_TEST_SUITE(VectorCache, gko::test::ValueTypesBase, TypenameNameGenerator);


TYPED_TEST(VectorCache, CanDefaultConstruct)
{
    using value_type = typename TestFixture::value_type;
    gko::experimental::distributed::detail::VectorCache<value_type> cache;

    ASSERT_EQ(cache.get(), nullptr);
}


TYPED_TEST(VectorCache, CanInitWithSize)
{
    this->cache.init(this->ref, this->comm, this->default_global_size,
                     this->default_local_size);

    ASSERT_NE(this->cache.get(), nullptr);
    GKO_ASSERT_EQUAL_DIMENSIONS(this->cache->get_size(),
                                this->default_global_size);
    GKO_ASSERT_EQUAL_DIMENSIONS(this->cache->get_local_vector()->get_size(),
                                this->default_local_size);
    ASSERT_EQ(this->cache->get_executor(), this->ref);
}


TYPED_TEST(VectorCache, SecondInitWithSameSizeIsNoOp)
{
    this->cache.init(this->ref, this->comm, this->default_global_size,
                     this->default_local_size);
    auto first_ptr = this->cache.get();
    auto local_val_ptr = this->cache->get_local_vector()->get_const_values();

    this->cache.init(this->ref, this->comm, this->default_global_size,
                     this->default_local_size);

    ASSERT_NE(this->cache.get(), nullptr);
    ASSERT_EQ(this->cache.get(), first_ptr);
    ASSERT_EQ(this->cache->get_local_vector()->get_const_values(),
              local_val_ptr);
}


TYPED_TEST(VectorCache, SecondInitWithDifferentGlobalSizeInitializes)
{
    this->cache.init(this->ref, this->comm, this->default_global_size,
                     this->default_local_size);
    auto first_ptr = this->cache.get();
    auto local_val_ptr = this->cache->get_local_vector()->get_const_values();
    // generate different global size
    gko::dim<2> second_local_size(2 * (this->rank + 1), 3);
    gko::dim<2> second_global_size((this->num_ranks + 1) * this->num_ranks, 3);

    this->cache.init(this->ref, this->comm, second_global_size,
                     second_local_size);

    ASSERT_NE(this->cache.get(), nullptr);
    ASSERT_NE(this->cache.get(), first_ptr);
    ASSERT_NE(this->cache->get_local_vector()->get_const_values(),
              local_val_ptr);
    GKO_ASSERT_EQUAL_DIMENSIONS(this->cache->get_size(), second_global_size);
    GKO_ASSERT_EQUAL_DIMENSIONS(this->cache->get_local_vector()->get_size(),
                                second_local_size);
    ASSERT_EQ(this->cache->get_executor(), this->ref);
}


TYPED_TEST(VectorCache, SecondInitWithDifferentLocalSizeInitializes)
{
    this->cache.init(this->ref, this->comm, this->default_global_size,
                     this->default_local_size);
    auto first_ptr = this->cache.get();
    auto local_val_ptr = this->cache->get_local_vector()->get_const_values();
    auto local_size = this->default_local_size;
    // switch the size of rank 0 and rank 1 to generate different local size but
    // the same global size
    if (this->rank == 0) {
        local_size = gko::dim<2>(2, 3);
    } else if (this->rank == 1) {
        local_size = gko::dim<2>(1, 3);
    }

    this->cache.init(this->ref, this->comm, this->default_global_size,
                     local_size);

    ASSERT_NE(this->cache.get(), nullptr);
    ASSERT_EQ(this->cache.get(), first_ptr);
    // we use move to replace the value pointer not the dense pointer
    if (this->rank >= 2) {
        ASSERT_EQ(this->cache->get_local_vector()->get_const_values(),
                  local_val_ptr);
    } else {
        ASSERT_NE(this->cache->get_local_vector()->get_const_values(),
                  local_val_ptr);
    }
    GKO_ASSERT_EQUAL_DIMENSIONS(this->cache->get_size(),
                                this->default_global_size);
    GKO_ASSERT_EQUAL_DIMENSIONS(this->cache->get_local_vector()->get_size(),
                                local_size);
    ASSERT_EQ(this->cache->get_executor(), this->ref);
}


TYPED_TEST(VectorCache, CanInitFromVector)
{
    this->cache.init_from(this->default_vector.get());

    ASSERT_NE(this->cache.get(), nullptr);
    ASSERT_NE(this->cache.get(), this->default_vector.get());
    GKO_ASSERT_EQUAL_DIMENSIONS(this->cache->get_size(),
                                this->default_global_size);
    GKO_ASSERT_EQUAL_DIMENSIONS(this->cache->get_local_vector()->get_size(),
                                this->default_local_size);
    ASSERT_EQ(this->cache->get_executor(), this->ref);
}


TYPED_TEST(VectorCache, SecondInitFromSameDenseIsNoOp)
{
    this->cache.init_from(this->default_vector.get());
    auto first_ptr = this->cache.get();
    auto local_val_ptr = this->cache->get_local_vector()->get_const_values();

    this->cache.init_from(this->default_vector.get());

    ASSERT_NE(this->cache.get(), nullptr);
    ASSERT_NE(this->cache.get(), this->default_vector.get());
    ASSERT_EQ(this->cache.get(), first_ptr);
    ASSERT_EQ(this->cache->get_local_vector()->get_const_values(),
              local_val_ptr);
}


TYPED_TEST(VectorCache, SecondInitFromDifferentDenseWithSameSizeIsNoOp)
{
    using vector_type = typename TestFixture::vector_type;
    this->cache.init_from(this->default_vector.get());
    auto first_ptr = this->cache.get();
    auto local_val_ptr = this->cache->get_local_vector()->get_const_values();
    auto vector =
        vector_type::create(this->ref, this->comm, this->default_global_size,
                            this->default_local_size);

    this->cache.init_from(vector.get());

    ASSERT_NE(this->cache.get(), nullptr);
    ASSERT_NE(this->cache.get(), this->default_vector.get());
    ASSERT_EQ(this->cache.get(), first_ptr);
    ASSERT_EQ(this->cache->get_local_vector()->get_const_values(),
              local_val_ptr);
}


TYPED_TEST(VectorCache,
           SecondInitFromDifferentVectorWithDifferentGlobalSizeInitializes)
{
    using vector_type = typename TestFixture::vector_type;
    this->cache.init_from(this->default_vector.get());
    auto first_ptr = this->cache.get();
    auto local_val_ptr = this->cache->get_local_vector()->get_const_values();
    gko::dim<2> second_local_size(2 * (this->rank + 1), 3);
    gko::dim<2> second_global_size((this->num_ranks + 1) * this->num_ranks, 3);
    auto vector = vector_type::create(this->ref, this->comm, second_global_size,
                                      second_local_size);

    this->cache.init_from(vector.get());

    ASSERT_NE(this->cache.get(), nullptr);
    ASSERT_NE(this->cache.get(), this->default_vector.get());
    ASSERT_NE(this->cache.get(), first_ptr);
    ASSERT_NE(this->cache->get_local_vector()->get_const_values(),
              local_val_ptr);
    GKO_ASSERT_EQUAL_DIMENSIONS(this->cache->get_size(), second_global_size);
    GKO_ASSERT_EQUAL_DIMENSIONS(this->cache->get_local_vector()->get_size(),
                                second_local_size);
    ASSERT_EQ(this->cache->get_executor(), this->ref);
}


TYPED_TEST(VectorCache,
           SecondInitFromDifferentVectorWithDifferentLocalSizeInitializes)
{
    using vector_type = typename TestFixture::vector_type;
    this->cache.init_from(this->default_vector.get());
    auto first_ptr = this->cache.get();
    auto local_val_ptr = this->cache->get_local_vector()->get_const_values();
    auto local_size = this->default_local_size;
    // switch the size of rank 0 and rank 1 to generate different local size but
    // the same global size
    if (this->rank == 0) {
        local_size = gko::dim<2>(2, 3);
    } else if (this->rank == 1) {
        local_size = gko::dim<2>(1, 3);
    }
    auto vector = vector_type::create(this->ref, this->comm,
                                      this->default_global_size, local_size);

    this->cache.init_from(vector.get());

    ASSERT_NE(this->cache.get(), nullptr);
    ASSERT_NE(this->cache.get(), this->default_vector.get());
    ASSERT_EQ(this->cache.get(), first_ptr);
    if (this->rank >= 2) {
        ASSERT_EQ(this->cache->get_local_vector()->get_const_values(),
                  local_val_ptr);
    } else {
        ASSERT_NE(this->cache->get_local_vector()->get_const_values(),
                  local_val_ptr);
    }
    GKO_ASSERT_EQUAL_DIMENSIONS(this->cache->get_size(),
                                this->default_global_size);
    GKO_ASSERT_EQUAL_DIMENSIONS(this->cache->get_local_vector()->get_size(),
                                local_size);
    ASSERT_EQ(this->cache->get_executor(), this->ref);
}


TYPED_TEST(VectorCache, VectorIsNotCopied)
{
    using value_type = typename TestFixture::value_type;
    this->cache.init(this->ref, this->comm, this->default_global_size,
                     this->default_local_size);
    gko::experimental::distributed::detail::VectorCache<value_type> cache(
        this->cache);

    ASSERT_EQ(cache.get(), nullptr);
    GKO_ASSERT_EQUAL_DIMENSIONS(this->cache->get_size(),
                                this->default_global_size);
    GKO_ASSERT_EQUAL_DIMENSIONS(this->cache->get_local_vector()->get_size(),
                                this->default_local_size);
}


TYPED_TEST(VectorCache, VectorIsNotMoved)
{
    using value_type = typename TestFixture::value_type;
    this->cache.init(this->ref, this->comm, this->default_global_size,
                     this->default_local_size);
    gko::experimental::distributed::detail::VectorCache<value_type> cache(
        std::move(this->cache));

    ASSERT_EQ(cache.get(), nullptr);
    GKO_ASSERT_EQUAL_DIMENSIONS(this->cache->get_size(),
                                this->default_global_size);
    GKO_ASSERT_EQUAL_DIMENSIONS(this->cache->get_local_vector()->get_size(),
                                this->default_local_size);
}


TYPED_TEST(VectorCache, VectorIsNotCopyAssigned)
{
    using value_type = typename TestFixture::value_type;
    this->cache.init(this->ref, this->comm, this->default_global_size,
                     this->default_local_size);
    gko::experimental::distributed::detail::VectorCache<value_type> cache;
    cache = this->cache;

    ASSERT_EQ(cache.get(), nullptr);
    GKO_ASSERT_EQUAL_DIMENSIONS(this->cache->get_size(),
                                this->default_global_size);
    GKO_ASSERT_EQUAL_DIMENSIONS(this->cache->get_local_vector()->get_size(),
                                this->default_local_size);
}


TYPED_TEST(VectorCache, VectorIsNotMoveAssigned)
{
    using value_type = typename TestFixture::value_type;
    this->cache.init(this->ref, this->comm, this->default_global_size,
                     this->default_local_size);
    gko::experimental::distributed::detail::VectorCache<value_type> cache;
    cache = std::move(this->cache);

    ASSERT_EQ(cache.get(), nullptr);
    GKO_ASSERT_EQUAL_DIMENSIONS(this->cache->get_size(),
                                this->default_global_size);
    GKO_ASSERT_EQUAL_DIMENSIONS(this->cache->get_local_vector()->get_size(),
                                this->default_local_size);
}


using generic_accessor =
    gko::experimental::distributed::detail::GenericVectorCacheAccessor;


template <typename ValueType>
class GenericVectorCache : public ::testing::Test {
protected:
    using value_type = ValueType;

    GenericVectorCache()
        : comm(gko::experimental::mpi::communicator(MPI_COMM_WORLD)),
          ref(gko::ReferenceExecutor::create()),
          rank(comm.rank()),
          num_ranks(comm.size()),
          default_local_size(rank + 1, 3),
          default_global_size((num_ranks + 1) * num_ranks / 2, 3)
    {}

    gko::experimental::mpi::communicator comm;
    std::shared_ptr<gko::ReferenceExecutor> ref;
    int rank;
    int num_ranks;
    gko::dim<2> default_local_size;
    gko::dim<2> default_global_size;
    gko::experimental::distributed::detail::GenericVectorCache cache;
};

TYPED_TEST_SUITE(GenericVectorCache, gko::test::ValueTypes,
                 TypenameNameGenerator);


TYPED_TEST(GenericVectorCache, GenericCanInitWithSize)
{
    using value_type = typename TestFixture::value_type;

    this->cache.init(this->ref, this->default_global_size,
                     this->default_local_size);
    // only initialize when knowning the type
    auto buffer = this->cache.template get<value_type>(this->comm);

    ASSERT_NE(buffer, nullptr);
    GKO_ASSERT_EQUAL_DIMENSIONS(buffer->get_size(), this->default_global_size);
    GKO_ASSERT_EQUAL_DIMENSIONS(buffer->get_local_vector()->get_size(),
                                this->default_local_size);
    ASSERT_EQ(buffer->get_communicator(), this->comm);
    ASSERT_EQ(buffer->get_executor(), this->ref);
}


TYPED_TEST(GenericVectorCache, SecondInitWithSameSizeIsNoOp)
{
    using value_type = typename TestFixture::value_type;
    this->cache.init(this->ref, this->default_global_size,
                     this->default_local_size);
    auto buffer = this->cache.template get<value_type>(this->comm);
    auto array_ptr =
        generic_accessor::get_workspace(this->cache).get_const_data();
    auto array_size = generic_accessor::get_workspace(this->cache).get_size();

    auto second_buffer = this->cache.template get<value_type>(this->comm);

    ASSERT_NE(second_buffer, nullptr);
    GKO_ASSERT_EQUAL_DIMENSIONS(second_buffer->get_size(),
                                this->default_global_size);
    GKO_ASSERT_EQUAL_DIMENSIONS(second_buffer->get_local_vector()->get_size(),
                                this->default_local_size);
    ASSERT_EQ(second_buffer->get_communicator(), this->comm);
    ASSERT_EQ(second_buffer->get_executor(), this->ref);
    ASSERT_EQ(array_ptr,
              generic_accessor::get_workspace(this->cache).get_const_data());
    ASSERT_EQ(array_size,
              generic_accessor::get_workspace(this->cache).get_size());
}


TYPED_TEST(GenericVectorCache, SecondInitWithTheSmallEqSizeIsNoOp)
{
    using value_type = typename TestFixture::value_type;
    gko::dim<2> second_local_size{1, 1};
    gko::dim<2> second_global_size{this->num_ranks, 1};
    this->cache.init(this->ref, this->default_global_size,
                     this->default_local_size);
    auto buffer = this->cache.template get<value_type>(this->comm);
    auto array_ptr =
        generic_accessor::get_workspace(this->cache).get_const_data();
    auto array_size = generic_accessor::get_workspace(this->cache).get_size();

    this->cache.init(this->ref, second_global_size, second_local_size);
    auto second_buffer = this->cache.template get<value_type>(this->comm);

    ASSERT_NE(second_buffer, nullptr);
    GKO_ASSERT_EQUAL_DIMENSIONS(second_buffer->get_size(), second_global_size);
    GKO_ASSERT_EQUAL_DIMENSIONS(second_buffer->get_local_vector()->get_size(),
                                second_local_size);
    ASSERT_EQ(second_buffer->get_communicator(), this->comm);
    ASSERT_EQ(second_buffer->get_executor(), this->ref);
    ASSERT_EQ(array_ptr,
              generic_accessor::get_workspace(this->cache).get_const_data());
    ASSERT_EQ(array_size,
              generic_accessor::get_workspace(this->cache).get_size());
}


TYPED_TEST(GenericVectorCache, SecondInitWithTheLargerSizeRecreate)
{
    using value_type = typename TestFixture::value_type;
    gko::dim<2> second_local_size{this->rank + 2, 3};
    gko::dim<2> second_global_size{this->num_ranks * (this->num_ranks + 3) / 2,
                                   3};
    this->cache.init(this->ref, this->default_global_size,
                     this->default_local_size);
    auto buffer = this->cache.template get<value_type>(this->comm);
    auto array_ptr =
        generic_accessor::get_workspace(this->cache).get_const_data();
    auto array_size = generic_accessor::get_workspace(this->cache).get_size();

    this->cache.init(this->ref, second_global_size, second_local_size);
    auto second_buffer = this->cache.template get<value_type>(this->comm);

    ASSERT_NE(second_buffer, nullptr);
    GKO_ASSERT_EQUAL_DIMENSIONS(second_buffer->get_size(), second_global_size);
    GKO_ASSERT_EQUAL_DIMENSIONS(second_buffer->get_local_vector()->get_size(),
                                second_local_size);
    ASSERT_EQ(second_buffer->get_communicator(), this->comm);
    ASSERT_EQ(second_buffer->get_executor(), this->ref);
    ASSERT_NE(array_ptr,
              generic_accessor::get_workspace(this->cache).get_const_data());
    ASSERT_GT(generic_accessor::get_workspace(this->cache).get_size(),
              array_size);
}


TYPED_TEST(GenericVectorCache, GenericCanInitWithSizeAndType)
{
    using value_type = typename TestFixture::value_type;
    using another_type = gko::next_precision<value_type>;
    this->cache.init(this->ref, this->default_global_size,
                     this->default_local_size);
    auto buffer = this->cache.template get<value_type>(this->comm);
    auto array_ptr =
        generic_accessor::get_workspace(this->cache).get_const_data();
    auto array_size = generic_accessor::get_workspace(this->cache).get_size();

    auto second_buffer = this->cache.template get<another_type>(this->comm);

    ASSERT_NE(second_buffer, nullptr);
    GKO_ASSERT_EQUAL_DIMENSIONS(second_buffer->get_size(),
                                this->default_global_size);
    GKO_ASSERT_EQUAL_DIMENSIONS(second_buffer->get_local_vector()->get_size(),
                                this->default_local_size);
    ASSERT_EQ(second_buffer->get_communicator(), this->comm);
    ASSERT_EQ(second_buffer->get_executor(), this->ref);
    if (sizeof(another_type) > sizeof(value_type)) {
        // the requring workspace will be bigger if the type is larger.
        ASSERT_NE(
            array_ptr,
            generic_accessor::get_workspace(this->cache).get_const_data());
        ASSERT_GT(generic_accessor::get_workspace(this->cache).get_size(),
                  array_size);
    } else {
        ASSERT_EQ(
            array_ptr,
            generic_accessor::get_workspace(this->cache).get_const_data());
        ASSERT_EQ(array_size,
                  generic_accessor::get_workspace(this->cache).get_size());
    }
}


TYPED_TEST(GenericVectorCache, GenericCanInitWithDifferentExecutor)
{
    using value_type = typename TestFixture::value_type;
    auto another_ref = gko::ReferenceExecutor::create();
    this->cache.init(this->ref, this->default_global_size,
                     this->default_local_size);
    auto buffer = this->cache.template get<value_type>(this->comm);
    auto array_ptr =
        generic_accessor::get_workspace(this->cache).get_const_data();
    auto array_size = generic_accessor::get_workspace(this->cache).get_size();

    this->cache.init(another_ref, this->default_global_size,
                     this->default_local_size);
    auto second_buffer = this->cache.template get<value_type>(this->comm);

    ASSERT_NE(second_buffer, nullptr);
    GKO_ASSERT_EQUAL_DIMENSIONS(second_buffer->get_size(),
                                this->default_global_size);
    GKO_ASSERT_EQUAL_DIMENSIONS(second_buffer->get_local_vector()->get_size(),
                                this->default_local_size);
    ASSERT_EQ(second_buffer->get_communicator(), this->comm);
    ASSERT_EQ(second_buffer->get_executor(), another_ref);
    // Different executor always regenerate different workspace
    ASSERT_NE(array_ptr,
              generic_accessor::get_workspace(this->cache).get_const_data());
}


TYPED_TEST(GenericVectorCache, WorkspaceIsNotCopied)
{
    using value_type = typename TestFixture::value_type;
    this->cache.init(this->ref, this->default_global_size,
                     this->default_local_size);
    auto buffer = this->cache.template get<value_type>(this->comm);

    gko::experimental::distributed::detail::GenericVectorCache cache(
        this->cache);

    ASSERT_EQ(generic_accessor::get_workspace(cache).get_size(), 0);
    ASSERT_EQ(generic_accessor::get_workspace(cache).get_executor(), nullptr);
}


TYPED_TEST(GenericVectorCache, WorkspaceIsNotMoved)
{
    using value_type = typename TestFixture::value_type;
    this->cache.init(this->ref, this->default_global_size,
                     this->default_local_size);
    auto buffer = this->cache.template get<value_type>(this->comm);

    gko::experimental::distributed::detail::GenericVectorCache cache(
        std::move(this->cache));

    ASSERT_EQ(generic_accessor::get_workspace(cache).get_size(), 0);
    ASSERT_EQ(generic_accessor::get_workspace(cache).get_executor(), nullptr);
}


TYPED_TEST(GenericVectorCache, WorkspaceIsNotCopyAssigned)
{
    using value_type = typename TestFixture::value_type;
    this->cache.init(this->ref, this->default_global_size,
                     this->default_local_size);
    auto buffer = this->cache.template get<value_type>(this->comm);
    gko::experimental::distributed::detail::GenericVectorCache cache;

    cache = this->cache;

    ASSERT_EQ(generic_accessor::get_workspace(cache).get_size(), 0);
    ASSERT_EQ(generic_accessor::get_workspace(cache).get_executor(), nullptr);
}


TYPED_TEST(GenericVectorCache, WorkspaceIsNotMoveAssigned)
{
    using value_type = typename TestFixture::value_type;
    this->cache.init(this->ref, this->default_global_size,
                     this->default_local_size);
    auto buffer = this->cache.template get<value_type>(this->comm);
    gko::experimental::distributed::detail::GenericVectorCache cache;

    cache = std::move(this->cache);

    ASSERT_EQ(generic_accessor::get_workspace(cache).get_size(), 0);
    ASSERT_EQ(generic_accessor::get_workspace(cache).get_executor(), nullptr);
}
