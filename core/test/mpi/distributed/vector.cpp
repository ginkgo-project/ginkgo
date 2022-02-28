/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include <mpi.h>


#include <gtest/gtest.h>


#include <ginkgo/config.hpp>
#include <ginkgo/core/distributed/vector.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename ValueLocalGlobalIndexType>
class VectorCreation : public ::testing::Test {
public:
    using value_type =
        typename std::tuple_element<0, decltype(
                                           ValueLocalGlobalIndexType())>::type;
    using local_index_type =
        typename std::tuple_element<1, decltype(
                                           ValueLocalGlobalIndexType())>::type;
    using global_index_type =
        typename std::tuple_element<2, decltype(
                                           ValueLocalGlobalIndexType())>::type;
    using part_type =
        gko::distributed::Partition<local_index_type, global_index_type>;
    using md_type = gko::matrix_data<value_type, global_index_type>;
    using d_md_type = gko::device_matrix_data<value_type, global_index_type>;
    using dist_vec_type = gko::distributed::Vector<value_type>;
    using dense_type = gko::matrix::Dense<value_type>;

    VectorCreation()
        : ref(gko::ReferenceExecutor::create()),
          comm(MPI_COMM_WORLD),
          part(gko::share(part_type::build_from_contiguous(
              this->ref, {ref, {0, 2, 4, 6}}))),
          md{{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}, {10, 11}},
          md_localized{{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}, {{8, 9}, {10, 11}}}
    {}

    void SetUp() override { ASSERT_GE(this->comm.size(), 3); }

    std::shared_ptr<gko::Executor> ref;
    gko::mpi::communicator comm;
    std::shared_ptr<part_type> part;

    md_type md;
    md_type md_localized[3];
};


TYPED_TEST_SUITE(VectorCreation, gko::test::ValueLocalGlobalIndexTypes);


TYPED_TEST(VectorCreation, CanReadGlobalMatrixData)
{
    using part_type = typename TestFixture::part_type;
    using value_type = typename TestFixture::value_type;
    auto vec = TestFixture::dist_vec_type::create(this->ref, this->comm);
    auto rank = this->comm.rank();

    vec->read_distributed(this->md, this->part.get());

    GKO_ASSERT_EQUAL_DIMENSIONS(vec->get_size(), gko::dim<2>(6, 2));
    GKO_ASSERT_EQUAL_DIMENSIONS(vec->get_local()->get_size(),
                                gko::dim<2>(2, 2));
    I<I<value_type>> ref_data[3] = {
        {{0, 1}, {2, 3}},
        {{4, 5}, {6, 7}},
        {{8, 9}, {10, 11}},
    };
    GKO_ASSERT_MTX_NEAR(vec->get_local(), ref_data[rank], r<value_type>::value);
}


TYPED_TEST(VectorCreation, CanReadGlobalMatrixDataSomeEmpty)
{
    using part_type = typename TestFixture::part_type;
    using value_type = typename TestFixture::value_type;
    auto part = gko::share(
        part_type::build_from_contiguous(this->ref, {this->ref, {0, 0, 6, 6}}));
    auto vec = TestFixture::dist_vec_type::create(this->ref, this->comm);
    auto rank = this->comm.rank();

    vec->read_distributed(this->md, part.get());

    GKO_ASSERT_EQUAL_DIMENSIONS(vec->get_size(), gko::dim<2>(6, 2));
    if (rank == 1) {
        GKO_ASSERT_EQUAL_DIMENSIONS(vec->get_local()->get_size(),
                                    gko::dim<2>(6, 2));
        GKO_ASSERT_MTX_NEAR(
            vec->get_local(),
            l({{0., 1.}, {2., 3.}, {4., 5.}, {6., 7.}, {8., 9.}, {10., 11.}}),
            r<value_type>::value);
    } else {
        GKO_ASSERT_EQUAL_DIMENSIONS(vec->get_local()->get_size(),
                                    gko::dim<2>(0, 2));
    }
}


TYPED_TEST(VectorCreation, CanReadGlobalDeviceMatrixData)
{
    using it = typename TestFixture::global_index_type;
    using d_md_type = typename TestFixture::d_md_type;
    using part_type = typename TestFixture::part_type;
    using vt = typename TestFixture::value_type;
    d_md_type md{
        this->ref, gko::dim<2>{6, 2},
        gko::Array<it>{this->ref, I<it>{0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5}},
        gko::Array<it>{this->ref, I<it>{0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1}},
        gko::Array<vt>{this->ref, I<vt>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}}};
    auto part = gko::share(
        part_type::build_from_contiguous(this->ref, {this->ref, {0, 2, 4, 6}}));
    auto vec = TestFixture::dist_vec_type::create(this->ref, this->comm);
    auto rank = this->comm.rank();
    I<I<vt>> ref_data[3] = {
        {{0, 1}, {2, 3}},
        {{4, 5}, {6, 7}},
        {{8, 9}, {10, 11}},
    };

    vec->read_distributed(md, part.get());

    GKO_ASSERT_EQUAL_DIMENSIONS(vec->get_size(), gko::dim<2>(6, 2));
    GKO_ASSERT_EQUAL_DIMENSIONS(vec->get_local()->get_size(),
                                gko::dim<2>(2, 2));
    GKO_ASSERT_MTX_NEAR(vec->get_local(), ref_data[rank], r<vt>::value);
}


TYPED_TEST(VectorCreation, CanReadGlobalMatrixDataScattered)
{
    using md_type = typename TestFixture::md_type;
    using part_type = typename TestFixture::part_type;
    using value_type = typename TestFixture::value_type;
    md_type md{{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}, {10, 11}};
    auto part = gko::share(part_type::build_from_mapping(
        this->ref, {this->ref, {0, 1, 2, 0, 2, 0}}, 3));
    auto vec = TestFixture::dist_vec_type::create(this->ref, this->comm);
    auto rank = this->comm.rank();
    gko::dim<2> ref_size[3] = {{3, 2}, {1, 2}, {2, 2}};
    I<I<value_type>> ref_data[3] = {
        {{0, 1}, {6, 7}, {10, 11}},
        {{2, 3}},
        {{4, 5}, {8, 9}},
    };

    vec->read_distributed(md, part.get());

    GKO_ASSERT_EQUAL_DIMENSIONS(vec->get_size(), gko::dim<2>(6, 2));
    GKO_ASSERT_EQUAL_DIMENSIONS(vec->get_local()->get_size(), ref_size[rank]);
    GKO_ASSERT_MTX_NEAR(vec->get_local(), ref_data[rank], r<value_type>::value);
}


TYPED_TEST(VectorCreation, CanReadLocalMatrixData)
{
    using md_type = typename TestFixture::md_type;
    using part_type = typename TestFixture::part_type;
    using value_type = typename TestFixture::value_type;
    md_type md[3] = {
        {gko::dim<2>{6, 2}, {{0, 0, 0}, {0, 1, 1}, {1, 0, 2}, {1, 1, 3}}},
        {gko::dim<2>{6, 2}, {{2, 0, 4}, {2, 1, 5}, {3, 0, 6}, {3, 1, 7}}},
        {gko::dim<2>{6, 2}, {{4, 0, 8}, {4, 1, 9}, {5, 0, 10}, {5, 1, 11}}}};
    auto part = gko::share(
        part_type::build_from_contiguous(this->ref, {this->ref, {0, 2, 4, 6}}));
    auto vec = TestFixture::dist_vec_type::create(this->ref, this->comm);
    auto rank = this->comm.rank();
    I<I<value_type>> ref_data[3] = {
        {{0, 1}, {2, 3}},
        {{4, 5}, {6, 7}},
        {{8, 9}, {10, 11}},
    };

    vec->read_distributed(md[rank], part.get());

    GKO_ASSERT_EQUAL_DIMENSIONS(vec->get_size(), gko::dim<2>(6, 2));
    GKO_ASSERT_EQUAL_DIMENSIONS(vec->get_local()->get_size(),
                                gko::dim<2>(2, 2));
    GKO_ASSERT_MTX_NEAR(vec->get_local(), ref_data[rank], r<value_type>::value);
}


TYPED_TEST(VectorCreation, CanReadLocalMatrixDataSomeEmpty)
{
    using md_type = typename TestFixture::md_type;
    using part_type = typename TestFixture::part_type;
    using value_type = typename TestFixture::value_type;
    md_type md[3] = {{gko::dim<2>{6, 2}, {}},
                     {gko::dim<2>{6, 2},
                      // clang-format off
                      {{0, 0, 0}, {0, 1, 1},
                       {1, 0, 2}, {1, 1, 3},
                       {2, 0, 4}, {2, 1, 5},
                       {3, 0, 6}, {3, 1, 7},
                       {4, 0, 8}, {4, 1, 9},
                       {5, 0, 10}, {5, 1, 11}}},
                     // clang-format on
                     {gko::dim<2>{6, 2}, {}}};
    auto part = gko::share(
        part_type::build_from_contiguous(this->ref, {this->ref, {0, 0, 6, 6}}));
    auto vec = TestFixture::dist_vec_type::create(this->ref, this->comm);
    auto rank = this->comm.rank();

    vec->read_distributed(md[rank], part.get());

    GKO_ASSERT_EQUAL_DIMENSIONS(vec->get_size(), gko::dim<2>(6, 2));
    if (rank == 1) {
        GKO_ASSERT_EQUAL_DIMENSIONS(vec->get_local()->get_size(),
                                    gko::dim<2>(6, 2));
        GKO_ASSERT_MTX_NEAR(
            vec->get_local(),
            I<I<value_type>>(
                {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}, {10, 11}}),
            r<value_type>::value);
    } else {
        GKO_ASSERT_EQUAL_DIMENSIONS(vec->get_local()->get_size(),
                                    gko::dim<2>(0, 2));
    }
}


TYPED_TEST(VectorCreation, CanCreateFromLocalVectorAndSize)
{
    using dist_vec_type = typename TestFixture::dist_vec_type;
    using dense_type = typename TestFixture::dense_type;
    auto local_vec = dense_type::create(this->ref);
    local_vec->read(this->md_localized[this->comm.rank()]);
    auto clone_local_vec = gko::clone(local_vec);

    auto vec = dist_vec_type::create(this->ref, this->comm, gko::dim<2>{6, 2},
                                     local_vec.get());

    GKO_ASSERT_EQUAL_DIMENSIONS(vec, gko::dim<2>(6, 2));
    GKO_ASSERT_MTX_NEAR(vec->get_local(), clone_local_vec, 0);
}


TYPED_TEST(VectorCreation, CanCreateFromLocalVectorWithoutSize)
{
    using dist_vec_type = typename TestFixture::dist_vec_type;
    using dense_type = typename TestFixture::dense_type;
    auto local_vec = dense_type::create(this->ref);
    local_vec->read(this->md_localized[this->comm.rank()]);
    auto clone_local_vec = gko::clone(local_vec);

    auto vec = dist_vec_type::create(this->ref, this->comm, local_vec.get());

    GKO_ASSERT_EQUAL_DIMENSIONS(vec, gko::dim<2>(6, 2));
    GKO_ASSERT_MTX_NEAR(vec->get_local(), clone_local_vec, 0);
}


template <typename ValueType>
class VectorReductions : public ::testing::Test {
public:
    using value_type = ValueType;
    using local_index_type = gko::int32;
    using global_index_type = gko::int64;
    using part_type =
        gko::distributed::Partition<local_index_type, global_index_type>;
    using md_type = gko::matrix_data<value_type, global_index_type>;
    using dist_vec_type = gko::distributed::Vector<value_type>;
    using dense_type = gko::matrix::Dense<value_type>;

    VectorReductions()
        : ref(gko::ReferenceExecutor::create()),
          comm(MPI_COMM_WORLD),
          part(gko::share(part_type::build_from_contiguous(
              this->ref, {ref, {0, 2, 4, 6}}))),
          vec_a(dist_vec_type::create(ref, comm)),
          vec_b(dist_vec_type::create(ref, comm))
    {
        md_type md_a{{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}, {10, 11}};
        md_type md_b{{10, -11}, {8, -9}, {-6, 7}, {4, -5}, {2, -3}, {0, 1}};

        vec_a->read_distributed(md_a, part.get());
        vec_b->read_distributed(md_b, part.get());
    }

    void SetUp() override { ASSERT_GE(this->comm.size(), 3); }

    std::shared_ptr<gko::Executor> ref;
    gko::mpi::communicator comm;
    std::shared_ptr<part_type> part;

    std::unique_ptr<dist_vec_type> vec_a;
    std::unique_ptr<dist_vec_type> vec_b;
};


TYPED_TEST_SUITE(VectorReductions, gko::test::ValueTypes);


TYPED_TEST(VectorReductions, ComputesDotProduct)
{
    using dense_type = typename TestFixture::dense_type;
    using value_type = typename TestFixture::value_type;
    auto res = dense_type::create(this->ref, gko::dim<2>{1, 2});
    auto ref_res =
        gko::initialize<dense_type>(I<I<value_type>>{{32, -54}}, this->ref);

    this->vec_a->compute_dot(this->vec_b.get(), res.get());

    GKO_ASSERT_MTX_NEAR(res, ref_res, r<value_type>::value);
}


TYPED_TEST(VectorReductions, ComputesConjDot)
{
    using dense_type = typename TestFixture::dense_type;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::global_index_type;
    using real_type = typename gko::remove_complex<value_type>;
    using dist_vec_type = typename TestFixture::dist_vec_type;
    auto md_a = gko::test::generate_random_matrix_data<value_type, index_type>(
        6, 2, std::uniform_int_distribution<int>(2, 2),
        std::normal_distribution<real_type>(0, 1),
        std::default_random_engine{42});
    auto md_b = gko::test::generate_random_matrix_data<value_type, index_type>(
        6, 2, std::uniform_int_distribution<int>(2, 2),
        std::normal_distribution<real_type>(0, 1),
        std::default_random_engine{42});
    auto dist_vec_a = dist_vec_type::create(this->ref, this->comm);
    auto dist_vec_b = dist_vec_type::create(this->ref, this->comm);
    auto dense_vec_a = dense_type::create(this->ref);
    auto dense_vec_b = dense_type::create(this->ref);
    dist_vec_a->read_distributed(md_a, this->part.get());
    dist_vec_b->read_distributed(md_b, this->part.get());
    dense_vec_a->read(md_a);
    dense_vec_b->read(md_b);
    auto res = dense_type::create(this->ref, gko::dim<2>{1, 2});
    auto ref_res = dense_type::create(this->ref, gko::dim<2>{1, 2});

    dist_vec_a->compute_dot(dist_vec_b.get(), res.get());
    dense_vec_a->compute_dot(dense_vec_b.get(), ref_res.get());

    GKO_ASSERT_MTX_NEAR(res, ref_res, r<value_type>::value);
}


TYPED_TEST(VectorReductions, ComputesNorm2)
{
    using dense_type = typename TestFixture::dense_type;
    using value_type = typename TestFixture::value_type;
    auto res = dense_type::absolute_type::create(this->ref, gko::dim<2>{1, 2});
    auto ref_res = gko::initialize<typename dense_type::absolute_type>(
        {{static_cast<gko::remove_complex<value_type>>(std::sqrt(220)),
          static_cast<gko::remove_complex<value_type>>(std::sqrt(286))}},
        this->ref);

    this->vec_a->compute_norm2(res.get());

    GKO_ASSERT_MTX_NEAR(res, ref_res, r<value_type>::value);
}


TYPED_TEST(VectorReductions, ComputesNorm1)
{
    using dense_type = typename TestFixture::dense_type;
    using value_type = typename TestFixture::value_type;
    auto res = dense_type::absolute_type::create(this->ref, gko::dim<2>{1, 2});
    auto ref_res = gko::initialize<typename dense_type::absolute_type>(
        {{30, 36}}, this->ref);

    this->vec_a->compute_norm1(res.get());

    GKO_ASSERT_MTX_NEAR(res, ref_res, r<value_type>::value);
}


template <typename ValueType>
class VectorLocalOp : public ::testing::Test {
public:
    using value_type = ValueType;
    using local_index_type = gko::int32;
    using global_index_type = gko::int64;
    using part_type =
        gko::distributed::Partition<local_index_type, global_index_type>;
    using md_type = gko::matrix_data<value_type, global_index_type>;
    using dist_vec_type = gko::distributed::Vector<value_type>;
    using dense_type = gko::matrix::Dense<value_type>;

    VectorLocalOp()
        : ref(gko::ReferenceExecutor::create()),
          comm(MPI_COMM_WORLD),
          part(gko::share(part_type::build_from_contiguous(
              this->ref, {ref, {0, 2, 4, 6}}))),
          vec_a(dist_vec_type::create(ref, comm)),
          vec_b(dist_vec_type::create(ref, comm)),
          engine(42)
    {
        md_type md_a{{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}, {10, 11}};
        md_type md_b{{10, -11}, {8, -9}, {-6, 7}, {4, -5}, {2, -3}, {0, 1}};

        vec_a->read_distributed(md_a, part.get());
        vec_b->read_distributed(md_b, part.get());
    }

    void SetUp() override { ASSERT_GE(this->comm.size(), 3); }

    auto generate_local_and_global_pair(gko::dim<2> local_size)
    {
        auto local_vec = gko::test::generate_random_matrix<dense_type>(
            local_size[0], local_size[1],
            std::uniform_int_distribution<gko::size_type>(0, local_size[1] - 1),
            std::normal_distribution<gko::remove_complex<value_type>>(),
            this->engine, this->ref);
        auto dist_vec = dist_vec_type::create(
            this->ref, this->comm,
            gko::dim<2>{local_size[0] * this->comm.size(), local_size[1]},
            local_size);
        dist_vec->get_local()->copy_from(local_vec.get());

        return std::make_pair(std::move(dist_vec), std::move(local_vec));
    }


    std::shared_ptr<gko::Executor> ref;
    gko::mpi::communicator comm;
    std::shared_ptr<part_type> part;

    std::unique_ptr<dist_vec_type> vec_a;
    std::unique_ptr<dist_vec_type> vec_b;

    std::default_random_engine engine;
};


TYPED_TEST_SUITE(VectorLocalOp, gko::test::ValueTypes);


TYPED_TEST(VectorLocalOp, ApplyNotSupported)
{
    using dist_vec_type = typename TestFixture::dist_vec_type;
    auto a = dist_vec_type::create(this->ref, this->comm, gko::dim<2>{2, 2},
                                   gko::dim<2>{2, 2});
    auto b = dist_vec_type::create(this->ref, this->comm, gko::dim<2>{2, 2},
                                   gko::dim<2>{2, 2});
    auto c = dist_vec_type::create(this->ref, this->comm, gko::dim<2>{2, 2},
                                   gko::dim<2>{2, 2});

    ASSERT_THROW(a->apply(b.get(), c.get()), gko::NotSupported);
}


TYPED_TEST(VectorLocalOp, AdvancedApplyNotSupported)
{
    using dist_vec_type = typename TestFixture::dist_vec_type;
    auto a = dist_vec_type::create(this->ref, this->comm, gko::dim<2>{2, 2},
                                   gko::dim<2>{2, 2});
    auto b = dist_vec_type::create(this->ref, this->comm, gko::dim<2>{1, 1},
                                   gko::dim<2>{1, 1});
    auto c = dist_vec_type::create(this->ref, this->comm, gko::dim<2>{2, 2},
                                   gko::dim<2>{2, 2});
    auto d = dist_vec_type::create(this->ref, this->comm, gko::dim<2>{1, 1},
                                   gko::dim<2>{1, 1});
    auto e = dist_vec_type::create(this->ref, this->comm, gko::dim<2>{2, 2},
                                   gko::dim<2>{2, 2});

    ASSERT_THROW(a->apply(b.get(), c.get(), d.get(), e.get()),
                 gko::NotSupported);
}


TYPED_TEST(VectorLocalOp, ConvertsToPrecision)
{
    using Vector = typename TestFixture::dist_vec_type;
    using T = typename TestFixture::value_type;
    using OtherT = typename gko::next_precision<T>;
    using OtherVector = typename gko::distributed::Vector<OtherT>;
    auto tmp = OtherVector::create(this->ref, this->comm);
    auto res = Vector::create(this->ref, this->comm);
    // If OtherT is more precise: 0, otherwise r
    auto residual = r<OtherT>::value < r<T>::value
                        ? gko::remove_complex<T>{0}
                        : gko::remove_complex<T>{r<OtherT>::value};

    this->vec_a->convert_to(tmp.get());
    tmp->convert_to(res.get());

    GKO_ASSERT_MTX_NEAR(this->vec_a->get_local(), res->get_local(), residual);
}


TYPED_TEST(VectorLocalOp, MovesToPrecision)
{
    using Vector = typename TestFixture::dist_vec_type;
    using T = typename TestFixture::value_type;
    using OtherT = typename gko::next_precision<T>;
    using OtherVector = typename gko::distributed::Vector<OtherT>;
    auto tmp = OtherVector::create(this->ref, this->comm);
    auto res = Vector::create(this->ref, this->comm);
    auto clone_vec_a = gko::clone(this->vec_a);
    // If OtherT is more precise: 0, otherwise r
    auto residual = r<OtherT>::value < r<T>::value
                        ? gko::remove_complex<T>{0}
                        : gko::remove_complex<T>{r<OtherT>::value};

    clone_vec_a->move_to(tmp.get());
    tmp->move_to(res.get());

    GKO_ASSERT_MTX_NEAR(this->vec_a->get_local(), res->get_local(), residual);
}


TYPED_TEST(VectorLocalOp, ComputeAbsoluteSameAsLocal)
{
    gko::size_type local_size = 20;
    gko::size_type num_cols = 7;
    auto pair =
        this->generate_local_and_global_pair(gko::dim<2>{local_size, num_cols});
    auto& dist = pair.first;
    auto& local = pair.second;

    auto dist_absolute = dist->compute_absolute();
    auto local_absolute = local->compute_absolute();

    GKO_ASSERT_MTX_NEAR(dist_absolute->get_const_local(), local_absolute, 0);
}


TYPED_TEST(VectorLocalOp, ComputeAbsoluteInplaceSameAsLocal)
{
    gko::size_type local_size = 20;
    gko::size_type num_cols = 7;
    auto pair =
        this->generate_local_and_global_pair(gko::dim<2>{local_size, num_cols});
    auto& dist = pair.first;
    auto& local = pair.second;

    dist->compute_absolute_inplace();
    local->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(dist->get_const_local(), local, 0);
}


TYPED_TEST(VectorLocalOp, MakeComplexSameAsLocal)
{
    gko::size_type local_size = 20;
    gko::size_type num_cols = 7;
    auto pair =
        this->generate_local_and_global_pair(gko::dim<2>{local_size, num_cols});
    auto& dist = pair.first;
    auto& local = pair.second;

    auto dist_complex = dist->make_complex();
    auto local_complex = local->make_complex();

    GKO_ASSERT_MTX_NEAR(dist_complex->get_const_local(), local_complex, 0);
}


TYPED_TEST(VectorLocalOp, MakeComplexInplaceSameAsLocal)
{
    gko::size_type local_size = 20;
    gko::size_type num_cols = 7;
    auto pair =
        this->generate_local_and_global_pair(gko::dim<2>{local_size, num_cols});
    auto& dist = pair.first;
    auto& local = pair.second;
    auto dist_complex = dist->make_complex();
    dist_complex->fill(0);
    auto local_complex = local->make_complex();
    local_complex->fill(0);

    dist->make_complex(dist_complex.get());
    local->make_complex(local_complex.get());

    GKO_ASSERT_MTX_NEAR(dist_complex->get_const_local(), local_complex, 0);
}


TYPED_TEST(VectorLocalOp, GetRealSameAsLocal)
{
    gko::size_type local_size = 20;
    gko::size_type num_cols = 7;
    auto pair =
        this->generate_local_and_global_pair(gko::dim<2>{local_size, num_cols});
    auto& dist = pair.first;
    auto& local = pair.second;

    auto dist_real = dist->get_real();
    auto local_real = local->get_real();

    GKO_ASSERT_MTX_NEAR(dist_real->get_const_local(), local_real, 0);
}


TYPED_TEST(VectorLocalOp, GetRealInplaceSameAsLocal)
{
    gko::size_type local_size = 20;
    gko::size_type num_cols = 7;
    auto pair =
        this->generate_local_and_global_pair(gko::dim<2>{local_size, num_cols});
    auto& dist = pair.first;
    auto& local = pair.second;
    auto dist_real = dist->get_real();
    dist_real->fill(0);
    auto local_real = local->get_real();
    local_real->fill(0);

    dist->get_real(dist_real.get());
    local->get_real(local_real.get());

    GKO_ASSERT_MTX_NEAR(dist_real->get_const_local(), local_real, 0);
}


TYPED_TEST(VectorLocalOp, GetImagSameAsLocal)
{
    gko::size_type local_size = 20;
    gko::size_type num_cols = 7;
    auto pair =
        this->generate_local_and_global_pair(gko::dim<2>{local_size, num_cols});
    auto& dist = pair.first;
    auto& local = pair.second;

    auto dist_imag = dist->get_imag();
    auto local_imag = local->get_imag();

    GKO_ASSERT_MTX_NEAR(dist_imag->get_const_local(), local_imag, 0);
}


TYPED_TEST(VectorLocalOp, GetImagInplaceSameAsLocal)
{
    gko::size_type local_size = 20;
    gko::size_type num_cols = 7;
    auto pair =
        this->generate_local_and_global_pair(gko::dim<2>{local_size, num_cols});
    auto& dist = pair.first;
    auto& local = pair.second;
    auto dist_imag = dist->get_imag();
    dist_imag->fill(0);
    auto local_imag = local->get_imag();
    local_imag->fill(0);

    dist->get_imag(dist_imag.get());
    local->get_imag(local_imag.get());

    GKO_ASSERT_MTX_NEAR(dist_imag->get_const_local(), local_imag, 0);
}


TYPED_TEST(VectorLocalOp, FillSameAsLocal)
{
    using value_type = typename TestFixture::value_type;
    gko::size_type local_size = 20;
    gko::size_type num_cols = 7;
    auto pair =
        this->generate_local_and_global_pair(gko::dim<2>{local_size, num_cols});
    auto& dist = pair.first;
    auto& local = pair.second;
    auto value = gko::test::detail::get_rand_value<value_type>(
        std::normal_distribution<gko::remove_complex<value_type>>(),
        this->engine);

    dist->fill(value);
    local->fill(value);

    GKO_ASSERT_MTX_NEAR(dist->get_const_local(), local, 0);
}


TYPED_TEST(VectorLocalOp, ScaleSameAsLocal)
{
    using value_type = typename TestFixture::value_type;
    gko::size_type local_size = 20;
    gko::size_type num_cols = 7;
    auto pair =
        this->generate_local_and_global_pair(gko::dim<2>{local_size, num_cols});
    auto& dist = pair.first;
    auto& local = pair.second;
    auto value = gko::test::generate_random_matrix(
        1, num_cols,
        std::uniform_int_distribution<gko::size_type>(num_cols, num_cols),
        std::normal_distribution<gko::remove_complex<value_type>>(),
        this->engine, this->ref);

    dist->scale(value.get());
    local->scale(value.get());

    GKO_ASSERT_MTX_NEAR(dist->get_const_local(), local, 0);
}


TYPED_TEST(VectorLocalOp, InvScaleSameAsLocal)
{
    using value_type = typename TestFixture::value_type;
    gko::size_type local_size = 20;
    gko::size_type num_cols = 7;
    auto pair =
        this->generate_local_and_global_pair(gko::dim<2>{local_size, num_cols});
    auto& dist = pair.first;
    auto& local = pair.second;
    auto value = gko::test::generate_random_matrix(
        1, num_cols,
        std::uniform_int_distribution<gko::size_type>(num_cols, num_cols),
        std::uniform_real_distribution<gko::remove_complex<value_type>>(1.0,
                                                                        2.0),
        this->engine, this->ref);

    dist->inv_scale(value.get());
    local->inv_scale(value.get());

    GKO_ASSERT_MTX_NEAR(dist->get_const_local(), local, 0);
}


TYPED_TEST(VectorLocalOp, AddScaleSameAsLocal)
{
    using value_type = typename TestFixture::value_type;
    gko::size_type local_size = 20;
    gko::size_type num_cols = 7;
    auto pair =
        this->generate_local_and_global_pair(gko::dim<2>{local_size, num_cols});
    auto pair_b =
        this->generate_local_and_global_pair(gko::dim<2>{local_size, num_cols});
    auto& dist = pair.first;
    auto& local = pair.second;
    auto& dist_b = pair_b.first;
    auto& local_b = pair_b.second;
    auto value = gko::test::generate_random_matrix(
        1, num_cols,
        std::uniform_int_distribution<gko::size_type>(num_cols, num_cols),
        std::normal_distribution<gko::remove_complex<value_type>>(),
        this->engine, this->ref);

    dist->add_scaled(value.get(), dist_b.get());
    local->add_scaled(value.get(), local_b.get());

    GKO_ASSERT_MTX_NEAR(dist->get_const_local(), local, 0);
}


TYPED_TEST(VectorLocalOp, SubScaleSameAsLocal)
{
    using value_type = typename TestFixture::value_type;
    gko::size_type local_size = 20;
    gko::size_type num_cols = 7;
    auto pair =
        this->generate_local_and_global_pair(gko::dim<2>{local_size, num_cols});
    auto pair_b =
        this->generate_local_and_global_pair(gko::dim<2>{local_size, num_cols});
    auto& dist = pair.first;
    auto& local = pair.second;
    auto& dist_b = pair_b.first;
    auto& local_b = pair_b.second;
    auto value = gko::test::generate_random_matrix(
        1, num_cols,
        std::uniform_int_distribution<gko::size_type>(num_cols, num_cols),
        std::normal_distribution<gko::remove_complex<value_type>>(),
        this->engine, this->ref);

    dist->sub_scaled(value.get(), dist_b.get());
    local->sub_scaled(value.get(), local_b.get());

    GKO_ASSERT_MTX_NEAR(dist->get_const_local(), local, 0);
}


TYPED_TEST(VectorLocalOp, CreateRealViewSameAsLocal)
{
    using value_type = typename TestFixture::value_type;
    using real_type = gko::remove_complex<value_type>;

    auto real_view = this->vec_a->create_real_view();
    auto local_real_view = this->vec_a->get_const_local()->create_real_view();

    if (gko::is_complex<value_type>()) {
        EXPECT_EQ(real_view->get_size()[0], this->vec_a->get_size()[0]);
        EXPECT_EQ(real_view->get_size()[1], 2 * this->vec_a->get_size()[1]);
        EXPECT_EQ(real_view->get_const_local()->get_stride(),
                  2 * this->vec_a->get_const_local()->get_stride());
        GKO_ASSERT_MTX_NEAR(real_view->get_const_local(), local_real_view, 0.);
    } else {
        EXPECT_EQ(real_view->get_size()[0], this->vec_a->get_size()[0]);
        EXPECT_EQ(real_view->get_size()[1], this->vec_a->get_size()[1]);
        GKO_ASSERT_MTX_NEAR(real_view->get_const_local(), local_real_view, 0.);
    }
}


}  // namespace
