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
class Vector : public ::testing::Test {
public:
    using value_type = typename std::tuple_element<
        0, decltype(ValueLocalGlobalIndexType())>::type;
    using local_index_type = typename std::tuple_element<
        1, decltype(ValueLocalGlobalIndexType())>::type;
    using global_index_type = typename std::tuple_element<
        2, decltype(ValueLocalGlobalIndexType())>::type;
    using part_type =
        gko::distributed::Partition<local_index_type, global_index_type>;
    using md_type = gko::matrix_data<value_type, global_index_type>;
    using d_md_type = gko::device_matrix_data<value_type, global_index_type>;
    using dist_vec_type = gko::distributed::Vector<value_type, local_index_type,
                                                   global_index_type>;
    using dense_type = gko::matrix::Dense<value_type>;
    using nz_type = gko::matrix_data_entry<value_type, global_index_type>;

    Vector()
        : ref(gko::ReferenceExecutor::create()),
          comm(MPI_COMM_WORLD),
          part(gko::share(part_type::build_from_contiguous(
              this->ref, {ref, {0, 2, 4, 6}}))),
          md_a{{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}, {10, 11}},
          md_b{{10, -11}, {8, -9}, {-6, 7}, {4, -5}, {2, -3}, {0, 1}},
          vec_a(dist_vec_type::create(ref, comm, part)),
          vec_b(dist_vec_type::create(ref, comm, part))
    {
        vec_a->read_distributed(md_a, part);
        vec_b->read_distributed(md_b, part);
    }

    std::shared_ptr<gko::Executor> ref;
    gko::mpi::communicator comm;
    std::shared_ptr<part_type> part;

    md_type md_a;
    md_type md_b;

    std::unique_ptr<dist_vec_type> vec_a;
    std::unique_ptr<dist_vec_type> vec_b;
};


TYPED_TEST_SUITE(Vector, gko::test::ValueLocalGlobalIndexTypes);

TYPED_TEST(Vector, CanReadGlobalMatrixData)
{
    using part_type = typename TestFixture::part_type;
    using value_type = typename TestFixture::value_type;
    auto vec = TestFixture::dist_vec_type::create(this->ref, this->comm);
    auto rank = this->comm.rank();

    vec->read_distributed(this->md_a, this->part);

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


TYPED_TEST(Vector, CanReadGlobalMatrixDataSomeEmpty)
{
    using part_type = typename TestFixture::part_type;
    using value_type = typename TestFixture::value_type;
    auto part = gko::share(
        part_type::build_from_contiguous(this->ref, {this->ref, {0, 0, 6, 6}}));
    auto vec = TestFixture::dist_vec_type::create(this->ref, this->comm);
    auto rank = this->comm.rank();

    vec->read_distributed(this->md_a, part);

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


TYPED_TEST(Vector, CanReadGlobalDeviceMatrixData)
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

    vec->read_distributed(md, part);

    GKO_ASSERT_EQUAL_DIMENSIONS(vec->get_size(), gko::dim<2>(6, 2));
    GKO_ASSERT_EQUAL_DIMENSIONS(vec->get_local()->get_size(),
                                gko::dim<2>(2, 2));
    GKO_ASSERT_MTX_NEAR(vec->get_local(), ref_data[rank], r<vt>::value);
}

TYPED_TEST(Vector, CanReadGlobalMatrixDataScattered)
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

    vec->read_distributed(md, part);

    GKO_ASSERT_EQUAL_DIMENSIONS(vec->get_size(), gko::dim<2>(6, 2));
    GKO_ASSERT_EQUAL_DIMENSIONS(vec->get_local()->get_size(), ref_size[rank]);
    GKO_ASSERT_MTX_NEAR(vec->get_local(), ref_data[rank], r<value_type>::value);
}


TYPED_TEST(Vector, CanReadLocalMatrixData)
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

    vec->read_distributed(md[rank], part);

    GKO_ASSERT_EQUAL_DIMENSIONS(vec->get_size(), gko::dim<2>(6, 2));
    GKO_ASSERT_EQUAL_DIMENSIONS(vec->get_local()->get_size(),
                                gko::dim<2>(2, 2));
    GKO_ASSERT_MTX_NEAR(vec->get_local(), ref_data[rank], r<value_type>::value);
}


TYPED_TEST(Vector, CanReadLocalMatrixDataSomeEmpty)
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

    vec->read_distributed(md[rank], part);

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


TYPED_TEST(Vector, ComputesDotProduct)
{
    using dense_type = typename TestFixture::dense_type;
    using value_type = typename TestFixture::value_type;
    auto res = dense_type::create(this->ref, gko::dim<2>{1, 2});
    auto ref_res =
        gko::initialize<dense_type>(I<I<value_type>>{{32, -54}}, this->ref);

    this->vec_a->compute_dot(this->vec_b.get(), res.get());

    GKO_ASSERT_MTX_NEAR(res, ref_res, r<value_type>::value);
}


TYPED_TEST(Vector, ComputesConjDot)
{
    using dense_type = typename TestFixture::dense_type;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::global_index_type;
    using real_type = typename gko::remove_complex<value_type>;
    using dist_vec_type = typename TestFixture::dist_vec_type;
    auto md_a = gko::test::generate_random_matrix_data<value_type, index_type>(
        6, 2, std::uniform_int_distribution<int>(2, 2),
        std::normal_distribution<real_type>(0, 1), std::ranlux48{42});
    auto md_b = gko::test::generate_random_matrix_data<value_type, index_type>(
        6, 2, std::uniform_int_distribution<int>(2, 2),
        std::normal_distribution<real_type>(0, 1), std::ranlux48{42});
    auto dist_vec_a = dist_vec_type::create(this->ref, this->comm, this->part);
    auto dist_vec_b = dist_vec_type::create(this->ref, this->comm, this->part);
    auto dense_vec_a = dense_type::create(this->ref);
    auto dense_vec_b = dense_type::create(this->ref);
    dist_vec_a->read_distributed(md_a, this->part);
    dist_vec_b->read_distributed(md_b, this->part);
    dense_vec_a->read(md_a);
    dense_vec_b->read(md_b);
    auto res = dense_type::create(this->ref, gko::dim<2>{1, 2});
    auto ref_res = dense_type::create(this->ref, gko::dim<2>{1, 2});

    dist_vec_a->compute_dot(dist_vec_b.get(), res.get());
    dense_vec_a->compute_dot(dense_vec_b.get(), ref_res.get());

    GKO_ASSERT_MTX_NEAR(res, ref_res, r<value_type>::value);
}


TYPED_TEST(Vector, ComputesNorm)
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

}  // namespace
