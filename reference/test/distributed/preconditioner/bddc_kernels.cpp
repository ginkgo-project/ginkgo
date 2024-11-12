// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/preconditioner/bddc_kernels.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/base/extended_float.hpp"
#include "core/test/utils.hpp"


namespace {


using comm_index_type = gko::experimental::distributed::comm_index_type;


template <typename ValueLocalGlobalIndexType>
class Bddc : public ::testing::Test {
protected:
    using value_type = typename std::tuple_element<
        0, decltype(ValueLocalGlobalIndexType())>::type;
    using local_index_type = typename std::tuple_element<
        1, decltype(ValueLocalGlobalIndexType())>::type;
    using global_index_type = typename std::tuple_element<
        2, decltype(ValueLocalGlobalIndexType())>::type;
    using Mtx = gko::matrix::Csr<value_type, local_index_type>;
    using Vec = gko::matrix::Dense<value_type>;

    Bddc() : ref(gko::ReferenceExecutor::create()) {}

    /**
     * apply the `filter_non_owning_idxs` kernel and validate the result
     * against provided reference values
     *
     * @param size  the expected global matrix size
     * @param row_partition  the row partition passed to the kernel
     * @param col_partition  the column partition passed to the kernel
     * @param input_rows  the row indices passed to the kernel
     * @param input_cols  the column indices passed to the kernel
     * @param input_vals  the values passed to the kernel
     * @param non_owning_rows  the reference non owning row idxs.
     * @param non_owning_cols  the reference non owning col idxs.
     */
    void act_and_assert_classify_dofs()
    {
        using real_type = gko::remove_complex<value_type>;
        using uint_type =
            typename gko::detail::float_traits<real_type>::bits_type;
        auto input =
            gko::matrix::Dense<real_type>::create(ref, gko::dim<2>{9, 1});
        uint_type int_inner_val = 1 << 1;
        uint_type int_face_val = (1 << 1) | (1 << 2);
        uint_type int_edge_val = (1 << 1) | (1 << 2) | (1 << 3);
        uint_type int_vertex_val = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4);
        real_type inner_val, face_val, edge_val, vertex_val;
        std::memcpy(&inner_val, &int_inner_val, sizeof(uint_type));
        std::memcpy(&face_val, &int_face_val, sizeof(uint_type));
        std::memcpy(&edge_val, &int_edge_val, sizeof(uint_type));
        std::memcpy(&vertex_val, &int_vertex_val, sizeof(uint_type));
        input->at(0, 0) = inner_val;
        input->at(1, 0) = inner_val;
        input->at(2, 0) = edge_val;
        input->at(3, 0) = inner_val;
        input->at(4, 0) = inner_val;
        input->at(5, 0) = edge_val;
        input->at(6, 0) = face_val;
        input->at(7, 0) = face_val;
        input->at(8, 0) = vertex_val;
        gko::array<gko::experimental::distributed::preconditioner::dof_type>
            result{ref, 9};
        gko::array<gko::experimental::distributed::preconditioner::dof_type>
            ref_result{
                ref,
                {gko::experimental::distributed::preconditioner::dof_type::
                     inner,
                 gko::experimental::distributed::preconditioner::dof_type::
                     inner,
                 gko::experimental::distributed::preconditioner::dof_type::edge,
                 gko::experimental::distributed::preconditioner::dof_type::
                     inner,
                 gko::experimental::distributed::preconditioner::dof_type::
                     inner,
                 gko::experimental::distributed::preconditioner::dof_type::edge,
                 gko::experimental::distributed::preconditioner::dof_type::face,
                 gko::experimental::distributed::preconditioner::dof_type::face,
                 gko::experimental::distributed::preconditioner::dof_type::
                     vertex}};
        gko::array<local_index_type> permutation_array{ref, 9};
        gko::array<local_index_type> ref_permutation_array{
            ref, {0, 1, 3, 4, 6, 7, 2, 5, 8}};
        gko::array<local_index_type> interface_sizes{ref, 9};
        gko::array<local_index_type> ref_interface_sizes{
            ref, {4, 4, 4, 4, 2, 2, 2, 2, 1}};
        gko::size_type n_inner_idxs, n_face_idxs, n_edge_idxs, n_vertices,
            n_faces, n_edges, n_constraints;

        gko::kernels::reference::bddc::classify_dofs(
            ref, input.get(), 0, result, permutation_array, interface_sizes,
            n_inner_idxs, n_face_idxs, n_edge_idxs, n_vertices, n_faces,
            n_edges, n_constraints);

        GKO_ASSERT_ARRAY_EQ(result, ref_result);
        GKO_ASSERT_ARRAY_EQ(permutation_array, ref_permutation_array);
        GKO_ASSERT_ARRAY_EQ(interface_sizes, ref_interface_sizes);
        GKO_ASSERT_EQ(n_inner_idxs, 4);
        GKO_ASSERT_EQ(n_face_idxs, 2);
        GKO_ASSERT_EQ(n_edge_idxs, 2);
        GKO_ASSERT_EQ(n_vertices, 1);
        GKO_ASSERT_EQ(n_faces, 1);
        GKO_ASSERT_EQ(n_edges, 1);
        GKO_ASSERT_EQ(n_constraints, 3);
    }

    std::shared_ptr<const gko::ReferenceExecutor> ref;
};

TYPED_TEST_SUITE(Bddc, gko::test::ValueLocalGlobalIndexTypes,
                 TupleTypenameNameGenerator);


TYPED_TEST(Bddc, ClassifyDofs) { this->act_and_assert_classify_dofs(); }


}  // namespace