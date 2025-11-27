// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
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
#include <ginkgo/core/matrix/permutation.hpp>

#include "core/base/extended_float.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"
#include "ginkgo/core/base/types.hpp"


namespace {


using comm_index_type = gko::experimental::distributed::comm_index_type;


template <typename ValueLocalGlobalIndexType>
class Bddc : public ::testing::Test {
protected:
    using value_type = typename std::tuple_element<
        0, decltype(ValueLocalGlobalIndexType())>::type;
    using real_type = gko::remove_complex<value_type>;
    using local_index_type = typename std::tuple_element<
        1, decltype(ValueLocalGlobalIndexType())>::type;
    using global_index_type = typename std::tuple_element<
        2, decltype(ValueLocalGlobalIndexType())>::type;
    using uint_type = typename gko::detail::float_traits<real_type>::bits_type;
    using Mtx = gko::matrix::Csr<value_type, local_index_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using RealVec = gko::matrix::Dense<gko::remove_complex<value_type>>;
    using perm_type = gko::matrix::Permutation<local_index_type>;

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
    void act_and_assert_classify_dofs(bool use_faces, bool use_edges)
    {
        auto input =
            gko::matrix::Dense<real_type>::create(ref, gko::dim<2>{9, 1});
        uint_type int_inner_val = 1 << 1;
        uint_type int_face_val = (1 << 1) | (1 << 2);
        uint_type int_diff_face_val = (1 << 0) | (1 << 2);
        uint_type int_edge_val = (1 << 0) | (1 << 1) | (1 << 2);
        uint_type int_vertex_val = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3);
        real_type inner_val, face_val, diff_face_val, edge_val, vertex_val;
        std::memcpy(&inner_val, &int_inner_val, sizeof(uint_type));
        std::memcpy(&face_val, &int_face_val, sizeof(uint_type));
        std::memcpy(&diff_face_val, &int_diff_face_val, sizeof(uint_type));
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
        auto i =
            gko::experimental::distributed::preconditioner::dof_type::inner;
        auto f =
            use_faces
                ? gko::experimental::distributed::preconditioner::dof_type::face
                : gko::experimental::distributed::preconditioner::dof_type::
                      inactive;
        auto e =
            use_edges
                ? gko::experimental::distributed::preconditioner::dof_type::edge
                : gko::experimental::distributed::preconditioner::dof_type::
                      inactive;
        auto v =
            gko::experimental::distributed::preconditioner::dof_type::vertex;
        gko::array<gko::experimental::distributed::preconditioner::dof_type>
            ref_result{ref, {i, i, e, i, i, e, f, f, v}};
        gko::array<local_index_type> permutation_array{ref, 9};
        gko::array<local_index_type> ref_permutation_array;
        if (use_edges) {
            ref_permutation_array =
                gko::array<local_index_type>{ref, {0, 1, 3, 4, 6, 7, 2, 5, 8}};
        } else {
            ref_permutation_array =
                gko::array<local_index_type>{ref, {0, 1, 3, 4, 2, 5, 6, 7, 8}};
        }
        gko::array<local_index_type> interface_sizes{ref};
        gko::array<local_index_type> ref_interface_sizes;
        if (use_faces && use_edges) {
            ref_interface_sizes = gko::array<local_index_type>{ref, {2, 2, 1}};
        } else if (use_faces || use_edges) {
            ref_interface_sizes = gko::array<local_index_type>{ref, {2, 1}};
        } else {
            ref_interface_sizes = gko::array<local_index_type>{ref, {1}};
        }
        gko::array<real_type> owning_labels{ref};
        gko::array<real_type> ref_owning_labels;
        if (use_faces) {
            ref_owning_labels = gko::array<real_type>{ref, {face_val}};
        } else {
            ref_owning_labels = gko::array<real_type>{ref, {}};
        }
        gko::array<real_type> unique_labels{ref};
        gko::array<real_type> ref_unique_labels;
        if (use_faces && use_edges) {
            ref_unique_labels =
                gko::array<real_type>{ref, {face_val, edge_val, vertex_val}};
        } else if (use_faces) {
            ref_unique_labels =
                gko::array<real_type>{ref, {face_val, vertex_val}};
        } else if (use_edges) {
            ref_unique_labels =
                gko::array<real_type>{ref, {edge_val, vertex_val}};
        } else {
            ref_unique_labels = gko::array<real_type>{ref, {vertex_val}};
        }
        gko::size_type n_inner_idxs, n_face_idxs, n_edge_idxs, n_vertices,
            n_faces, n_edges, n_constraints;
        int n_owning_interfaces;

        gko::kernels::reference::bddc::classify_dofs(
            ref, input.get(), 1, result, permutation_array, interface_sizes,
            unique_labels, owning_labels, n_inner_idxs, n_face_idxs,
            n_edge_idxs, n_vertices, n_faces, n_edges, n_constraints,
            n_owning_interfaces, use_faces, use_edges);

        gko::array<local_index_type> permutation_array_2{permutation_array};
        auto perm = perm_type::create(ref, permutation_array_2);
        auto perm_mtx = input->permute(perm, gko::matrix::permute_mode::rows);

        GKO_ASSERT_ARRAY_EQ(result, ref_result);
        GKO_ASSERT_ARRAY_EQ(permutation_array, ref_permutation_array);
        GKO_ASSERT_ARRAY_EQ(interface_sizes, ref_interface_sizes);
        GKO_ASSERT_EQ(n_inner_idxs, 4);
        GKO_ASSERT_EQ(n_face_idxs, 2);
        GKO_ASSERT_EQ(n_edge_idxs, 2);
        GKO_ASSERT_EQ(n_vertices, 1);
        GKO_ASSERT_EQ(n_faces, 1);
        GKO_ASSERT_EQ(n_edges, 1);
        GKO_ASSERT_EQ(n_constraints, static_cast<gko::size_type>(
                                         use_faces ? (use_edges ? 3 : 2)
                                                   : (use_edges ? 2 : 1)));
        GKO_ASSERT_EQ(n_owning_interfaces, use_faces ? 1 : 0);
        GKO_ASSERT_ARRAY_EQ(owning_labels, ref_owning_labels);
        GKO_ASSERT_ARRAY_EQ(unique_labels, ref_unique_labels);

        auto perm_input =
            gko::as<RealVec>(input->row_permute(&permutation_array));
        gko::device_matrix_data<real_type, local_index_type> C_data{ref};
        gko::array<local_index_type> ref_row_idxs;
        gko::array<local_index_type> ref_col_idxs;
        gko::array<real_type> ref_values;
        if (use_faces) {
            if (use_edges) {
                C_data = gko::device_matrix_data<real_type, local_index_type>{
                    ref, gko::dim<2>{2, 8}, 4};
                ref_row_idxs = gko::array<local_index_type>{ref, {0, 0, 1, 1}};
                ref_col_idxs = gko::array<local_index_type>{ref, {4, 5, 6, 7}};
                ref_values = gko::array<real_type>{ref, {.5, .5, .5, .5}};
            } else {
                C_data = gko::device_matrix_data<real_type, local_index_type>{
                    ref, gko::dim<2>{1, 8}, 2};
                ref_row_idxs = gko::array<local_index_type>{ref, {0, 0}};
                ref_col_idxs = gko::array<local_index_type>{ref, {6, 7}};
                ref_values = gko::array<real_type>{ref, {.5, .5}};
            }
        } else if (use_edges) {
            C_data = gko::device_matrix_data<real_type, local_index_type>{
                ref, gko::dim<2>{1, 8}, 2};
            ref_row_idxs = gko::array<local_index_type>{ref, {0, 0}};
            ref_col_idxs = gko::array<local_index_type>{ref, {6, 7}};
            ref_values = gko::array<real_type>{ref, {.5, .5}};
        } else {
            C_data = gko::device_matrix_data<real_type, local_index_type>{
                ref, gko::dim<2>{0, 8}, 0};
            ref_row_idxs = gko::array<local_index_type>{ref, {}};
            ref_col_idxs = gko::array<local_index_type>{ref, {}};
            ref_values = gko::array<real_type>{ref, {}};
        }
        gko::size_type n_inactive = n_inner_idxs;
        n_inactive += use_faces ? 0 : 2;
        n_inactive += use_edges ? 0 : 2;

        gko::kernels::reference::bddc::generate_constraints(
            ref, perm_input.get(), n_inactive, n_constraints - 1,
            interface_sizes, C_data);

        GKO_ASSERT_ARRAY_EQ(ref_row_idxs, gko::make_const_array_view(
                                              ref, 8 - n_inactive,
                                              C_data.get_const_row_idxs()));
        GKO_ASSERT_ARRAY_EQ(ref_col_idxs, gko::make_const_array_view(
                                              ref, 8 - n_inactive,
                                              C_data.get_const_col_idxs()));
        GKO_ASSERT_ARRAY_EQ(
            ref_values, gko::make_const_array_view(ref, 8 - n_inactive,
                                                   C_data.get_const_values()));

        auto global_labels = gko::array<real_type>{
            ref, {diff_face_val, face_val, edge_val, vertex_val}};
        auto local_labels =
            gko::array<real_type>{ref, {face_val, edge_val, vertex_val}};
        auto lambda =
            gko::matrix::Dense<value_type>::create(ref, gko::dim<2>{3, 3});
        lambda->at(0, 0) = 1.;
        lambda->at(0, 1) = -2.;
        lambda->at(0, 2) = 3.;
        lambda->at(1, 0) = -4.;
        lambda->at(1, 1) = 5.;
        lambda->at(1, 2) = -6.;
        lambda->at(2, 0) = 7.;
        lambda->at(2, 1) = -8.;
        lambda->at(2, 2) = 9.;
        gko::array<local_index_type> local_to_global{ref, {10, 10, 10}};
        gko::device_matrix_data<value_type, int> coarse_contribution{
            ref, gko::dim<2>{4, 4}, 9};
        gko::array<int> ref_coarse_row_idxs{ref, {1, 1, 1, 2, 2, 2, 3, 3, 3}};
        gko::array<int> ref_coarse_col_idxs{ref, {1, 2, 3, 1, 2, 3, 1, 2, 3}};
        gko::array<value_type> ref_coarse_values{
            ref, {-1., 2., -3., 4., -5., 6., -7., 8., -9.}};
        gko::array<local_index_type> local_to_global_ref{ref, {1, 2, 3}};

        gko::kernels::reference::bddc::build_coarse_contribution(
            ref, local_labels, global_labels, lambda.get(), coarse_contribution,
            local_to_global);

        GKO_ASSERT_ARRAY_EQ(
            ref_coarse_row_idxs,
            gko::make_const_array_view(
                ref, 9, coarse_contribution.get_const_row_idxs()));
        GKO_ASSERT_ARRAY_EQ(
            ref_coarse_col_idxs,
            gko::make_const_array_view(
                ref, 9, coarse_contribution.get_const_col_idxs()));
        GKO_ASSERT_ARRAY_EQ(
            ref_coarse_values,
            gko::make_const_array_view(ref, 9,
                                       coarse_contribution.get_const_values()));
        GKO_ASSERT_ARRAY_EQ(local_to_global, local_to_global_ref);
    }

    std::shared_ptr<const gko::ReferenceExecutor> ref;
};

TYPED_TEST_SUITE(Bddc, gko::test::ValueLocalGlobalIndexTypes,
                 TupleTypenameNameGenerator);


TYPED_TEST(Bddc, ClassifyDofsFacesEdges)
{
    this->act_and_assert_classify_dofs(true, true);
}


TYPED_TEST(Bddc, ClassifyDofsFacesNoEdges)
{
    this->act_and_assert_classify_dofs(true, false);
}


TYPED_TEST(Bddc, ClassifyDofsNoFacesEdges)
{
    this->act_and_assert_classify_dofs(false, true);
}


TYPED_TEST(Bddc, ClassifyDofsNoFacesNoEdges)
{
    this->act_and_assert_classify_dofs(false, false);
}


}  // namespace
