// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <complex>
#include <iostream>
#include <vector>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/mtx_io.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/reorder/amd.hpp>
#include <ginkgo/core/reorder/nested_dissection.hpp>
#include <ginkgo/core/reorder/rcm.hpp>

#include "core/components/fill_array_kernels.hpp"
#include "core/factorization/elimination_forest_kernels.hpp"
#include "core/utils/matrix_utils.hpp"


#ifdef GKO_TOOL_COMPLEX
using value_type = std::complex<double>;
#else
using value_type = double;
#endif


using matrix_data = gko::matrix_data<value_type, gko::int64>;


int main(int argc, char** argv)
{
    if (argc == 1) {
        std::cerr
            << "Usage: " << argv[0]
            << " [-b] [operation1] [operation2]\nApplies the given operations "
               "to the input matrix read from stdin\nand writes it to "
               "stdout.\nUses binary format if -b is set, otherwise matrix "
               "market format.Operations are:\n"
               "  lower-triangular   removes nonzeros above the diagonal\n"
               "  upper-triangular   removes nonzeros below the diagonal\n"
               "  remove-diagonal    removes diagonal entries\n"
               "  remove-zeros       removes numerical zero entries\n"
               "  unit-diagonal      sets diagonal entries to zero\n"
               "  symmetric          computes (A + A^T)/2\n"
               "  skew-symmetric     computes (A - A^T)/2\n"
               "  hermitian          computes (A + A^H)/2\n"
               "  skew-hermitian     computes (A - A^H)/2\n"
               "  diagonal-dominant  scales diagonal entries so the\n"
               "                     matrix becomes diagonally dominant\n"
               "  spd                symmetric + diagonal-dominant\n"
               "  hpd                hermitian + diagonal-dominant\n"
               "  skeleton-tree      computes an edge-max-node MST that has "
               "                     the same elimination tree as A"
            << std::endl;
        return 1;
    }
    bool binary = std::string{argv[1]} == "-b";

    auto data = gko::read_generic_raw<value_type, gko::int64>(std::cin);
    data.sort_row_major();
    for (int argi = binary ? 2 : 1; argi < argc; argi++) {
        std::string arg{argv[argi]};
        const std::string reorder_str{"reorder:"};
        if (arg == "lower-triangular") {
            gko::utils::make_lower_triangular(data);
        } else if (arg == "upper-triangular") {
            gko::utils::make_upper_triangular(data);
        } else if (arg == "remove-diagonal") {
            gko::utils::make_remove_diagonal(data);
        } else if (arg == "remove-zeros") {
            data.remove_zeros();
        } else if (arg == "unit-diagonal") {
            gko::utils::make_unit_diagonal(data);
        } else if (arg == "symmetric") {
            gko::utils::make_symmetric(data);
        } else if (arg == "skew-symmetric") {
            gko::utils::make_symmetric_generic(data, [](auto v) { return -v; });
        } else if (arg == "hermitian") {
            gko::utils::make_hermitian(data);
        } else if (arg == "skew-hermitian") {
            gko::utils::make_symmetric_generic(
                data, [](auto v) { return -gko::conj(v); });
        } else if (arg == "diagonal-dominant") {
            gko::utils::make_diag_dominant(data);
        } else if (arg == "spd") {
            gko::utils::make_spd(data);
        } else if (arg == "hpd") {
            gko::utils::make_hpd(data);
        } else if (arg == "skeleton-tree") {
            auto exec = gko::ReferenceExecutor::create();
            auto mtx = gko::matrix::Csr<value_type, gko::int64>::create(exec);
            mtx->read(data);
            auto tree = gko::matrix::Csr<value_type, gko::int64>::create(
                exec, mtx->get_size(), mtx->get_size()[0]);
            gko::kernels::reference::elimination_forest::compute_skeleton_tree(
                exec, mtx->get_const_row_ptrs(), mtx->get_const_col_idxs(),
                mtx->get_size()[0], tree->get_row_ptrs(), tree->get_col_idxs());
            gko::kernels::reference::components::fill_array(
                exec, mtx->get_values(), mtx->get_num_stored_elements(),
                gko::one<value_type>());
            tree->write(data);
            std::unique_ptr<gko::factorization::elimination_forest<gko::int64>>
                etree, etree2;
            gko::factorization::compute_elimination_forest(mtx.get(), etree);
            gko::factorization::compute_elimination_forest(tree.get(), etree2);
            if (!std::equal(
                    etree->parents.get_const_data(),
                    etree->parents.get_const_data() + mtx->get_size()[0],
                    etree2->parents.get_const_data())) {
                std::cerr << "failed\n";
            }
        } else if (arg.size() > reorder_str.size() &&
                   std::equal(reorder_str.begin(), reorder_str.end(),
                              arg.begin())) {
            std::string op{arg.begin() + reorder_str.size(), arg.end()};
            std::unique_ptr<gko::matrix::Permutation<gko::int64>> perm;
            auto exec = gko::ReferenceExecutor::create();
            auto mtx = gko::share(
                gko::matrix::Csr<value_type, gko::int64>::create(exec));
            mtx->read(data);
            if (op == "amd") {
                perm = gko::experimental::reorder::Amd<gko::int64>::build()
                           .on(exec)
                           ->generate(mtx);
            } else if (op == "rcm") {
                perm = gko::experimental::reorder::Rcm<gko::int64>::build()
                           .on(exec)
                           ->generate(mtx);
#if GKO_HAVE_METIS
            } else if (op == "nd") {
                perm = gko::experimental::reorder::NestedDissection<
                           value_type, gko::int64>::build()
                           .on(exec)
                           ->generate(mtx);
#endif
            } else {
                std::cerr << "Unknown reordering algorithm " << op << std::endl;
                return 1;
            }
            mtx->permute(perm)->write(data);
        } else {
            std::cerr << "Unknown operation " << arg << std::endl;
            return 1;
        }
    }
    if (binary) {
        gko::write_binary_raw(std::cout, data);
    } else {
        gko::write_raw(std::cout, data, gko::layout_type::coordinate);
    }
}
