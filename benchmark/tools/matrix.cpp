// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <complex>
#include <iostream>
#include <vector>


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/mtx_io.hpp>


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
               "  hpd                hermitian + diagonal-dominant"
            << std::endl;
        return 1;
    }
    bool binary = std::string{argv[1]} == "-b";

    auto data = gko::read_generic_raw<value_type, gko::int64>(std::cin);
    data.sort_row_major();
    for (int argi = binary ? 2 : 1; argi < argc; argi++) {
        std::string arg{argv[argi]};
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
