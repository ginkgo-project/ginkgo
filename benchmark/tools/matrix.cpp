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
    data.ensure_row_major_order();
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
