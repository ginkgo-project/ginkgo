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

#include <cstdlib>
#include <fstream>
#include <ginkgo/ginkgo.hpp>
#include <iostream>

int main(int argc, char** argv)
{
    using value_type = double;
    using index_type = gko::int32;
    if (argc != 4) {
        std::cout << "Please execute with the following parameters:\n"
                  << argv[0]
                  << " <S matrix path> <L matrix path> <I matrix path>\n";
        std::exit(1);
    }
    std::ifstream S_file(argv[1]);
    std::ifstream L_file(argv[2]);
    std::ifstream I_file(argv[3]);
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using Coo = gko::matrix::Coo<value_type, index_type>;
    // Instantiate a CUDA executor
    auto gpu = gko::CudaExecutor::create(0, gko::OmpExecutor::create());
    // Read data
    auto S_csr = gko::read<Csr>(S_file, gpu);
    auto L = gko::read<Csr>(L_file, gpu);
    auto I = gko::read<Csr>(I_file, gpu);

    const auto num_row_ptrs = S_csr->get_size()[0] + 1;
    gko::array<index_type> row_ptrs_array(gpu, num_row_ptrs);
    gpu->copy_from(gpu, num_row_ptrs, S_csr->get_const_row_ptrs(),
                   row_ptrs_array.get_data());
    auto S_coo = Coo::create(gpu);
    S_csr->move_to(S_coo);

    auto S_row_ptrs = row_ptrs_array.get_data();
    auto S_row_idxs = S_coo->get_row_idxs();
    auto S_col_idxs = S_coo->get_col_idxs();
    auto S_values = S_coo->get_values();

    auto L_transpose_linop = L->transpose();
    auto L_transpose =
        static_cast<typename Csr::transposed_type*>(L_transpose_linop.get());
    // Solve system
    // Write result
    write(std::cout, I);
}
