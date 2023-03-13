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

#include <ginkgo/core/reorder/amd.hpp>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/permutation.hpp>


namespace gko {
namespace reorder {
namespace {


template <typename IndexType>
void amd_reorder(std::shared_ptr<const Executor> host_exec, size_type num_rows,
                 const IndexType* row_ptrs, const IndexType* col_idxs,
                 IndexType* permutation)
{}

GKO_REGISTER_HOST_OPERATION(amd_reorder, amd_reorder);


}  // namespace


template <typename IndexType>
std::unique_ptr<matrix::Permutation<IndexType>> Amd<IndexType>::generate(
    std::shared_ptr<const LinOp> system_matrix) const
{
    auto product =
        std::unique_ptr<permutation_type>(static_cast<permutation_type*>(
            this->LinOpFactory::generate(std::move(system_matrix)).release()));
    return product;
}


template <typename IndexType>
std::unique_ptr<LinOp> Amd<IndexType>::generate_impl(
    std::shared_ptr<const LinOp> system_matrix) const
{
    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);
    const auto exec = this->get_executor();
    const auto host_exec = exec->get_master();
    const auto num_rows = system_matrix->get_size()[0];
    using complex_mtx = matrix::Csr<std::complex<double>, IndexType>;
    using real_mtx = matrix::Csr<double, IndexType>;
    std::unique_ptr<LinOp> converted;
    const IndexType* row_ptrs{};
    const IndexType* col_idxs{};
    if (auto convertible =
            dynamic_cast<ConvertibleTo<complex_mtx>*>(system_matrix.get())) {
        auto conv_csr = complex_mtx::create(host_exec);
        convertible->convert_to(conv_csr);
        row_ptrs = conv_csr->get_const_row_ptrs();
        col_idxs = conv_csr->get_const_col_idxs();
        converted = std::move(conv_csr);
    } else {
        auto conv_csr = real_mtx::create(host_exec);
        as<ConvertibleTo<real_mtx>>(system_matrix)->convert_to(conv_csr);
        row_ptrs = conv_csr->get_const_row_ptrs();
        col_idxs = conv_csr->get_const_col_idxs();
        converted = std::move(conv_csr);
    }

    array<IndexType> permutation{host_exec, num_rows};
    exec->run(make_amd_reorder(host_exec, num_rows, row_ptrs, col_idxs,
                               permutation.get_data()));
    permutation.set_executor(exec);

    return permutation_type::create(exec, dim<2>{num_rows, num_rows},
                                    std::move(permutation));
}


}  // namespace reorder
}  // namespace gko