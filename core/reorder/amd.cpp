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


#include <map>
#include <set>
#include <vector>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/permutation.hpp>


#include "core/base/allocator.hpp"
#include "core/components/addressable_pq.hpp"


namespace gko {
namespace reorder {
namespace {


template <typename IndexType>
void amd_reorder(std::shared_ptr<const Executor> host_exec, IndexType num_rows,
                 const IndexType* row_ptrs, const IndexType* col_idxs,
                 IndexType* permutation)
{
    vector<IndexType> degrees(num_rows, {host_exec});
    std::vector<std::set<IndexType>> variable_neighbors(num_rows);
    std::vector<std::set<IndexType>> element_neighbors(num_rows);

    // compute initial degrees (assuming symmetric matrix)
    {
        IndexType output_idx{};
        for (IndexType row = 0; row < num_rows; row++) {
            const auto begin = row_ptrs[row];
            const auto end = row_ptrs[row + 1];
            IndexType degree{};
            for (auto nz = begin; nz < end; nz++) {
                const auto col = col_idxs[nz];
                if (col != row) {
                    variable_neighbors[row].insert(col);
                    degree++;
                }
            }
            degrees[row] = degree;
        }
    }

    // initialize min queue
    std::vector<IndexType> queue(num_rows);
    std::iota(queue.begin(), queue.end(), 0);
    std::sort(queue.begin(), queue.end(), [&](IndexType i, IndexType j) {
        return std::tie(degrees[i], i) < std::tie(degrees[j], j);
    });
    // pop minimum until we chose every variable as pivot
    for (IndexType i = 0; i < num_rows; i++) {
        auto pivot = queue.front();
        permutation[i] = pivot;
        queue.erase(queue.begin());
        auto& cur_var_neighbors = variable_neighbors[pivot];
        const auto& cur_el_neighbors = element_neighbors[pivot];
        // update L_p
        for (auto element : cur_el_neighbors) {
            assert(element != pivot);
            const auto& el_neighbors = variable_neighbors[element];
            cur_var_neighbors.insert(el_neighbors.begin(), el_neighbors.end());
        }
        cur_var_neighbors.erase(pivot);
        for (auto variable : cur_var_neighbors) {
            assert(variable != pivot);
            // A_i = A_i \setminus L_p \setminus p
            for (auto other_variable : cur_var_neighbors) {
                variable_neighbors[variable].erase(other_variable);
            }
            variable_neighbors[variable].erase(pivot);
            // E_i = E_i \setminus E_p \cup {p}
            for (auto element : cur_el_neighbors) {
                element_neighbors[variable].erase(element);
            }
            element_neighbors[variable].insert(pivot);
            // d_i = |A_i \setminus i| + |\bigcup_{e \in E_i} L_e \setminus i|
            auto set1 = variable_neighbors[variable];
            set1.erase(variable);
            std::set<IndexType> set2;
            for (auto element : element_neighbors[variable]) {
                set2.insert(variable_neighbors[element].begin(),
                            variable_neighbors[element].end());
            }
            set2.erase(variable);
            auto set3 = set1;
            set3.insert(set2.begin(), set2.end());
            // TODO why are they disjoint?
            assert(set3.size() == set1.size() + set2.size());
            // update d_i in queue
            degrees[variable] = set1.size() + set2.size();
        }
        std::sort(queue.begin(), queue.end(), [&](IndexType i, IndexType j) {
            return std::tie(degrees[i], i) < std::tie(degrees[j], j);
        });
    }
}

GKO_REGISTER_HOST_OPERATION(amd_reorder, amd_reorder);


}  // namespace


template <typename IndexType>
Amd<IndexType>::Amd(std::shared_ptr<const Executor> exec,
                    const parameters_type& params)
    : EnablePolymorphicObject<Amd, LinOpFactory>(std::move(exec))
{}


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
    if (auto convertible = dynamic_cast<const ConvertibleTo<complex_mtx>*>(
            system_matrix.get())) {
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
    exec->run(make_amd_reorder(host_exec, static_cast<IndexType>(num_rows),
                               row_ptrs, col_idxs, permutation.get_data()));
    permutation.set_executor(exec);

    return permutation_type::create(exec, dim<2>{num_rows, num_rows},
                                    std::move(permutation));
}


#define GKO_DECLARE_AMD(IndexType) class Amd<IndexType>
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_AMD);


}  // namespace reorder
}  // namespace gko