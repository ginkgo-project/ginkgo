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

#include "core/solver/batch_band_solver_kernels.hpp"


#include "core/matrix/batch_struct.hpp"
#include "reference/matrix/batch_struct.hpp"

namespace gko {
namespace kernels {
namespace reference {
namespace batch_band_solver {

namespace {

// include all depedencies (note: do not remove this comment)
#include "reference/matrix/batch_dense_kernels.hpp.inc"
#include "reference/solver/batch_band_solver_kernels.hpp.inc"


}  // namespace


template <typename ValueType>
void apply(std::shared_ptr<const DefaultExecutor> exec,
           const matrix::BatchBand<ValueType>* const band_mat,
           const matrix::BatchDense<ValueType>* const b,
           matrix::BatchDense<ValueType>* const x, const int workspace_size,
           ValueType* const workspace_ptr,
           const enum gko::solver::batch_band_solve_approach approach,
           const int blocked_solve_panel_size)
{
    const auto nbatch = band_mat->get_num_batch_entries();
    const auto nrows = static_cast<int>(band_mat->get_size().at(0)[0]);
    const auto nrhs = static_cast<int>(b->get_size().at(0)[1]);
    assert(nrhs == 1);

    namespace device = gko::kernels::host;
    const auto x_batch = device::get_batch_struct(x);
    const auto b_batch = device::get_batch_struct(b);

    assert(workspace_size >= band_mat->get_num_stored_elements());

    if (workspace_size < band_mat->get_num_stored_elements()) {
        std::cout << " file: " << __FILE__ << " line: " << __LINE__
                  << " workspace size is not enough" << std::endl;
        exit(0);
    }

    ValueType* const batch_band_mat_array = workspace_ptr;
    exec->copy(band_mat->get_num_stored_elements(),
               band_mat->get_const_band_array(), batch_band_mat_array);

    const int KL = static_cast<int>(band_mat->get_num_subdiagonals().at(0));
    const int KU = static_cast<int>(band_mat->get_num_superdiagonals().at(0));

    for (size_type ibatch = 0; ibatch < nbatch; ibatch++) {
        if (approach == gko::solver::batch_band_solve_approach::unblocked ||
            (approach == gko::solver::batch_band_solve_approach::blocked &&
             blocked_solve_panel_size > KL)) {
            const int local_size_bytes =
                gko::kernels::batch_band_solver::local_memory_requirement<
                    ValueType>(
                    nrows, nrhs,
                    gko::solver::batch_band_solve_approach::unblocked);

            std::vector<unsigned char> local_space(local_size_bytes);

            batch_entry_band_unblocked_solve_impl(ibatch, nbatch, KL, KU,
                                                  batch_band_mat_array, b_batch,
                                                  x_batch, local_space.data());

        } else if (approach ==
                   gko::solver::batch_band_solve_approach::blocked) {
            const int local_size_bytes =
                gko::kernels::batch_band_solver::local_memory_requirement<
                    ValueType>(nrows, nrhs,
                               gko::solver::batch_band_solve_approach::blocked,
                               blocked_solve_panel_size);

            std::vector<unsigned char> local_space(local_size_bytes);

            batch_entry_band_blocked_solve_impl(
                ibatch, nbatch, KL, KU, batch_band_mat_array,
                blocked_solve_panel_size, b_batch, x_batch, local_space.data());
        } else {
            GKO_NOT_IMPLEMENTED;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_BAND_SOLVER_APPLY_KERNEL);


}  // namespace batch_band_solver
}  // namespace reference
}  // namespace kernels
}  // namespace gko
