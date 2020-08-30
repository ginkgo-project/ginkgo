/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include "benchmark_wrappers.hpp"


#include <ginkgo/core/base/executor.hpp>


#include <Kokkos_Core.hpp>
// separator to avoid clang-format reordering the includes :)
#include <KokkosSparse_spgemm.hpp>


#include "core/matrix/csr_builder.hpp"

namespace gko {


template <typename ValueType>
KokkosCsr<ValueType>::KokkosCsr(std::shared_ptr<const gko::Executor> exec,
                                const gko::dim<2> &size)
    : gko::EnableLinOp<KokkosCsr<ValueType>>(exec, size),
      csr_(gko::share(
          csr::create(exec, std::make_shared<typename csr::classical>())))
{
    auto cuda_exec = as<CudaExecutor>(exec);
    Kokkos::InitArguments args{};
    args.device_id = cuda_exec->get_device_id();
    Kokkos::initialize(args);
}


template <typename ValueType>
KokkosCsr<ValueType>::~KokkosCsr()
{
    Kokkos::finalize();
}


template <typename ValueType>
void KokkosCsr<ValueType>::apply_impl(const LinOp *b, LinOp *x) const
{
    auto cuda_exec = as<CudaExecutor>(this->get_executor());

    auto m = this->get_size()[0];
    auto k = b->get_size()[0];
    auto n = b->get_size()[1];

    auto x_csr = as<csr>(x);
    auto b_csr = as<csr>(b);
    auto this_row_ptrs = this->csr_->get_const_row_ptrs();
    auto this_col_idxs = this->csr_->get_const_col_idxs();
    auto this_vals = this->csr_->get_const_values();
    auto this_nnz = this->csr_->get_num_stored_elements();
    auto b_row_ptrs = b_csr->get_const_row_ptrs();
    auto b_col_idxs = b_csr->get_const_col_idxs();
    auto b_vals = b_csr->get_const_values();
    auto b_nnz = b_csr->get_num_stored_elements();
    auto x_row_ptrs = x_csr->get_row_ptrs();

    using exec_space = Kokkos::Cuda::execution_space;
    using mem_space = Kokkos::Cuda::memory_space;

    KokkosKernels::Experimental::KokkosKernelsHandle<
        int32, int32, ValueType, exec_space, mem_space, mem_space>
        handle;
    handle.create_spgemm_handle(KokkosSparse::SPGEMM_KK);

    using int_view = Kokkos::View<const int32 *, mem_space,
                                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using val_view = Kokkos::View<const ValueType *, mem_space,
                                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using int_out_view = Kokkos::View<int32 *, mem_space,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using val_out_view = Kokkos::View<ValueType *, mem_space,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    int_view this_row_view{this_row_ptrs, m + 1};
    int_view this_col_view{this_col_idxs, this_nnz};
    val_view this_val_view{this_vals, this_nnz};
    int_view b_row_view{b_row_ptrs, k + 1};
    int_view b_col_view{b_col_idxs, b_nnz};
    val_view b_val_view{b_vals, b_nnz};
    int_out_view x_row_view{x_row_ptrs, m + 1};

    this->get_executor()->run(
        "symbolic", [] {},
        [&] {
            KokkosSparse::Experimental::spgemm_symbolic(
                &handle, m, k, n, this_row_view, this_col_view, false,
                b_row_view, b_col_view, false, x_row_view);
        },
        [] {});

    auto x_nnz =
        static_cast<size_type>(handle.get_spgemm_handle()->get_c_nnz());
    matrix::CsrBuilder<ValueType, int32> x_builder{x_csr};
    x_builder.get_col_idx_array().resize_and_reset(x_nnz);
    x_builder.get_value_array().resize_and_reset(x_nnz);

    int_out_view x_col_view{x_csr->get_col_idxs(), x_nnz};
    val_out_view x_val_view{x_csr->get_values(), x_nnz};

    this->get_executor()->run(
        "numeric", [] {},
        [&] {
            KokkosSparse::Experimental::spgemm_numeric(
                &handle, m, k, n, this_row_view, this_col_view, this_val_view,
                false, b_row_view, b_col_view, b_val_view, false, x_row_view,
                x_col_view, x_val_view);
        },
        [] {});

    handle.destroy_spgemm_handle();
}


template class KokkosCsr<float>;
template class KokkosCsr<double>;

}  // namespace gko
