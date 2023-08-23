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

#include "core/solver/batch_direct_kernels.hpp"


#include "core/matrix/batch_struct.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/dpct.hpp"
#include "dpcpp/base/helper.hpp"
#include "dpcpp/base/onemkl_bindings.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
namespace batch_direct {


#include "dpcpp/solver/batch_direct_kernels.hpp.inc"


void check_exception(onemkl::exception const& e,
                     const std::int64_t scratchpad_size)
{
    int info_code = e.info();
    int detail_code = e.detail();
    if (info_code < 0)
        std::cout << "The " << info_code
                  << "-th parameter had an illegal value." << std::endl;
    else {
        if (info_code == scratchpad_size && detail_code != 0)
            std::cout << "The passed scratchpad size , " << scratchpad_size
                      << ", is insufficient. The size should be at least "
                      << detail_code << "." << std::endl;
        else if (info_code != 0 && detail_code == 0)
            std::cout << "There were some errors for some of the problems in "
                         "the supplied batch. "
                      << info_code << " calculations are failed in each batch."
                      << std::endl;
        else
            std::cout << "Some U_i are exactly singular!" << std::endl;
    }
}


template <typename ValueType>
void apply(std::shared_ptr<const DpcppExecutor> exec,
           matrix::BatchDense<ValueType>* const a_t,
           matrix::BatchDense<ValueType>* const b_t,
           gko::log::BatchLogData<ValueType>& logdata)
{
    const int num_batches = a_t->get_num_batch_entries();
    const int nbatch = static_cast<int>(num_batches);
    const int nrows = a_t->get_size().at()[0];
    const int ncols = a_t->get_size().at()[1];

    const int nrhs = b_t->get_size().at()[1];
    // For getrf_batch and getrs_batch
    const int a_stride = nrows * ncols;
    const int lda = nrows;
    const int b_stride = nrows * nrhs;
    const int ldb = nrows;

    std::int64_t* pivot_array = exec->alloc<std::int64_t>(nbatch * nrows);
    const int pivot_stride = nrows;
    auto a_t_values = a_t->get_values();
    auto b_t_values = b_t->get_values();

    auto queue = exec->get_queue();

    std::int64_t scratchpad_size;
    try {
        scratchpad_size = onemkl::getrf_batch_scratchpad_size<ValueType>(
            *queue, nrows, ncols, lda, a_stride, pivot_stride, nbatch);

        ValueType* const getrf_scratchpad =
            exec->alloc<ValueType>(scratchpad_size);
        onemkl::getrf_batch(*queue, nrows, ncols, a_t_values, lda, a_stride,
                            pivot_array, pivot_stride, nbatch, getrf_scratchpad,
                            scratchpad_size)
            .wait();
        exec->free(getrf_scratchpad);
    } catch (onemkl::exception const& e) {
        std::cout << "Unexpected exception caught during synchronous call to "
                     "LAPACK API - Getrf_batch:\nDetail:"
                  << std::endl;
        check_exception(e, scratchpad_size);
    }

    try {
        scratchpad_size = onemkl::getrs_batch_scratchpad_size<ValueType>(
            *queue, onemkl::nontrans, nrows, nrhs, lda, a_stride, pivot_stride,
            ldb, b_stride, nbatch);

        ValueType* const getrs_scratchpad =
            exec->alloc<ValueType>(scratchpad_size);
        onemkl::getrs_batch(*queue, onemkl::nontrans, nrows, nrhs, a_t_values,
                            lda, a_stride, pivot_array, pivot_stride,
                            b_t_values, ldb, b_stride, nbatch, getrs_scratchpad,
                            scratchpad_size)
            .wait();
        exec->free(getrs_scratchpad);
    } catch (onemkl::exception const& e) {
        std::cout << "Unexpected exception caught during synchronous call to "
                     "LAPACK API - Getrs_batch:\nDetail:"
                  << std::endl;
        check_exception(e, scratchpad_size);
    }
    exec->free(pivot_array);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DIRECT_APPLY_KERNEL);


template <typename ValueType>
void transpose_scale_copy(std::shared_ptr<const DpcppExecutor> exec,
                          const matrix::BatchDiagonal<ValueType>* const scaling,
                          const matrix::BatchDense<ValueType>* const orig,
                          matrix::BatchDense<ValueType>* const scaled)
{
    const size_type nbatch = orig->get_num_batch_entries();
    const int nrows = static_cast<int>(scaled->get_size().at()[0]);
    const int nrhs = static_cast<int>(scaled->get_size().at()[1]);
    const size_type orig_stride = orig->get_stride().at();
    const size_type scaled_stride = scaled->get_stride().at();

    auto queue = exec->get_queue();
    auto group_size =
        queue->get_device().get_info<sycl::info::device::max_work_group_size>();

    const dim3 block(group_size);
    const dim3 grid(nbatch);

    const auto scaling_values = scaling->get_const_values();
    const auto orig_values = orig->get_const_values();
    auto scaled_values = scaled->get_values();

    queue->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                auto batch_id = item_ct1.get_group_linear_id();
                auto scaled_entry = gko::batch::batch_entry_ptr(
                    scaled_values, scaled_stride, nrows, batch_id);
                auto orig_entry = gko::batch::batch_entry_ptr(
                    orig_values, orig_stride, nrhs, batch_id);
                auto scaling_entry = gko::batch::batch_entry_ptr(
                    scaling_values, 1, nrows, batch_id);
                transpose_scale_copy_kernel(nbatch, nrows, nrhs, orig_stride,
                                            scaled_stride, scaling_entry,
                                            orig_entry, scaled_entry, item_ct1);
            });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DIRECT_TRANSPOSE_SCALE_COPY);


template <typename ValueType>
void pre_diag_scale_system_transpose(
    std::shared_ptr<const DpcppExecutor> exec,
    const matrix::BatchDense<ValueType>* const a,
    const matrix::BatchDense<ValueType>* const b,
    const matrix::BatchDiagonal<ValueType>* const left_scale,
    const matrix::BatchDiagonal<ValueType>* const right_scale,
    matrix::BatchDense<ValueType>* const a_scaled_t,
    matrix::BatchDense<ValueType>* const b_scaled_t)
{
    const size_type nbatch = a->get_num_batch_entries();
    const int nrows = static_cast<int>(a->get_size().at()[0]);
    const int ncols = static_cast<int>(a->get_size().at()[1]);
    const int nrhs = static_cast<int>(b->get_size().at()[1]);
    const size_type a_stride = a->get_stride().at();
    const size_type a_scaled_stride = a_scaled_t->get_stride().at();
    const size_type b_stride = b->get_stride().at();
    const size_type b_scaled_stride = b_scaled_t->get_stride().at();
    constexpr size_type left_scale_stride = 1;
    constexpr size_type right_scale_stride = 1;

    auto queue = exec->get_queue();
    auto group_size =
        queue->get_device().get_info<sycl::info::device::max_work_group_size>();

    const dim3 block(group_size);
    const dim3 grid(nbatch);

    const auto a_values = a->get_const_values();
    const auto b_values = b->get_const_values();
    const auto left_scale_values = left_scale->get_const_values();
    const auto right_scale_values = right_scale->get_const_values();
    auto a_scaled_values = a_scaled_t->get_values();
    auto b_scaled_values = b_scaled_t->get_values();

    queue->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                auto batch_id = item_ct1.get_group_linear_id();
                auto a_b = gko::batch::batch_entry_ptr(a_values, a_stride,
                                                       nrows, batch_id);
                auto a_scaled_b = gko::batch::batch_entry_ptr(
                    a_scaled_values, a_scaled_stride, ncols, batch_id);
                auto b_b = gko::batch::batch_entry_ptr(b_values, b_stride,
                                                       nrows, batch_id);
                auto b_scaled_b = gko::batch::batch_entry_ptr(
                    b_scaled_values, b_scaled_stride, nrhs, batch_id);
                auto l_scale_b = gko::batch::batch_entry_ptr(
                    left_scale_values, left_scale_stride, nrows, batch_id);
                auto r_scale_b = gko::batch::batch_entry_ptr(
                    right_scale_values, right_scale_stride, ncols, batch_id);

                pre_diag_scale_system_transpose(
                    nrows, ncols, a_stride, a_b, nrhs, b_stride, b_b, l_scale_b,
                    r_scale_b, a_scaled_stride, a_scaled_b, b_scaled_stride,
                    b_scaled_b, item_ct1);
            });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DIRECT_PRE_DIAG_SCALE_SYSTEM_TRANSPOSE);


}  // namespace batch_direct
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
