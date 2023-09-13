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

#include "core/solver/batch_idr_kernels.hpp"


#include <random>


// #include <curand_kernel.h>

#include <ginkgo/core/base/math.hpp>


#include "core/matrix/batch_struct.hpp"
#include "core/solver/batch_dispatch.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/dpct.hpp"
#include "dpcpp/base/helper.hpp"
// #include "dpcpp/base/onemkl_bindings.hpp"
#include "dpcpp/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {

/**
 * @brief The batch Idr solver namespace.
 *
 * @ingroup batch_idr
 */
namespace batch_idr {


#include "dpcpp/matrix/batch_csr_kernels.hpp.inc"
#include "dpcpp/matrix/batch_dense_kernels.hpp.inc"
#include "dpcpp/matrix/batch_ell_kernels.hpp.inc"
#include "dpcpp/matrix/batch_vector_kernels.hpp.inc"
#include "dpcpp/solver/batch_idr_kernels.hpp.inc"


template <typename T>
using BatchIdrOptions = gko::kernels::batch_idr::BatchIdrOptions<T>;

template <typename ValueType>
class KernelCaller {
public:
    using value_type = ValueType;

    KernelCaller(std::shared_ptr<const DpcppExecutor> exec,
                 const BatchIdrOptions<remove_complex<value_type>> opts,
                 const value_type* const subspace_vectors)
        : exec_{exec}, opts_{opts}, subspace_vectors_{subspace_vectors}
    {}

    template <typename BatchMatrixType, typename PrecType, typename StopType,
              typename LogType>
    void call_kernel(LogType logger, const BatchMatrixType& a, PrecType prec,
                     const gko::batch_dense::UniformBatch<const value_type>& b,
                     const gko::batch_dense::UniformBatch<value_type>& x) const
    {
        using real_type = gko::remove_complex<value_type>;
        const size_type nbatch = a.num_batch;
        const auto nrows = a.num_rows;
        const auto nrhs = b.num_rhs;

        auto device = exec_->get_queue()->get_device();
        auto group_size =
            device.get_info<sycl::info::device::max_work_group_size>();
        constexpr int subgroup_size = config::warp_size;
        GKO_ASSERT(group_size >= 2 * subgroup_size);
        size_type slm_size =
            device.get_info<sycl::info::device::local_mem_size>();
        size_type shmem_per_blk =
            slm_size - 4 * sizeof(real_type) -
            4 * sizeof(value_type);  // for shared-norms and intermediate data
        const dim3 block(group_size);
        const dim3 grid(nbatch);

        const int shared_size =
            gko::kernels::batch_idr::local_memory_requirement<value_type>(
                a.num_rows, b.num_rhs, opts_.subspace_dim_val) +
            PrecType::dynamic_work_size(a.num_rows, a.num_nnz) *
                sizeof(ValueType);
        GKO_ASSERT(shared_size <= shmem_per_blk);

        const auto slm_values_size =
            (shared_size - opts_.subspace_dim_val * sizeof(real_type)) /
            sizeof(ValueType);

        auto b_values = b.values;
        auto x_values = x.values;
        auto opts = opts_;
        auto subspace_vectors = subspace_vectors_;

        (exec_->get_queue())->submit([&](sycl::handler& cgh) {
            sycl::accessor<ValueType, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                slm_values(sycl::range<1>(slm_values_size), cgh);
            sycl::accessor<real_type, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                slm_reals(sycl::range<1>(opts.subspace_dim_val), cgh);

            cgh.parallel_for(
                sycl_nd_range(grid, block),
                [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                    subgroup_size)]] {
                    auto group = item_ct1.get_group();
                    auto batch_id = group.get_group_linear_id();
                    const auto a_global_entry =
                        gko::batch::batch_entry(a, batch_id);
                    const ValueType* const b_global_entry =
                        gko::batch::batch_entry_ptr(b_values, 1, nrows,
                                                    batch_id);
                    ValueType* const x_global_entry =
                        gko::batch::batch_entry_ptr(x_values, 1, nrows,
                                                    batch_id);
                    ValueType* const slm_values_ptr = slm_values.get_pointer();
                    real_type* const slm_reals_ptr = slm_reals.get_pointer();
                    apply_kernel<StopType>(
                        opts.max_its, opts.residual_tol, opts.subspace_dim_val,
                        opts.kappa_val, opts.to_use_smoothing,
                        opts.deterministic_gen, logger, prec, subspace_vectors,
                        a_global_entry, b_global_entry, x_global_entry, nrows,
                        a.num_nnz, slm_values_ptr, slm_reals_ptr, item_ct1);
                });
        });
    }

private:
    std::shared_ptr<const DpcppExecutor> exec_;
    const BatchIdrOptions<remove_complex<value_type>> opts_;
    const value_type* const subspace_vectors_;
};

namespace {

template <typename ValueType, typename Distribution, typename Generator>
typename std::enable_if<!is_complex_s<ValueType>::value, ValueType>::type
get_rand_value(Distribution&& dist, Generator&& gen)
{
    return dist(gen);
}


template <typename ValueType, typename Distribution, typename Generator>
typename std::enable_if<is_complex_s<ValueType>::value, ValueType>::type
get_rand_value(Distribution&& dist, Generator&& gen)
{
    return ValueType(dist(gen), dist(gen));
}

}  // unnamed namespace


template <typename ValueType>
void apply(std::shared_ptr<const DpcppExecutor> exec,
           const BatchIdrOptions<remove_complex<ValueType>>& opts,
           const BatchLinOp* const a, const BatchLinOp* const prec,
           const matrix::BatchDense<ValueType>* const b,
           matrix::BatchDense<ValueType>* const x,
           log::BatchLogData<ValueType>& logdata)
{
    if (opts.is_complex_subspace == true && !is_complex<ValueType>()) {
        GKO_NOT_IMPLEMENTED;
    }
    const gko::batch_dense::UniformBatch<ValueType> x_b = get_batch_struct(x);
    array<ValueType> arr(exec->get_master());
    if (opts.deterministic_gen) {
        arr.resize_and_reset(x_b.num_rows * opts.subspace_dim_val);
        auto dist =
            std::normal_distribution<remove_complex<ValueType>>(0.0, 1.0);
        const auto seed = 15;
        auto gen = std::ranlux48(seed);
        // WARNING: The same ranlux48 object MUST be used for all entries of
        //  the array or the IDR does not work for complex problems!
        for (int vec_index = 0; vec_index < opts.subspace_dim_val;
             vec_index++) {
            for (int row_index = 0; row_index < x_b.num_rows; row_index++) {
                ValueType val = get_rand_value<ValueType>(dist, gen);
                arr.get_data()[vec_index * x_b.num_rows + row_index] = val;
            }
        }
        arr.set_executor(exec);
    }
    const ValueType* const subspace_vectors_entry =
        opts.deterministic_gen ? arr.get_const_data() : nullptr;

    auto dispatcher = batch_solver::create_dispatcher<ValueType>(
        KernelCaller<ValueType>(exec, opts, subspace_vectors_entry), opts, a,
        prec);
    dispatcher.apply(b, x, logdata);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_IDR_APPLY_KERNEL);


}  // namespace batch_idr
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
