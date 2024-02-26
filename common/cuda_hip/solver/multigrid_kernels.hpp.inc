// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

namespace kernel {


// grid_nrows is the number of rows handled in the whole grid at the same time.
// Thus, the threads whose index is larger than grid_nrows * nrhs are not used.
// Let the thread handle the same col (has same scalar) in whole loop.
template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void kcycle_step_1_kernel(
    const size_type num_rows, const size_type nrhs, const size_type stride,
    const size_type grid_nrows, const ValueType* __restrict__ alpha,
    const ValueType* __restrict__ rho, const ValueType* __restrict__ v,
    ValueType* __restrict__ g, ValueType* __restrict__ d,
    ValueType* __restrict__ e)
{
    const auto tidx = thread::get_thread_id_flat();
    const auto col = tidx % nrhs;
    const auto num_elems = grid_nrows * nrhs;
    if (tidx >= num_elems) {
        return;
    }
    const auto total_elems = num_rows * stride;
    const auto grid_stride = grid_nrows * stride;
    const auto temp = alpha[col] / rho[col];
    const bool update = is_finite(temp);
    for (auto idx = tidx / nrhs * stride + col; idx < total_elems;
         idx += grid_stride) {
        auto store_e = e[idx];
        if (update) {
            g[idx] -= temp * v[idx];
            store_e *= temp;
            e[idx] = store_e;
        }
        d[idx] = store_e;
    }
}


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void kcycle_step_2_kernel(
    const size_type num_rows, const size_type nrhs, const size_type stride,
    const size_type grid_nrows, const ValueType* __restrict__ alpha,
    const ValueType* __restrict__ rho, const ValueType* __restrict__ gamma,
    const ValueType* __restrict__ beta, const ValueType* __restrict__ zeta,
    const ValueType* __restrict__ d, ValueType* __restrict__ e)
{
    const auto tidx = thread::get_thread_id_flat();
    const auto col = tidx % nrhs;
    const auto num_elems = grid_nrows * nrhs;
    if (tidx >= num_elems) {
        return;
    }
    const auto total_elems = num_rows * stride;
    const auto grid_stride = grid_nrows * stride;
    const auto scalar_d =
        zeta[col] / (beta[col] - gamma[col] * gamma[col] / rho[col]);
    const auto scalar_e = one<ValueType>() - gamma[col] / alpha[col] * scalar_d;
    if (is_finite(scalar_d) && is_finite(scalar_e)) {
        for (auto idx = tidx / nrhs * stride + col; idx < total_elems;
             idx += grid_stride) {
            e[idx] = scalar_e * e[idx] + scalar_d * d[idx];
        }
    }
}


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void kcycle_check_stop_kernel(
    const size_type nrhs, const ValueType* __restrict__ old_norm,
    const ValueType* __restrict__ new_norm, const ValueType rel_tol,
    bool* __restrict__ is_stop)
{
    auto tidx = thread::get_thread_id_flat();
    if (tidx >= nrhs) {
        return;
    }
    if (new_norm[tidx] > rel_tol * old_norm[tidx]) {
        *is_stop = false;
    }
}


}  // namespace kernel


template <typename ValueType>
void kcycle_step_1(std::shared_ptr<const DefaultExecutor> exec,
                   const matrix::Dense<ValueType>* alpha,
                   const matrix::Dense<ValueType>* rho,
                   const matrix::Dense<ValueType>* v,
                   matrix::Dense<ValueType>* g, matrix::Dense<ValueType>* d,
                   matrix::Dense<ValueType>* e)
{
    const auto nrows = e->get_size()[0];
    const auto nrhs = e->get_size()[1];
    constexpr int max_size = (1U << 31) - 1;
    const size_type grid_nrows =
        max_size / nrhs < nrows ? max_size / nrhs : nrows;
    const auto grid = ceildiv(grid_nrows * nrhs, default_block_size);
    if (grid > 0) {
        kernel::kcycle_step_1_kernel<<<grid, default_block_size, 0,
                                       exec->get_stream()>>>(
            nrows, nrhs, e->get_stride(), grid_nrows,
            as_device_type(alpha->get_const_values()),
            as_device_type(rho->get_const_values()),
            as_device_type(v->get_const_values()),
            as_device_type(g->get_values()), as_device_type(d->get_values()),
            as_device_type(e->get_values()));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MULTIGRID_KCYCLE_STEP_1_KERNEL);


template <typename ValueType>
void kcycle_step_2(std::shared_ptr<const DefaultExecutor> exec,
                   const matrix::Dense<ValueType>* alpha,
                   const matrix::Dense<ValueType>* rho,
                   const matrix::Dense<ValueType>* gamma,
                   const matrix::Dense<ValueType>* beta,
                   const matrix::Dense<ValueType>* zeta,
                   const matrix::Dense<ValueType>* d,
                   matrix::Dense<ValueType>* e)
{
    const auto nrows = e->get_size()[0];
    const auto nrhs = e->get_size()[1];
    constexpr int max_size = (1U << 31) - 1;
    const size_type grid_nrows =
        max_size / nrhs < nrows ? max_size / nrhs : nrows;
    const auto grid = ceildiv(grid_nrows * nrhs, default_block_size);
    if (grid > 0) {
        kernel::kcycle_step_2_kernel<<<grid, default_block_size, 0,
                                       exec->get_stream()>>>(
            nrows, nrhs, e->get_stride(), grid_nrows,
            as_device_type(alpha->get_const_values()),
            as_device_type(rho->get_const_values()),
            as_device_type(gamma->get_const_values()),
            as_device_type(beta->get_const_values()),
            as_device_type(zeta->get_const_values()),
            as_device_type(d->get_const_values()),
            as_device_type(e->get_values()));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MULTIGRID_KCYCLE_STEP_2_KERNEL);


template <typename ValueType>
void kcycle_check_stop(std::shared_ptr<const DefaultExecutor> exec,
                       const matrix::Dense<ValueType>* old_norm,
                       const matrix::Dense<ValueType>* new_norm,
                       const ValueType rel_tol, bool& is_stop)
{
    gko::array<bool> dis_stop(exec, 1);
    components::fill_array(exec, dis_stop.get_data(), dis_stop.get_size(),
                           true);
    const auto nrhs = new_norm->get_size()[1];
    const auto grid = ceildiv(nrhs, default_block_size);
    if (grid > 0) {
        kernel::kcycle_check_stop_kernel<<<grid, default_block_size, 0,
                                           exec->get_stream()>>>(
            nrhs, as_device_type(old_norm->get_const_values()),
            as_device_type(new_norm->get_const_values()), rel_tol,
            as_device_type(dis_stop.get_data()));
    }
    is_stop = get_element(dis_stop, 0);
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(
    GKO_DECLARE_MULTIGRID_KCYCLE_CHECK_STOP_KERNEL);
