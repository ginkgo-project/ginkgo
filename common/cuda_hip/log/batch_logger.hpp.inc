// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @see reference/log/batch_logger.hpp
 */
template <typename RealType>
class SimpleFinalLogger final {
public:
    using real_type = RealType;

    SimpleFinalLogger(real_type* const batch_residuals, int* const batch_iters)
        : final_residuals_{batch_residuals}, final_iters_{batch_iters}
    {}

    __device__ __forceinline__ void log_iteration(const size_type batch_idx,
                                                  const int iter,
                                                  const real_type res_norm)
    {
        final_iters_[batch_idx] = iter;
        final_residuals_[batch_idx] = res_norm;
    }

private:
    real_type* const final_residuals_;
    int* const final_iters_;
};
