// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @see reference/stop/batch_criteria.hpp
 */
template <typename ValueType>
class SimpleRelResidual {
public:
    using real_type = remove_complex<ValueType>;

    __device__ __forceinline__ SimpleRelResidual(
        const real_type rel_res_tol, const real_type* const rhs_b_norms)
        : rel_tol_{rel_res_tol}, rhs_norms_{rhs_b_norms}
    {}

    __device__ __forceinline__ bool check_converged(
        const real_type* const residual_norms) const
    {
        return residual_norms[0] <= (rel_tol_ * rhs_norms_[0]);
    }

private:
    const real_type rel_tol_;
    const real_type* const rhs_norms_;
};


/**
 * @see reference/stop/batch_criteria.hpp
 */
template <typename ValueType>
class SimpleAbsResidual {
public:
    using real_type = remove_complex<ValueType>;

    __device__ __forceinline__ SimpleAbsResidual(const real_type tol,
                                                 const real_type*)
        : abs_tol_{tol}
    {}

    __device__ __forceinline__ bool check_converged(
        const real_type* const residual_norms) const
    {
        return (residual_norms[0] <= abs_tol_);
    }

private:
    const real_type abs_tol_;
};
