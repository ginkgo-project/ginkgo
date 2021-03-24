/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#ifndef GKO_CORE_SOLVER_IDR_KERNELS_HPP_
#define GKO_CORE_SOLVER_IDR_KERNELS_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>


#include "accessor/reduced_row_major.hpp"
#include "accessor/scaled_reduced_row_major.hpp"
#include "core/base/extended_float.hpp"


// TODO Find way around using it!
#define GKO_UNPACK(...) __VA_ARGS__
/**
 * Instantiates a template for each value type with each lower precision type
 * supported by Ginkgo for CbGmres.
 *
 * @param _macro  A macro which expands the template instantiation
 *                (not including the leading `template` specifier).
 *                Should take two arguments:
 *                1. the first will be used as the regular ValueType
 *                     (precisions supported by Ginkgo), and
 *                2. the second the value type of the reduced precision.
 * @param _const  qualifier used for the storage type, indicating if it is a
 *                const accessor, or not.
 */
#define GKO_INSTANTIATE_FOR_EACH_IDR_TYPE_HELPER(_macro, _const)               \
    template _macro(                                                           \
        double,                                                                \
        GKO_UNPACK(                                                            \
            acc::range<acc::reduced_row_major<2, double, _const double>>));    \
    template _macro(                                                           \
        double,                                                                \
        GKO_UNPACK(                                                            \
            acc::range<acc::reduced_row_major<2, double, _const float>>));     \
    template _macro(                                                           \
        double,                                                                \
        GKO_UNPACK(                                                            \
            acc::range<acc::reduced_row_major<2, double, _const half>>));      \
    template _macro(                                                           \
        float,                                                                 \
        GKO_UNPACK(                                                            \
            acc::range<acc::reduced_row_major<2, float, _const float>>));      \
    template _macro(                                                           \
        float,                                                                 \
        GKO_UNPACK(                                                            \
            acc::range<acc::reduced_row_major<2, float, _const half>>));       \
    template _macro(                                                           \
        std::complex<double>,                                                  \
        GKO_UNPACK(                                                            \
            acc::range<acc::reduced_row_major<2, std::complex<double>,         \
                                              _const std::complex<double>>>)); \
    template _macro(                                                           \
        std::complex<double>,                                                  \
        GKO_UNPACK(                                                            \
            acc::range<acc::reduced_row_major<2, std::complex<double>,         \
                                              _const std::complex<float>>>));  \
    template _macro(                                                           \
        std::complex<double>,                                                  \
        GKO_UNPACK(                                                            \
            acc::range<acc::reduced_row_major<2, std::complex<double>,         \
                                              _const std::complex<half>>>));   \
    template _macro(                                                           \
        std::complex<float>,                                                   \
        GKO_UNPACK(                                                            \
            acc::range<acc::reduced_row_major<2, std::complex<float>,          \
                                              _const std::complex<float>>>));  \
    template _macro(                                                           \
        std::complex<float>,                                                   \
        GKO_UNPACK(                                                            \
            acc::range<acc::reduced_row_major<2, std::complex<float>,          \
                                              _const std::complex<half>>>))

#define GKO_INSTANTIATE_FOR_EACH_IDR_TYPE(_macro) \
    GKO_INSTANTIATE_FOR_EACH_IDR_TYPE_HELPER(_macro, )

#define GKO_INSTANTIATE_FOR_EACH_IDR_CONST_TYPE(_macro) \
    GKO_INSTANTIATE_FOR_EACH_IDR_TYPE_HELPER(_macro, const)


namespace gko {
namespace kernels {
namespace idr {


#define GKO_DECLARE_IDR_INITIALIZE_KERNEL(_type, _range)           \
    void initialize(std::shared_ptr<const DefaultExecutor> exec,   \
                    const size_type nrhs, matrix::Dense<_type> *m, \
                    _range subspace_vectors, bool deterministic,   \
                    Array<stopping_status> *stop_status)


#define GKO_DECLARE_IDR_APPLY_SUBSPACE_KERNEL(_type, _range)                  \
    void apply_subspace(                                                      \
        std::shared_ptr<const DefaultExecutor> exec, _range subspace_vectors, \
        const matrix::Dense<_type> *residual, matrix::Dense<_type> *f)


#define GKO_DECLARE_IDR_STEP_1_KERNEL(_type)                                 \
    void step_1(                                                             \
        std::shared_ptr<const DefaultExecutor> exec, const size_type nrhs,   \
        const size_type k, const matrix::Dense<_type> *m,                    \
        const matrix::Dense<_type> *f, const matrix::Dense<_type> *residual, \
        const matrix::Dense<_type> *g, matrix::Dense<_type> *c,              \
        matrix::Dense<_type> *v, const Array<stopping_status> *stop_status)


#define GKO_DECLARE_IDR_STEP_2_KERNEL(_type)                            \
    void step_2(std::shared_ptr<const DefaultExecutor> exec,            \
                const size_type nrhs, const size_type k,                \
                const matrix::Dense<_type> *omega,                      \
                const matrix::Dense<_type> *preconditioned_vector,      \
                const matrix::Dense<_type> *c, matrix::Dense<_type> *u, \
                const Array<stopping_status> *stop_status)


#define GKO_DECLARE_IDR_STEP_3_KERNEL(_type, _range)                     \
    void step_3(std::shared_ptr<const DefaultExecutor> exec,             \
                const size_type nrhs, const size_type k, _range p,       \
                matrix::Dense<_type> *g, matrix::Dense<_type> *g_k,      \
                matrix::Dense<_type> *u, matrix::Dense<_type> *m,        \
                matrix::Dense<_type> *f, matrix::Dense<_type> *alpha,    \
                matrix::Dense<_type> *residual, matrix::Dense<_type> *x, \
                const Array<stopping_status> *stop_status)


#define GKO_DECLARE_IDR_COMPUTE_OMEGA_KERNEL(_type)                         \
    void compute_omega(                                                     \
        std::shared_ptr<const DefaultExecutor> exec, const size_type nrhs,  \
        const remove_complex<_type> kappa, const matrix::Dense<_type> *tht, \
        const matrix::Dense<remove_complex<_type>> *residual_norm,          \
        matrix::Dense<_type> *omega,                                        \
        const Array<stopping_status> *stop_status)


#define GKO_DECLARE_IDR_COMPUTE_GAMMA_KERNEL(_type)                           \
    void compute_gamma(std::shared_ptr<const DefaultExecutor> exec,           \
                       const size_type nrhs, const matrix::Dense<_type> *tht, \
                       matrix::Dense<_type> *gamma,                           \
                       matrix::Dense<_type> *one_minus_gamma,                 \
                       const Array<stopping_status> *stop_status)


#define GKO_DECLARE_ALL_AS_TEMPLATES                       \
    template <typename ValueType, typename Acc>            \
    GKO_DECLARE_IDR_INITIALIZE_KERNEL(ValueType, Acc);     \
    template <typename ValueType, typename Acc>            \
    GKO_DECLARE_IDR_APPLY_SUBSPACE_KERNEL(ValueType, Acc); \
    template <typename ValueType>                          \
    GKO_DECLARE_IDR_STEP_1_KERNEL(ValueType);              \
    template <typename ValueType>                          \
    GKO_DECLARE_IDR_STEP_2_KERNEL(ValueType);              \
    template <typename ValueType, typename Acc>            \
    GKO_DECLARE_IDR_STEP_3_KERNEL(ValueType, Acc);         \
    template <typename ValueType>                          \
    GKO_DECLARE_IDR_COMPUTE_OMEGA_KERNEL(ValueType);       \
    template <typename ValueType>                          \
    GKO_DECLARE_IDR_COMPUTE_GAMMA_KERNEL(ValueType)

}  // namespace idr


namespace omp {
namespace idr {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace idr
}  // namespace omp


namespace cuda {
namespace idr {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace idr
}  // namespace cuda


namespace reference {
namespace idr {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace idr
}  // namespace reference


namespace hip {
namespace idr {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace idr
}  // namespace hip


namespace dpcpp {
namespace idr {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace idr
}  // namespace dpcpp


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_IDR_KERNELS_HPP_
