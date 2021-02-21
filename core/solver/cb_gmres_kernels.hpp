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

#ifndef GKO_CORE_SOLVER_CB_GMRES_KERNELS_HPP_
#define GKO_CORE_SOLVER_CB_GMRES_KERNELS_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/range.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/accessors.hpp"
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
#define GKO_INSTANTIATE_FOR_EACH_CB_GMRES_TYPE_HELPER(_macro, _const)          \
    template _macro(                                                           \
        double,                                                                \
        GKO_UNPACK(                                                            \
            range<accessor::reduced_row_major<3, double, _const double>>));    \
    template _macro(                                                           \
        double,                                                                \
        GKO_UNPACK(                                                            \
            range<accessor::reduced_row_major<3, double, _const float>>));     \
    template _macro(                                                           \
        double,                                                                \
        GKO_UNPACK(                                                            \
            range<accessor::reduced_row_major<3, double, _const half>>));      \
    template _macro(double,                                                    \
                    GKO_UNPACK(range<accessor::scaled_reduced_row_major<       \
                                   3, double, _const int64, 0b101>>));         \
    template _macro(double,                                                    \
                    GKO_UNPACK(range<accessor::scaled_reduced_row_major<       \
                                   3, double, _const int32, 0b101>>));         \
    template _macro(double,                                                    \
                    GKO_UNPACK(range<accessor::scaled_reduced_row_major<       \
                                   3, double, _const int16, 0b101>>));         \
    template _macro(                                                           \
        float,                                                                 \
        GKO_UNPACK(                                                            \
            range<accessor::reduced_row_major<3, float, _const float>>));      \
    template _macro(                                                           \
        float,                                                                 \
        GKO_UNPACK(                                                            \
            range<accessor::reduced_row_major<3, float, _const half>>));       \
    template _macro(float,                                                     \
                    GKO_UNPACK(range<accessor::scaled_reduced_row_major<       \
                                   3, float, _const int32, 0b101>>));          \
    template _macro(float,                                                     \
                    GKO_UNPACK(range<accessor::scaled_reduced_row_major<       \
                                   3, float, _const int16, 0b101>>));          \
    template _macro(                                                           \
        std::complex<double>,                                                  \
        GKO_UNPACK(                                                            \
            range<accessor::reduced_row_major<3, std::complex<double>,         \
                                              _const std::complex<double>>>)); \
    template _macro(                                                           \
        std::complex<double>,                                                  \
        GKO_UNPACK(                                                            \
            range<accessor::reduced_row_major<3, std::complex<double>,         \
                                              _const std::complex<float>>>));  \
    template _macro(                                                           \
        std::complex<float>,                                                   \
        GKO_UNPACK(                                                            \
            range<accessor::reduced_row_major<3, std::complex<float>,          \
                                              _const std::complex<float>>>))

#define GKO_INSTANTIATE_FOR_EACH_CB_GMRES_TYPE(_macro) \
    GKO_INSTANTIATE_FOR_EACH_CB_GMRES_TYPE_HELPER(_macro, )

#define GKO_INSTANTIATE_FOR_EACH_CB_GMRES_CONST_TYPE(_macro) \
    GKO_INSTANTIATE_FOR_EACH_CB_GMRES_TYPE_HELPER(_macro, const)


namespace gko {
namespace kernels {


#define GKO_DECLARE_CB_GMRES_INITIALIZE_1_KERNEL(_type)                     \
    void initialize_1(                                                      \
        std::shared_ptr<const DefaultExecutor> exec,                        \
        const matrix::Dense<_type> *b, matrix::Dense<_type> *residual,      \
        matrix::Dense<_type> *givens_sin, matrix::Dense<_type> *givens_cos, \
        Array<stopping_status> *stop_status, size_type krylov_dim)


#define GKO_DECLARE_CB_GMRES_INITIALIZE_2_KERNEL(_type1, _range)            \
    void initialize_2(std::shared_ptr<const DefaultExecutor> exec,          \
                      const matrix::Dense<_type1> *residual,                \
                      matrix::Dense<remove_complex<_type1>> *residual_norm, \
                      matrix::Dense<_type1> *residual_norm_collection,      \
                      matrix::Dense<remove_complex<_type1>> *arnoldi_norm,  \
                      _range krylov_bases,                                  \
                      matrix::Dense<_type1> *next_krylov_basis,             \
                      Array<size_type> *final_iter_nums, size_type krylov_dim)


#define GKO_DECLARE_CB_GMRES_STEP_1_KERNEL(_type1, _range)                    \
    void step_1(                                                              \
        std::shared_ptr<const DefaultExecutor> exec,                          \
        matrix::Dense<_type1> *next_krylov_basis,                             \
        matrix::Dense<_type1> *givens_sin, matrix::Dense<_type1> *givens_cos, \
        matrix::Dense<remove_complex<_type1>> *residual_norm,                 \
        matrix::Dense<_type1> *residual_norm_collection, _range krylov_bases, \
        matrix::Dense<_type1> *hessenberg_iter,                               \
        matrix::Dense<_type1> *buffer_iter,                                   \
        matrix::Dense<remove_complex<_type1>> *arnoldi_norm, size_type iter,  \
        Array<size_type> *final_iter_nums,                                    \
        const Array<stopping_status> *stop_status,                            \
        Array<stopping_status> *reorth_status, Array<size_type> *num_reorth)

#define GKO_DECLARE_CB_GMRES_STEP_2_KERNEL(_type1, _range)                    \
    void step_2(std::shared_ptr<const DefaultExecutor> exec,                  \
                const matrix::Dense<_type1> *residual_norm_collection,        \
                _range krylov_bases, const matrix::Dense<_type1> *hessenberg, \
                matrix::Dense<_type1> *y,                                     \
                matrix::Dense<_type1> *before_preconditioner,                 \
                const Array<size_type> *final_iter_nums)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                 \
    template <typename ValueType>                                    \
    GKO_DECLARE_CB_GMRES_INITIALIZE_1_KERNEL(ValueType);             \
    template <typename ValueType, typename Accessor3d>               \
    GKO_DECLARE_CB_GMRES_INITIALIZE_2_KERNEL(ValueType, Accessor3d); \
    template <typename ValueType, typename Accessor3d>               \
    GKO_DECLARE_CB_GMRES_STEP_1_KERNEL(ValueType, Accessor3d);       \
    template <typename ValueType, typename Accessor3d>               \
    GKO_DECLARE_CB_GMRES_STEP_2_KERNEL(ValueType, Accessor3d)


namespace omp {
namespace cb_gmres {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace cb_gmres
}  // namespace omp


namespace cuda {
namespace cb_gmres {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace cb_gmres
}  // namespace cuda


namespace reference {
namespace cb_gmres {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace cb_gmres
}  // namespace reference


namespace hip {
namespace cb_gmres {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace cb_gmres
}  // namespace hip


namespace dpcpp {
namespace cb_gmres {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace cb_gmres
}  // namespace dpcpp


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_CB_GMRES_KERNELS_HPP_
