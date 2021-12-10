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

#include <ginkgo/ginkgo.hpp>


#include <ginkgo/kernels/kernel_launch.hpp>


namespace GKO_DEVICE_NAMESPACE {


using namespace gko::kernels::GKO_DEVICE_NAMESPACE;
using std::cos;
using std::sin;

template <typename T>
struct err {};


void linear_step(std::shared_ptr<const DefaultExecutor> exec, int n,
                 double phase_scale,
                 gko::matrix::Dense<std::complex<double>>* freq)
{
    using device_complex = device_type<std::complex<double>>;
    run_kernel(
        exec,
        GKO_KERNEL(auto i, auto j, auto n, auto phase_scale,
                   auto amplitude_scale, auto freq) {
            auto phase = -(i * i + j * j) * phase_scale;
            freq[i * n + j] *=
                device_complex{cos(phase), sin(phase)} * amplitude_scale;
        },
        gko::dim<2>{n, n}, n, phase_scale, 1.0 / (n * n), freq);
}


void nonlinear_step(std::shared_ptr<const DefaultExecutor> exec, int n,
                    double nonlinear_scale, double potential_scale,
                    double time_scale,
                    const gko::matrix::Dense<double>* potential,
                    gko::matrix::Dense<std::complex<double>>* ampl)
{
    using device_complex = device_type<std::complex<double>>;
    run_kernel(
        exec,
        GKO_KERNEL(auto i, auto j, auto n, auto nonlinear_scale,
                   auto potential_scale, auto time_scale, auto potential,
                   auto ampl) {
            auto idx = i * n + j;
            auto phase = -(nonlinear_scale * gko::squared_norm(ampl[idx]) +
                           potential_scale * potential[idx]) *
                         time_scale;
            ampl[idx] *= device_complex{cos(phase), sin(phase)};
        },
        gko::dim<2>{n, n}, n, nonlinear_scale, potential_scale, time_scale,
        potential, ampl);
}


}  // namespace GKO_DEVICE_NAMESPACE
