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

#ifndef GKO_PUBLIC_CORE_STOP_BATCH_STOP_ENUM_HPP_
#define GKO_PUBLIC_CORE_STOP_BATCH_STOP_ENUM_HPP_


namespace gko {
namespace batch {
namespace stop {


/**
 * This enum provides two types of options for the convergence of an iterative
 * solver.
 *
 * `absolute` tolerance implies that the convergence criteria check is
 * against the computed residual ($||r|| \leq \tau$)
 *
 * With the `relative` tolerance type, the solver
 * convergence criteria checks against the relative residual norm
 * ($||r|| \leq ||b|| \times \tau$, where $||b||$ is the L2 norm of the rhs).
 *
 * @note the computed residual norm, $||r||$ may be implicit or explicit
 * depending on the solver algorithm.
 */
enum class tolerance_type { absolute, relative };


}  // namespace stop
}  // namespace batch
}  // namespace gko

#endif  // GKO_PUBLIC_CORE_STOP_BATCH_STOP_ENUM_HPP_
