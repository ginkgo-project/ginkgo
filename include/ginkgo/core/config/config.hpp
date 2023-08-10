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

#ifndef GKO_PUBLIC_CORE_CONFIG_CONFIG_HPP_
#define GKO_PUBLIC_CORE_CONFIG_CONFIG_HPP_


#include <map>
#include <string>
#include <unordered_map>


#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/config/registry.hpp>


namespace gko {
namespace config {


enum LinOpFactoryType : int { Cg = 0 };


// It is only an intermediate step. If we do not provide the SolverType with VT,
// IT selection, we do not need it in public.
template <int flag>
std::unique_ptr<gko::LinOpFactory> build_from_config(
    const gko::config::Config& config, const gko::config::registry& context,
    std::shared_ptr<const Executor>& exec);

// The main function
std::unique_ptr<gko::LinOpFactory> build_from_config(
    const gko::config::Config& config, const gko::config::registry& context,
    std::shared_ptr<const Executor>& exec);


BuildFromConfigMap generate_config_map();


}  // namespace config
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_CONFIG_CONFIG_HPP_
