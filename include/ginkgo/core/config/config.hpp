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

#ifndef GINKGO_CONFIG_HPP
#define GINKGO_CONFIG_HPP


#include <ginkgo/config.hpp>


#include <map>
#include <variant>

#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/config/context.hpp>
#include <ginkgo/core/config/dispatch.hpp>
#include <ginkgo/core/config/property_tree.hpp>
#include <ginkgo/core/config/type_config.hpp>


namespace gko {
namespace config {

using configure_fn = std::function<std::shared_ptr<LinOpFactory>(
    std::shared_ptr<const Executor>, const property_tree&, const context&,
    const type_config&)>;


template <typename T>
configure_fn create_default_configure_fn()
{
    return [=](std::shared_ptr<const Executor> exec, const property_tree& pt,
               const context& ctx, const type_config& cfg) {
        return T::configure(std::move(exec), pt, ctx, cfg);
    };
}


template <template <class...> class T, typename... Variants>
configure_fn create_default_configure_fn(Variants&&... vs)
{
    return [=](std::shared_ptr<const Executor> exec, const property_tree& pt,
               const context& ctx, const type_config& cfg) {
        return dispatch<T>(std::move(exec), pt, ctx, cfg, vs...);
    };
}

std::shared_ptr<LinOpFactory> parse(
    std::shared_ptr<const Executor> exec, const property_tree& pt,
    const context& ctx, const type_config& tcfg = default_type_config());


template <typename ValueType = default_precision, typename IndexType = int32,
          typename GlobalIndexType = int64>
std::shared_ptr<LinOpFactory> parse(std::shared_ptr<const Executor> exec,
                                    const property_tree& pt, const context& ctx)
{
    auto type_config =
        encode_type_config(pt, {ValueType{}, IndexType{}, GlobalIndexType{}});
    return parse(std::move(exec), pt, ctx, type_config);
}


}  // namespace config
}  // namespace gko


#endif  // GINKGO_CONFIG_HPP
