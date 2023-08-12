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

#ifndef GKO_PUBLIC_EXT_FILE_CONFIG_BASE_HELPER_HPP_
#define GKO_PUBLIC_EXT_FILE_CONFIG_BASE_HELPER_HPP_


#include <iostream>
#include <memory>


#include <nlohmann/json.hpp>


#include "file_config/base/generic_constructor.hpp"
#include "file_config/base/resource_manager.hpp"
#include "file_config/base/types.hpp"


namespace gko {
namespace extensions {
namespace file_config {


/**
 * call is a helper function to decide to use the ResourceManager function or
 * the free function depends on the manager is existed or not.
 *
 * @tparam T  the type
 *
 * @param item  the RapidJson::Value
 * @param exec  the Executor from outside
 * @param linop  the LinOp from outside
 * @param manager  the ResourceManager pointer
 *
 * @note the `build_item` from `ResourceManager` also calls `GenericHelper` free
 *       function in practice, but `build_item` can store the data if it
 *       contains name.
 */
template <typename T>
inline typename std::enable_if<!is_on_linopfactory<T>::value &&
                                   !is_on_criterionfactory<T>::value,
                               std::shared_ptr<T>>::type
call(const nlohmann::json& item, std::shared_ptr<const Executor> exec,
     std::shared_ptr<const LinOp> linop, ResourceManager* manager)
{
    if (manager == nullptr) {
        return Generic<T>::build(item, exec, linop, manager);
    } else {
        std::cout << exec.get() << std::endl;
        return manager->build_item<T>(item, exec, linop);
    }
}

// In dpcpp, GenericHelper static function does not instantiate the
// corresponding Generic function. We use inline function to instantiate all
// possible template
template <typename T>
inline typename std::enable_if<is_on_linopfactory<T>::value ||
                                   is_on_criterionfactory<T>::value,
                               std::shared_ptr<T>>::type
call(const nlohmann::json& item, std::shared_ptr<const Executor> exec,
     std::shared_ptr<const LinOp> linop, ResourceManager* manager)
{
    if (manager == nullptr) {
        return Generic<T, typename T::base_type>::build(item, exec, linop,
                                                        manager);
    } else {
        std::cout << exec.get() << std::endl;
        return manager->build_item<T>(item, exec, linop);
    }
}


}  // namespace file_config
}  // namespace extensions
}  // namespace gko

#endif  // GKO_PUBLIC_EXT_FILE_CONFIG_BASE_HELPER_HPP_
