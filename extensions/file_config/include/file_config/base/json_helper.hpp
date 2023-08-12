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

#ifndef GKO_PUBLIC_EXT_FILE_CONFIG_BASE_JSON_HELPER_HPP_
#define GKO_PUBLIC_EXT_FILE_CONFIG_BASE_JSON_HELPER_HPP_


#include <iostream>
#include <memory>


#include <nlohmann/json.hpp>


#include "file_config/base/generic_constructor.hpp"
#include "file_config/base/macro_helper.hpp"
#include "file_config/base/resource_manager.hpp"


namespace gko {
namespace extensions {
namespace file_config {


template <typename T>
inline T get_value(const nlohmann::json& item, const char* key)
{
    return item.at(key).get<T>();
}

template <>
inline dim<2> get_value<dim<2>>(const nlohmann::json& item, const char* key)
{
    if (item.at(key).is_array()) {
        auto array = item.at(key);
        if (array.size() == 2) {
            return dim<2>(array[0].get<gko::int64>(),
                          array[1].get<gko::int64>());
        } else if (array.size() == 1) {
            return dim<2>(array[0].get<gko::int64>(),
                          array[0].get<gko::int64>());
        } else {
            assert(false);
            // avoid the warning about return type
            return dim<2>();
        }
    } else if (item.at(key).is_number()) {
        return dim<2>(item.at(key).get<gko::int64>(),
                      item.at(key).get<gko::int64>());
    } else {
        assert(false);
        // avoid the warning about return type
        return dim<2>();
    }
}


template <typename T>
inline T get_value_with_default(const nlohmann::json& item, const char* key,
                                T default_value)
{
    if (item.contains(key)) {
        return get_value<T>(item, key);
    } else {
        return default_value;
    }
}


template <typename T>
T get_required_value(const nlohmann::json& item, const char* key)
{
    if (!item.contains(key)) {
        std::cerr << "the value of key " << key << " must not be empty"
                  << std::endl;
        assert(false);
        return T{};
    } else {
        return get_value<T>(item, key);
    }
}


// It can not use the overload get_value because mask_type is uint.
inline gko::log::Logger::mask_type get_mask_value_with_default(
    const nlohmann::json& item, std::string key,
    gko::log::Logger::mask_type default_val)
{
    // clang-format off
#define GKORM_LOGGER_EVENT(_event_mask) \
    {#_event_mask, gko::log::Logger::_event_mask}
    // clang-format on

    // Note: before using the following command, clang ColumnLimit should use
    // 160 or large enough value. Otherwise, some like criterion_check_completed
    // will not be caught, so need to manually add them.
    // clang-format off
    // For single:
    // grep -E 'GKO_LOGGER_REGISTER_EVENT\([0-9]+, [^,]*,' include/ginkgo/core/log/logger.hpp | sed -E 's/.*GKO_LOGGER_REGISTER_EVENT\([0-9]+, ([^,]*),.*/GKORM_LOGGER_EVENT(\1_mask),/g'
    // For combine:
    // grep -E 'mask_type +[^ ]*_mask +=' include/ginkgo/core/log/logger.hpp | sed -E 's/.*mask_type +([^ ]*_mask).*/GKORM_LOGGER_EVENT(\1),/g'
    // clang-format on
    static std::unordered_map<std::string, gko::log::Logger::mask_type>
        mask_type_map{
            GKORM_LOGGER_EVENT(allocation_started_mask),
            GKORM_LOGGER_EVENT(allocation_completed_mask),
            GKORM_LOGGER_EVENT(free_started_mask),
            GKORM_LOGGER_EVENT(free_completed_mask),
            GKORM_LOGGER_EVENT(copy_started_mask),
            GKORM_LOGGER_EVENT(copy_completed_mask),
            GKORM_LOGGER_EVENT(operation_launched_mask),
            GKORM_LOGGER_EVENT(operation_completed_mask),
            GKORM_LOGGER_EVENT(polymorphic_object_create_started_mask),
            GKORM_LOGGER_EVENT(polymorphic_object_create_completed_mask),
            GKORM_LOGGER_EVENT(polymorphic_object_copy_started_mask),
            GKORM_LOGGER_EVENT(polymorphic_object_copy_completed_mask),
            GKORM_LOGGER_EVENT(polymorphic_object_deleted_mask),
            GKORM_LOGGER_EVENT(linop_apply_started_mask),
            GKORM_LOGGER_EVENT(linop_apply_completed_mask),
            GKORM_LOGGER_EVENT(linop_advanced_apply_started_mask),
            GKORM_LOGGER_EVENT(linop_advanced_apply_completed_mask),
            GKORM_LOGGER_EVENT(linop_factory_generate_started_mask),
            GKORM_LOGGER_EVENT(linop_factory_generate_completed_mask),
            GKORM_LOGGER_EVENT(criterion_check_started_mask),
            GKORM_LOGGER_EVENT(criterion_check_completed_mask),
            GKORM_LOGGER_EVENT(iteration_complete_mask),
            GKORM_LOGGER_EVENT(polymorphic_object_move_started_mask),
            GKORM_LOGGER_EVENT(polymorphic_object_move_completed_mask),
            GKORM_LOGGER_EVENT(all_events_mask),
            GKORM_LOGGER_EVENT(executor_events_mask),
            GKORM_LOGGER_EVENT(operation_events_mask),
            GKORM_LOGGER_EVENT(polymorphic_object_events_mask),
            GKORM_LOGGER_EVENT(linop_events_mask),
            GKORM_LOGGER_EVENT(linop_factory_events_mask),
            GKORM_LOGGER_EVENT(criterion_events_mask)};


    gko::log::Logger::mask_type mask_value = 0;
    if (item.contains(key)) {
        auto& mask_item = item.at(key);
        if (mask_item.is_string()) {
            mask_value |= mask_type_map.at(mask_item.get<std::string>());
        } else if (mask_item.is_array()) {
            for (auto& it : mask_item) {
                mask_value |= mask_type_map.at(it.get<std::string>());
            }
        } else {
            assert(false);
        }
    } else {
        mask_value = default_val;
    }

    return mask_value;
}


/**
 * get_pointer gives the shared_ptr<const type> from inputs.
 *
 * @tparam T  the type
 *
 * @param item  the RapidJson::Value
 * @param exec  the Executor from outside
 * @param linop  the LinOp from outside
 * @param manager  the ResourceManager pointer
 *
 * @return std::shared_ptr<const T>
 */
template <typename T>
inline std::shared_ptr<T> get_pointer(const nlohmann::json& item,
                                      std::shared_ptr<const gko::Executor> exec,
                                      std::shared_ptr<const LinOp> linop,
                                      ResourceManager* manager)
{
    std::shared_ptr<T> ptr;
    using T_non_const = std::remove_const_t<T>;
    if (manager == nullptr) {
        if (item.is_object()) {
            ptr = GenericHelper<T_non_const>::build(item, exec, linop, manager);
        } else {
            assert(false);
        }
    } else {
        if (item.is_string()) {
            std::cout << "search item" << std::endl;
            std::string opt = item.get<std::string>();
            ptr = manager->search_data<T_non_const>(opt);
            std::cout << "get ptr " << ptr.get() << std::endl;
        } else if (item.is_object()) {
            ptr = manager->build_item<T_non_const>(item, exec, linop);
        } else {
            assert(false);
        }
    }
    assert(ptr.get() != nullptr);
    return std::move(ptr);
}

template <>
inline std::shared_ptr<const Executor> get_pointer<const Executor>(
    const nlohmann::json& item, std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const LinOp> linop, ResourceManager* manager)
{
    std::shared_ptr<const Executor> ptr;
    if (manager == nullptr) {
        if (item.is_object()) {
            ptr = GenericHelper<Executor>::build(item, exec, linop, manager);
        } else if (item.is_string() &&
                   item.get<std::string>() == std::string("inherit")) {
            ptr = exec;
        } else {
            assert(false);
        }
    } else {
        // assert(false);
        // TODO: manager
        if (item.is_string()) {
            std::string opt = item.get<std::string>();
            if (opt == std::string("inherit")) {
                ptr = exec;
            } else {
                ptr = manager->search_data<Executor>(opt);
            }
        } else if (item.is_object()) {
            ptr = manager->build_item<Executor>(item);
        } else {
            assert(false);
        }
    }
    assert(ptr.get() != nullptr);
    return std::move(ptr);
}

template <>
inline std::shared_ptr<const LinOp> get_pointer<const LinOp>(
    const nlohmann::json& item, std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const LinOp> linop, ResourceManager* manager)
{
    std::shared_ptr<const LinOp> ptr;
    if (manager == nullptr) {
        if (item.is_object()) {
            ptr = GenericHelper<LinOp>::build(item, exec, linop, manager);
        } else if (item.is_string() &&
                   item.get<std::string>() == std::string("given")) {
            ptr = linop;
        } else {
            assert(false);
        }
    } else {
        if (item.is_string()) {
            std::string opt = item.get<std::string>();
            if (opt == std::string("given")) {
                ptr = linop;
            } else {
                ptr = manager->search_data<LinOp>(opt);
            }
        } else if (item.is_object()) {
            ptr = manager->build_item<LinOp>(item);
        } else {
            assert(false);
        }
    }
    assert(ptr.get() != nullptr);
    return std::move(ptr);
}


/**
 * get_pointer_check considers existence of the key to decide the behavior.
 *
 * @tparam T  the type
 *
 * @param item  the RapidJson::Value
 * @param key  the key string
 * @param exec  the Executor from outside
 * @param linop  the LinOp from outside
 * @param manager  the ResourceManager pointer
 *
 * @return std::shared_ptr<const T>
 */
template <typename T>
inline std::shared_ptr<const T> get_pointer_check(
    const nlohmann::json& item, std::string key,
    std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const LinOp> linop, ResourceManager* manager)
{
    assert(item.contains(key));
    return get_pointer<const T>(item.at(key), exec, linop, manager);
}

template <>
inline std::shared_ptr<const Executor> get_pointer_check<const Executor>(
    const nlohmann::json& item, std::string key,
    std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const LinOp> linop, ResourceManager* manager)
{
    if (item.contains(key)) {
        return get_pointer<const Executor>(item.at(key), exec, linop, manager);
    } else if (exec != nullptr) {
        return exec;
    } else {
        assert(false);
        return nullptr;
    }
}


template <typename Type>
inline void add_logger(Type& obj, const nlohmann::json& item,
                       std::shared_ptr<const Executor> exec,
                       std::shared_ptr<const LinOp> linop,
                       ResourceManager* manager)
{
    if (item.contains("add_loggers")) {
        auto& logger = item.at("add_loggers");
        if (logger.is_array()) {
            for (auto& it : logger) {
                obj->add_logger(get_pointer<Logger>(it, exec, linop, manager));
            }
        } else {
            obj->add_logger(get_pointer<Logger>(logger, exec, linop, manager));
        }
    }
}


template <class Csr>
inline std::shared_ptr<typename Csr::strategy_type> get_csr_strategy(
    const std::string& strategy, std::shared_ptr<const Executor> exec_ptr)
{
    std::shared_ptr<typename Csr::strategy_type> strategy_ptr;
    if (strategy == std::string("sparselib") ||
        strategy == std::string("cusparse")) {
        strategy_ptr = std::make_shared<typename Csr::sparselib>();
    } else if (strategy == std::string("automatical")) {
        if (auto explicit_exec =
                std::dynamic_pointer_cast<const gko::CudaExecutor>(exec_ptr)) {
            strategy_ptr =
                std::make_shared<typename Csr::automatical>(explicit_exec);
        } else if (auto explicit_exec =
                       std::dynamic_pointer_cast<const gko::HipExecutor>(
                           exec_ptr)) {
            strategy_ptr =
                std::make_shared<typename Csr::automatical>(explicit_exec);
        } else if (auto explicit_exec =
                       std::dynamic_pointer_cast<const gko::DpcppExecutor>(
                           exec_ptr)) {
            strategy_ptr =
                std::make_shared<typename Csr::automatical>(explicit_exec);
        } else {
            // fallback to classical
            strategy_ptr = std::make_shared<typename Csr::classical>();
        }
    } else if (strategy == std::string("load_balance")) {
        if (auto explicit_exec =
                std::dynamic_pointer_cast<const gko::CudaExecutor>(exec_ptr)) {
            strategy_ptr =
                std::make_shared<typename Csr::load_balance>(explicit_exec);
        } else if (auto explicit_exec =
                       std::dynamic_pointer_cast<const gko::HipExecutor>(
                           exec_ptr)) {
            strategy_ptr =
                std::make_shared<typename Csr::load_balance>(explicit_exec);
        } else if (auto explicit_exec =
                       std::dynamic_pointer_cast<const gko::DpcppExecutor>(
                           exec_ptr)) {
            strategy_ptr =
                std::make_shared<typename Csr::load_balance>(explicit_exec);
        } else {
            // fallback to classical
            strategy_ptr = std::make_shared<typename Csr::classical>();
        }

    } else if (strategy == std::string("merge_path")) {
        strategy_ptr = std::make_shared<typename Csr::merge_path>();
    } else if (strategy == std::string("classical")) {
        strategy_ptr = std::make_shared<typename Csr::classical>();
    }
    return std::move(strategy_ptr);
}


}  // namespace file_config
}  // namespace extensions
}  // namespace gko
#endif  // GKO_PUBLIC_EXT_FILE_CONFIG_BASE_JSON_HELPER_HPP_
