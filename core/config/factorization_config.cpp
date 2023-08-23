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

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/factorization/cholesky.hpp>
#include <ginkgo/core/factorization/ic.hpp>
#include <ginkgo/core/factorization/ilu.hpp>
#include <ginkgo/core/factorization/lu.hpp>
#include <ginkgo/core/factorization/par_ic.hpp>
#include <ginkgo/core/factorization/par_ict.hpp>
#include <ginkgo/core/factorization/par_ilu.hpp>
#include <ginkgo/core/factorization/par_ilut.hpp>


#include "core/config/config.hpp"
#include "core/config/dispatch.hpp"


namespace gko {
namespace config {


template <typename ValueType, typename IndexType>
class IcConfigurator {
public:
    static std::unique_ptr<
        typename factorization::Ic<ValueType, IndexType>::Factory>
    build_from_config(const pnode& config, const registry& context,
                      std::shared_ptr<const Executor> exec,
                      type_descriptor td_for_child)
    {
        using matrix_type =
            typename factorization::Ic<ValueType, IndexType>::matrix_type;
        auto factory = factorization::Ic<ValueType, IndexType>::build();
        SET_CSR_STRATEGY(factory, matrix_type, l_strategy, config, context,
                         exec, td_for_child);
        SET_VALUE(factory, bool, skip_sorting, config);
        SET_VALUE(factory, bool, both_factors, config);
        return factory.on(exec);
    }
};


template <>
std::unique_ptr<gko::LinOpFactory>
build_from_config<LinOpFactoryType::Factorization_Ic>(
    const pnode& config, const registry& context,
    std::shared_ptr<const Executor>& exec, gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, IcConfigurator>(
        updated.first + "," + updated.second, config, context, exec, updated,
        value_type_list(), index_type_list());
}


template <typename ValueType, typename IndexType>
class IluConfigurator {
public:
    static std::unique_ptr<
        typename factorization::Ilu<ValueType, IndexType>::Factory>
    build_from_config(const pnode& config, const registry& context,
                      std::shared_ptr<const Executor> exec,
                      type_descriptor td_for_child)
    {
        using matrix_type =
            typename factorization::Ilu<ValueType, IndexType>::matrix_type;
        auto factory = factorization::Ilu<ValueType, IndexType>::build();
        SET_CSR_STRATEGY(factory, matrix_type, l_strategy, config, context,
                         exec, td_for_child);
        SET_CSR_STRATEGY(factory, matrix_type, u_strategy, config, context,
                         exec, td_for_child);
        SET_VALUE(factory, bool, skip_sorting, config);
        return factory.on(exec);
    }
};


template <>
std::unique_ptr<gko::LinOpFactory>
build_from_config<LinOpFactoryType::Factorization_Ilu>(
    const pnode& config, const registry& context,
    std::shared_ptr<const Executor>& exec, gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, IluConfigurator>(
        updated.first + "," + updated.second, config, context, exec, updated,
        value_type_list(), index_type_list());
}


// Cholesky is already Factory, which generates Factorization LinOp
template <typename ValueType, typename IndexType>
class CholeskyConfigurator {
public:
    static std::unique_ptr<
        typename experimental::factorization::Cholesky<ValueType, IndexType>>
    build_from_config(const pnode& config, const registry& context,
                      std::shared_ptr<const Executor> exec,
                      type_descriptor td_for_child)
    {
        using sparsity_pattern_type =
            typename experimental::factorization::Cholesky<
                ValueType, IndexType>::sparsity_pattern_type;
        auto factory =
            experimental::factorization::Cholesky<ValueType,
                                                  IndexType>::build();
        SET_POINTER(factory, const sparsity_pattern_type,
                    symbolic_factorization, config, context, exec,
                    td_for_child);
        SET_VALUE(factory, bool, skip_sorting, config);
        return factory.on(exec);
    }
};


template <>
std::unique_ptr<gko::LinOpFactory>
build_from_config<LinOpFactoryType::Cholesky>(
    const pnode& config, const registry& context,
    std::shared_ptr<const Executor>& exec, gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, CholeskyConfigurator>(
        updated.first + "," + updated.second, config, context, exec, updated,
        value_type_list(), index_type_list());
}


// Lu is already Factory, which generates Factorization LinOp
template <typename ValueType, typename IndexType>
class LuConfigurator {
public:
    static std::unique_ptr<
        experimental::factorization::Lu<ValueType, IndexType>>
    build_from_config(const pnode& config, const registry& context,
                      std::shared_ptr<const Executor> exec,
                      type_descriptor td_for_child)
    {
        using sparsity_pattern_type = typename experimental::factorization::Lu<
            ValueType, IndexType>::sparsity_pattern_type;
        auto factory =
            experimental::factorization::Lu<ValueType, IndexType>::build();
        SET_POINTER(factory, const sparsity_pattern_type,
                    symbolic_factorization, config, context, exec,
                    td_for_child);
        SET_VALUE(factory, bool, symmetric_sparsity, config);
        SET_VALUE(factory, bool, skip_sorting, config);
        return factory.on(exec);
    }
};


template <>
std::unique_ptr<gko::LinOpFactory> build_from_config<LinOpFactoryType::Lu>(
    const pnode& config, const registry& context,
    std::shared_ptr<const Executor>& exec, gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, LuConfigurator>(
        updated.first + "," + updated.second, config, context, exec, updated,
        value_type_list(), index_type_list());
}


template <typename ValueType, typename IndexType>
class ParIluConfigurator {
public:
    static std::unique_ptr<
        typename factorization::ParIlu<ValueType, IndexType>::Factory>
    build_from_config(const pnode& config, const registry& context,
                      std::shared_ptr<const Executor> exec,
                      type_descriptor td_for_child)
    {
        using matrix_type =
            typename factorization::ParIlu<ValueType, IndexType>::matrix_type;
        auto factory = factorization::ParIlu<ValueType, IndexType>::build();
        SET_VALUE(factory, size_type, iterations, config);
        SET_VALUE(factory, bool, skip_sorting, config);
        SET_CSR_STRATEGY(factory, matrix_type, l_strategy, config, context,
                         exec, td_for_child);
        SET_CSR_STRATEGY(factory, matrix_type, u_strategy, config, context,
                         exec, td_for_child);
        return factory.on(exec);
    }
};


template <>
std::unique_ptr<gko::LinOpFactory> build_from_config<LinOpFactoryType::ParIlu>(
    const pnode& config, const registry& context,
    std::shared_ptr<const Executor>& exec, gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, ParIluConfigurator>(
        updated.first + "," + updated.second, config, context, exec, updated,
        value_type_list(), index_type_list());
}


template <typename ValueType, typename IndexType>
class ParIlutConfigurator {
public:
    static std::unique_ptr<
        typename factorization::ParIlut<ValueType, IndexType>::Factory>
    build_from_config(const pnode& config, const registry& context,
                      std::shared_ptr<const Executor> exec,
                      type_descriptor td_for_child)
    {
        using matrix_type =
            typename factorization::ParIlut<ValueType, IndexType>::matrix_type;
        auto factory = factorization::ParIlut<ValueType, IndexType>::build();
        SET_VALUE(factory, size_type, iterations, config);
        SET_VALUE(factory, bool, skip_sorting, config);
        SET_VALUE(factory, bool, approximate_select, config);
        SET_VALUE(factory, bool, deterministic_sample, config);
        SET_VALUE(factory, double, fill_in_limit, config);
        SET_CSR_STRATEGY(factory, matrix_type, l_strategy, config, context,
                         exec, td_for_child);
        SET_CSR_STRATEGY(factory, matrix_type, u_strategy, config, context,
                         exec, td_for_child);
        return factory.on(exec);
    }
};


template <>
std::unique_ptr<gko::LinOpFactory> build_from_config<LinOpFactoryType::ParIlut>(
    const pnode& config, const registry& context,
    std::shared_ptr<const Executor>& exec, gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, ParIlutConfigurator>(
        updated.first + "," + updated.second, config, context, exec, updated,
        value_type_list(), index_type_list());
}


template <typename ValueType, typename IndexType>
class ParIcConfigurator {
public:
    static std::unique_ptr<
        typename factorization::ParIc<ValueType, IndexType>::Factory>
    build_from_config(const pnode& config, const registry& context,
                      std::shared_ptr<const Executor> exec,
                      type_descriptor td_for_child)
    {
        using matrix_type =
            typename factorization::ParIc<ValueType, IndexType>::matrix_type;
        auto factory = factorization::ParIc<ValueType, IndexType>::build();
        SET_VALUE(factory, size_type, iterations, config);
        SET_VALUE(factory, bool, skip_sorting, config);
        SET_CSR_STRATEGY(factory, matrix_type, l_strategy, config, context,
                         exec, td_for_child);
        SET_VALUE(factory, bool, both_factors, config);
        return factory.on(exec);
    }
};


template <>
std::unique_ptr<gko::LinOpFactory> build_from_config<LinOpFactoryType::ParIc>(
    const pnode& config, const registry& context,
    std::shared_ptr<const Executor>& exec, gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, ParIcConfigurator>(
        updated.first + "," + updated.second, config, context, exec, updated,
        value_type_list(), index_type_list());
}


template <typename ValueType, typename IndexType>
class ParIctConfigurator {
public:
    static std::unique_ptr<
        typename factorization::ParIct<ValueType, IndexType>::Factory>
    build_from_config(const pnode& config, const registry& context,
                      std::shared_ptr<const Executor> exec,
                      type_descriptor td_for_child)
    {
        using matrix_type =
            typename factorization::ParIct<ValueType, IndexType>::matrix_type;
        auto factory = factorization::ParIct<ValueType, IndexType>::build();
        SET_VALUE(factory, size_type, iterations, config);
        SET_VALUE(factory, bool, skip_sorting, config);
        SET_VALUE(factory, bool, approximate_select, config);
        SET_VALUE(factory, bool, deterministic_sample, config);
        SET_VALUE(factory, double, fill_in_limit, config);
        SET_CSR_STRATEGY(factory, matrix_type, l_strategy, config, context,
                         exec, td_for_child);
        SET_CSR_STRATEGY(factory, matrix_type, lt_strategy, config, context,
                         exec, td_for_child);
        return factory.on(exec);
    }
};


template <>
std::unique_ptr<gko::LinOpFactory> build_from_config<LinOpFactoryType::ParIct>(
    const pnode& config, const registry& context,
    std::shared_ptr<const Executor>& exec, gko::config::type_descriptor td)
{
    auto updated = update_type(config, td);
    return dispatch<gko::LinOpFactory, ParIctConfigurator>(
        updated.first + "," + updated.second, config, context, exec, updated,
        value_type_list(), index_type_list());
}


}  // namespace config
}  // namespace gko
