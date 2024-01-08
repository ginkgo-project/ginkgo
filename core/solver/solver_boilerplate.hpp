// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <type_traits>


#define GKO_SOLVER_TRAITS \
    ::gko::solver::workspace_traits<::std::decay_t<decltype(*this)>>


#define GKO_SOLVER_VECTOR(_x, _template)                                      \
    auto _x = this->create_workspace_op_with_config_of(GKO_SOLVER_TRAITS::_x, \
                                                       _template)


#define GKO_SOLVER_SCALAR(_x, _template)                         \
    auto _x = this->template create_workspace_scalar<ValueType>( \
        GKO_SOLVER_TRAITS::_x, _template->get_size()[1])


#define GKO_SOLVER_ONE_MINUS_ONE()                                       \
    auto one_op = this->template create_workspace_scalar<ValueType>(     \
        GKO_SOLVER_TRAITS::one, 1);                                      \
    auto neg_one_op = this->template create_workspace_scalar<ValueType>( \
        GKO_SOLVER_TRAITS::minus_one, 1);                                \
    one_op->fill(one<ValueType>());                                      \
    neg_one_op->fill(-one<ValueType>())

#define GKO_SOLVER_STOP_REDUCTION_ARRAYS()                      \
    auto& stop_status =                                         \
        this->template create_workspace_array<stopping_status>( \
            GKO_SOLVER_TRAITS::stop, dense_b->get_size()[1]);   \
    auto& reduction_tmp =                                       \
        this->template create_workspace_array<char>(GKO_SOLVER_TRAITS::tmp)
