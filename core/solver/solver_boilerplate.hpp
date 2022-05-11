/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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


#define GKO_SOLVER_VECTOR(x) \
    auto x = this->create_workspace_with_config_of(ws::x, dense_b)


#define GKO_SOLVER_SCALAR(x)                                    \
    auto x = this->template create_workspace_scalar<ValueType>( \
        ws::x, dense_b->get_size()[1])


#define GKO_SOLVER_ONE_MINUS_ONE()                                           \
    auto one_op =                                                            \
        this->template create_workspace_scalar<ValueType>(ws::one, 1);       \
    auto neg_one_op =                                                        \
        this->template create_workspace_scalar<ValueType>(ws::minus_one, 1); \
    one_op->fill(one<ValueType>());                                          \
    neg_one_op->fill(-one<ValueType>())

#define GKO_SOLVER_STOP_REDUCTION_ARRAYS()                      \
    auto& stop_status =                                         \
        this->template create_workspace_array<stopping_status>( \
            ws::stop, dense_b->get_size()[1]);                  \
    auto& reduction_tmp =                                       \
        this->template create_workspace_array<char>(ws::tmp, 0)
