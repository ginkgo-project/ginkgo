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

#ifndef GKO_PUBLIC_CORE_BASE_DEVICE_MATRIX_DATA_HPP_
#define GKO_PUBLIC_CORE_BASE_DEVICE_MATRIX_DATA_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/matrix_data.hpp>


namespace gko {


template <typename ValueType = double, typename IndexType = int32>
struct device_matrix_data {
    using nonzero_type = matrix_data_entry<ValueType, IndexType>;
    using host_type = matrix_data<ValueType, IndexType>;

    device_matrix_data(std::shared_ptr<const Executor> exec, dim<2> size = {},
                       size_type nnz = 0);

    device_matrix_data(dim<2> size, Array<nonzero_type> data);

    host_type copy_to_host() const;

    static device_matrix_data create_from_host(
        std::shared_ptr<const Executor> exec, host_type& data);

    void sort_row_major();

    void remove_zeros();

    dim<2> size;
    Array<nonzero_type> nonzeros;
};


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_DEVICE_MATRIX_DATA_HPP_
