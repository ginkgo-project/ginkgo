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


#ifndef GKO_PUBLIC_CORE_BASE_CACHE_HPP_
#define GKO_PUBLIC_CORE_BASE_CACHE_HPP_


#include <memory>


#include <ginkgo/core/matrix/dense.hpp>


namespace gko {


/**
 * Manages a Dense vector that is buffered and reused internally to avoid
 * repeated allocations. Copying an instance will only yield an empty object
 * since copying the cached vector would not make sense.
 *
 * @internal  The struct is present to wrap cache-like buffer storage that will
 *            not be copied when the outer object gets copied.
 */
template <typename ValueType>
struct DenseCache {
    DenseCache() = default;
    ~DenseCache() = default;
    DenseCache(const DenseCache &) {}
    DenseCache(DenseCache &&) {}
    DenseCache &operator=(const DenseCache &) { return *this; }
    DenseCache &operator=(DenseCache &&) { return *this; }
    std::unique_ptr<matrix::Dense<ValueType>> vec{};

    void init_from(const matrix::Dense<ValueType> *template_vec)
    {
        if (!vec || vec->get_size() != template_vec->get_size()) {
            vec = matrix::Dense<ValueType>::create_with_config_of(template_vec);
        }
        vec->copy_from(template_vec);
    }

    void init(std::shared_ptr<const Executor> exec, dim<2> size)
    {
        if (!vec || vec->get_size() != size) {
            vec = matrix::Dense<ValueType>::create(exec, size);
        }
    }

    matrix::Dense<ValueType> &operator*() { return *vec; }

    matrix::Dense<ValueType> *operator->() { return vec.get(); }

    matrix::Dense<ValueType> *get() { return vec.get(); }
};


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_CACHE_HPP_
