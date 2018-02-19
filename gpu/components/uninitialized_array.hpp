/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_GPU_COMPONENTS_UNINITIALIZED_ARRAY_HPP_
#define GKO_GPU_COMPONENTS_UNINITIALIZED_ARRAY_HPP_


#include "core/base/types.hpp"


namespace gko {
namespace kernels {
namespace gpu {


template <typename ValueType, size_type size>
class UninitializedArray {
public:
    constexpr GKO_ATTRIBUTES operator ValueType *() const noexcept
    {
        return reinterpret_cast<const ValueType *>(data_);
    }

    GKO_ATTRIBUTES operator ValueType *() noexcept
    {
        return reinterpret_cast<ValueType *>(data_);
    }

    constexpr GKO_ATTRIBUTES ValueType &operator[](size_type pos) const noexcept
    {
        return reinterpret_cast<const ValueType *>(data_)[pos];
    }

    GKO_ATTRIBUTES ValueType &operator[](size_type pos) noexcept
    {
        return reinterpret_cast<ValueType *>(data_)[pos];
    }

private:
    unsigned char data_[sizeof(ValueType) / sizeof(unsigned char) * size];
};


}  // namespace gpu
}  // namespace kernels
}  // namespace gko


#endif  // GKO_GPU_BASE_COMPONENTS_ARRAY_HPP_
