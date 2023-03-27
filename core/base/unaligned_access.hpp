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

#ifndef GKO_CORE_BASE_UNALIGNED_ACCESS_HPP_
#define GKO_CORE_BASE_UNALIGNED_ACCESS_HPP_


#include <cstring>
#include <functional>


#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace matrix {
namespace bccoo {


/**
 * Copies the value in the m-th byte of ptr.
 *
 * @tparam T     the type of value
 *
 * @param ptr    the starting pointer
 * @param start  the offset
 * @param value  the value
 */
template <typename T>
void set_value_chunk(void* ptr, size_type start, T value)
{
    std::memcpy(static_cast<unsigned char*>(ptr) + start, &value, sizeof(T));
}


/**
 * Returns the value in the m-th byte of ptr, which is adjusting to T class.
 *
 * @tparam T     the type of value
 *
 * @param ptr    the starting pointer
 * @param start  the offset
 *
 * @return the value in the m-th byte of ptr, which is adjusting to T class.
 */
template <typename T>
T get_value_chunk(const void* ptr, size_type start)
{
    T val{};
    std::memcpy(&val, static_cast<const unsigned char*>(ptr) + start,
                sizeof(T));
    return val;
}


/**
 * Returns the value in the m-th byte of ptr, which is adjusting to T class.
 *
 * @tparam T     the type of value
 *
 * @param ptr_res    the starting pointer of the result
 * @param start_res  the offset of the result
 * @param ptr_src    the starting pointer of the source
 * @param start_src  the offset of the source
 * @param num        the number of values to copy
 *
 * @note The memory does not need to be aligned to be written or read.
 */
template <typename T>
void get_set_value_chunk(void* ptr_res, size_type start_res,
                         const void* ptr_src, size_type start_src,
                         size_type num)
{
    //  return *reinterpret_cast<const T*>
    //    (static_cast<const unsigned char*>(ptr) + start);
    // TODO: Defined behaviour, but might be slower
    T val{};
    // auto value_ptr = reinterpret_cast<unsigned char*>(&value);
    // for (int i = 0; i < sizeof(T); ++i) {
    //     value_ptr[i] = static_cast<const unsigned char*>(ptr)[start + i];
    // }
    memcpy(static_cast<unsigned char*>(ptr_res) + start_res,
           static_cast<const unsigned char*>(ptr_src) + start_src,
           sizeof(T) * num);
}


/**
 * Returns the address in the m-th byte of ptr, which is adjusting to T class.
 *
 * @tparam T     the type of the address
 *
 * @param ptr    the starting pointer
 * @param start  the offset
 *
 * @return the address in the m-th byte of ptr, which is adjusting to T class.
 */
/*
template <typename T>
T* get_address_chunk(const void* ptr, size_type start)
{
    const unsigned char* ptr2 = static_cast<const unsigned char*>(ptr) + start;
    return static_cast<T*>(ptr2);
}
*/


}  // namespace bccoo
}  // namespace matrix
}  // namespace gko


#endif  // GKO_CORE_BASE_UNALIGNED_ACCESS_HPP_
