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


/**
 * Copies value of type T starting at byte dspl from ptr.
 *
 * @tparam T     the type of value
 *
 * @param ptr    the starting pointer
 * @param dspl   the offset (in bytes)
 * @param value  the value
 *
 * @note The memory does not need to be aligned to be read.
 */
template <typename T>
__device__ void set_value_compressed_data(void* ptr, size_type dspl, T value)
{
    memcpy(static_cast<unsigned char*>(ptr) + dspl, &value, sizeof(T));
}


/**
 * Copies value of type T at pos-th component of array<T> starting
 *        at byte dspl from ptr.
 *
 * @tparam T     the type of value
 *
 * @param ptr    the starting pointer
 * @param dspl   the offset (in bytes)
 * @param pos    the component position of the vector
 * @param value  the value
 */
template <typename T>
__device__ void set_value_compressed_data(void* ptr, size_type dspl,
                                          size_type pos, T value)
{
    set_value_compressed_data<T>(ptr, dspl + pos * sizeof(T), value);
}


/**
 * Returns the value of type T starting at byte dspl from ptr.
 *
 * @tparam T     the type of value
 *
 * @param ptr    the starting pointer
 * @param dspl   the offset (in bytes)
 *
 * @return the value of type T starting at byte dspl from ptr.
 *
 * @note The memory does not need to be aligned to be written.
 */
template <typename T>
__device__ T get_value_compressed_data(const void* ptr, size_type dspl)
{
    T value;
    memcpy(&value, static_cast<const unsigned char*>(ptr) + dspl, sizeof(T));
    return value;
}


/**
 * Returns the value of type T at pos-th component of array<T> starting
           at byte dspl from ptr.
 *
 * @tparam T     the type of value
 *
 * @param ptr    the starting pointer
 * @param dspl   the offset (in bytes)
 * @param pos    the component position of the vector
 *
 * @return the value of type T at pos-th component of array<T> starting
 *         at byte dspl from ptr.
 *
 * @note The memory does not need to be aligned to be written or read.
 */
template <typename T>
__device__ T get_value_compressed_data(const void* ptr, size_type dspl,
                                       size_type pos)
{
    return get_value_compressed_data<T>(ptr, dspl + pos * sizeof(T));
}


/**
 * Copies num values of type T
 *    from array<T> starting at byte dspl_src from ptr_src
 *    to array<T> starting at byte dspl_res from ptr_res.
 *
 * @tparam T     the type of value
 *
 * @param ptr_res    the starting pointer of the result
 * @param dspl_res   the offset of the result (in bytes)
 * @param ptr_src    the starting pointer of the source
 * @param dspl_src   the offset of the source (in bytes)
 * @param num        the number of values to copy
 *
 * @note The memory does not need to be aligned to be written or read.
 */
template <typename T>
__device__ void copy_array_compressed_data(void* ptr_res, size_type dspl_res,
                                           const void* ptr_src,
                                           size_type dspl_src, size_type num)
{
    memcpy(static_cast<unsigned char*>(ptr_res) + dspl_res,
           static_cast<const unsigned char*>(ptr_src) + dspl_src,
           sizeof(T) * num);
}
