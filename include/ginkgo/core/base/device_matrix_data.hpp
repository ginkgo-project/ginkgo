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

#ifndef GKO_PUBLIC_CORE_BASE_DEVICE_MATRIX_DATA_HPP_
#define GKO_PUBLIC_CORE_BASE_DEVICE_MATRIX_DATA_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/matrix_data.hpp>


namespace gko {


/**
 * This type is a device-side equivalent to matrix_data.
 * It stores the data necessary to initialize any matrix format in Ginkgo in a
 * array of matrix_data_entry values together with associated matrix dimensions.
 *
 * @note To be used with a Ginkgo matrix type, the entry array must be sorted in
 *       row-major order, i.e. by row index, then by column index within rows.
 *       This can be achieved by calling the sort_row_major function.
 * @note The data must not contain any duplicate (row, column) pairs.
 *
 * @tparam ValueType  the type used to store matrix values
 * @tparam IndexType  the type used to store matrix row and column indices
 */
template <typename ValueType, typename IndexType>
struct device_matrix_data {
    using nonzero_type = matrix_data_entry<ValueType, IndexType>;
    using host_type = matrix_data<ValueType, IndexType>;

    /**
     * Initializes a new device_matrix_data object.
     * It uses the given executor to allocate storage for the given number of
     * entries and matrix dimensions.
     *
     * @param exec  the executor to be used to store the matrix entries
     * @param size  the matrix dimensions
     * @param num_entries  the number of entries to be stored
     */
    device_matrix_data(std::shared_ptr<const Executor> exec, dim<2> size = {},
                       size_type num_entries = 0);

    /**
     * Initializes a new device_matrix_data object from existing data.
     *
     * @param size  the matrix dimensions
     * @param data  the array containing the matrix entries
     */
    device_matrix_data(dim<2> size, Array<nonzero_type> data);

    /**
     * Copies the device_matrix_data entries to the host to return a regular
     * matrix_data object with the same dimensions and entries.
     *
     * @return a matrix_data object with the same dimensions and entries.
     */
    host_type copy_to_host() const;

    /**
     * Creates a view of the given host data on the given executor.
     *
     * @param exec  the executor to create the device_matrix_data on.
     * @param data  the data to be wrapped or copied into a device_matrix_data.
     * @return  a device_matrix_data object with the same size as `data` and the
     *          same entries, either wrapped as a non-owning view if `exec` is a
     *          host executor or copied into an owning Array if `exec` is a
     *          device executor.
     */
    static device_matrix_data create_view_from_host(
        std::shared_ptr<const Executor> exec, host_type& data);

    /**
     * Sorts the matrix entries in row-major order
     * This means that they will be sorted by row index first, and then by
     * column index inside each row.
     */
    void sort_row_major();

    /**
     * Removes all zero entries from the storage.
     * This does not modify the storage if there are no zero entries, and keeps
     * the relative order of nonzero entries otherwise.
     */
    void remove_zeros();

    /** The matrix dimensions. */
    dim<2> size;
    /**
     * The matrix entries.
     *
     * @note Despite the name, the entry values may be zero, which can be
     *       necessary dependent on the matrix format.
     * @note To be used with a Ginkgo matrix type, the entry array must be
     *       sorted in row-major order, i.e. by row index, then by column index
     *       within rows. This can be achieved by calling sort_row_major()
     */
    Array<nonzero_type> nonzeros;
};


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_DEVICE_MATRIX_DATA_HPP_
