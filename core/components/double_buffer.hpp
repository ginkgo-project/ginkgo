// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_COMPONENTS_DOUBLE_BUFFER_HPP_
#define GKO_CORE_COMPONENTS_DOUBLE_BUFFER_HPP_


#include <cassert>

#include <ginkgo/core/base/types.hpp>


namespace gko {


/**
 * A double buffer alternating between two storage areas of the same size
 *
 * @tparam ValueType  the type of values in this double buffer
 */
template <typename ValueType>
struct double_buffer {
    double_buffer(ValueType* first, ValueType* second, size_type size)
        : size{size}, flip{false}, first{first}, second{second}
    {
        // make sure the pointers don't overlap
        assert((first < second && first + size <= second) ||
               (second < first && second + size <= first));
    }

    ValueType* get() const { return flip ? second : first; }

    ValueType* get_other() const { return flip ? first : second; }

    void swap() { flip = !flip; }

    size_type size;
    bool flip;
    ValueType* first;
    ValueType* second;
};


}  // namespace gko


#endif  // GKO_CORE_COMPONENTS_DOUBLE_BUFFER_HPP_
