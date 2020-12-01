/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#ifndef GINKGO_CORE_BASE_BLOCKUTILS_HPP_
#define GINKGO_CORE_BASE_BLOCKUTILS_HPP_


#include <ginkgo/core/base/exception.hpp>


namespace gko {
namespace blockutils {


/// Error that denotes issues between block sizes and matrix dimensions
template <typename IndexType>
class BlockSizeError : public Error {
public:
    BlockSizeError(const std::string &file, const int line,
                   const int block_size, const IndexType size)
        : Error(file, line,
                " block size = " + std::to_string(block_size) +
                    ", size = " + std::to_string(size))
    {}
};


/**
 * Computes the number of blocks
 *
 * @param block_size The size of each block
 * @param size The total size of some array/vector
 * @return The quotient of the size divided by the block size
 *         but throws when they don't divide
 */
template <typename IndexType>
IndexType getNumBlocks(const int block_size, const IndexType size)
{
    if (size % block_size != 0)
        throw BlockSizeError<IndexType>(__FILE__, __LINE__, block_size, size);
    return size / block_size;
}


}  // namespace blockutils
}  // namespace gko

#endif
