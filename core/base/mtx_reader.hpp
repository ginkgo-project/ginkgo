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

#ifndef GKO_CORE_BASE_MTX_READER_HPP_
#define GKO_CORE_BASE_MTX_READER_HPP_


#include "core/base/array.hpp"


#include <string>
#include <tuple>
#include <vector>


namespace gko {


template <typename ValueType = default_precision, typename IndexType = int32>
struct MtxData {
    size_type num_rows;
    size_type num_cols;
    std::vector<std::tuple<IndexType, IndexType, ValueType>> values;
};


// TODO: replace filenames with streams
template <typename ValueType = default_precision, typename IndexType = int32>
MtxData<ValueType, IndexType> read_raw_from_mtx(const std::string &filename);


// TODO: replace filenames with streams
template <typename ValueType = default_precision, typename IndexType = int32>
void save_raw_to_mtx(const std::string &filename,
                     const MtxData<ValueType, IndexType> &data);


class ReadableFromMtx {
public:
    // TODO: replace filenames with streams
    virtual void read_from_mtx(const std::string &filename) = 0;
    virtual void save_to_mtx(const std::string &filename) const = 0;
};


}  // namespace gko


#endif  // GKO_CORE_BASE_MTX_READER_HPP_
