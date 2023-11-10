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

#include <fstream>
#include <ios>
#include <limits>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/mtx_io.hpp>


template <typename ValueType>
void process(const char* input, const char* output, bool validate)
{
    std::ifstream is(input);
    std::cerr << "Reading from " << input << '\n';
    auto data = gko::read_raw<ValueType, gko::int64>(is);
    {
        std::ofstream os(output, std::ios_base::out | std::ios_base::binary);
        std::cerr << "Writing to " << output << '\n';
        if (data.size[0] <= std::numeric_limits<gko::int32>::max()) {
            gko::matrix_data<ValueType, gko::int32> int_data(data.size);
            for (auto entry : data.nonzeros) {
                int_data.nonzeros.emplace_back(
                    static_cast<gko::int32>(entry.row),
                    static_cast<gko::int32>(entry.column), entry.value);
            }
            gko::write_binary_raw(os, int_data);
        } else {
            gko::write_binary_raw(os, data);
        }
    }
    if (validate) {
        std::ifstream is(output, std::ios_base::in | std::ios_base::binary);
        auto data2 = gko::read_binary_raw<ValueType, gko::int64>(is);
        std::cerr << "Comparing against previously read data\n";
        if (data.size != data2.size) {
            throw GKO_STREAM_ERROR("Mismatching sizes!");
        }
        if (data.nonzeros != data2.nonzeros) {
            throw GKO_STREAM_ERROR("Differing data!");
        }
        std::cerr << "Validation successful!\n";
    }
}


int main(int argc, char** argv)
{
    if (argc < 3 || (std::string{argv[1]} == "-v" && argc < 4)) {
        std::cerr
            << "Usage: " << argv[0]
            << " [-v] [input] [output]\n"
               "Reads the input file in MatrixMarket format and converts it"
               "to Ginkgo's binary format.\nWith the optional -v flag, reads "
               "the written binary output again and compares it with the "
               "original input to validate the conversion.\n"
               "The conversion uses a complex value type if necessary, "
               "the highest possible value precision and the smallest "
               "possible index type.\n";
        return 1;
    }
    bool validate = std::string{argv[1]} == "-v";
    const auto input = validate ? argv[2] : argv[1];
    const auto output = validate ? argv[3] : argv[2];
    std::string header;
    {
        // read header, close file again
        std::ifstream is(input);
        std::getline(is, header);
    }
    try {
        if (header.find("complex") != std::string::npos) {
            std::cerr << "Input matrix is complex\n";
            process<std::complex<double>>(input, output, validate);
        } else {
            std::cerr << "Input matrix is real\n";
            process<double>(input, output, validate);
        }
    } catch (gko::Error& err) {
        std::cerr << err.what() << '\n';
        return 2;
    }
}
