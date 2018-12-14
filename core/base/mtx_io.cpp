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

#include <ginkgo/core/base/mtx_io.hpp>


#include <algorithm>
#include <cctype>
#include <map>
#include <regex>
#include <string>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>


namespace gko {
namespace {


// utilities for error checking
#define CHECK_STREAM(_stream, _message) \
    if ((_stream).fail()) {             \
        throw STREAM_ERROR(_message);   \
    }


#define CHECK_MATCH(_result, _message) \
    if (!_result) {                    \
        throw STREAM_ERROR(_message);  \
    }


// this class encapsulates the complexity of reading and writing matrix market
// format files
template <typename ValueType, typename IndexType>
class mtx_io {
public:
    // returns an instance of a mtx_io
    static const mtx_io &get()
    {
        static mtx_io instance;
        return instance;
    }

    // reads a matrix from a stream
    matrix_data<ValueType, IndexType> read(std::istream &is) const
    {
        auto parsed_header = this->read_header(is);
        std::istringstream dimensions_stream(parsed_header.dimensions_line);
        auto data = parsed_header.layout->read_data(
            dimensions_stream, is, parsed_header.entry, parsed_header.modifier);
        data.ensure_row_major_order();
        return data;
    }

    // writes a matrix to a stream
    void write(std::ostream &os, const matrix_data<ValueType, IndexType> &data,
               const std::string &header) const
    {
        std::istringstream header_stream(header);
        auto parsed_header = this->read_description_line(header_stream);
        CHECK_STREAM(os << header,
                     "error when writing the matrix market header");
        parsed_header.layout->write_data(os, data, parsed_header.entry,
                                         parsed_header.modifier);
    }

private:
    // entry format hierarchy provides algorithms for reading/writing a single
    // entry of the matrix, depending on its storage scheme:
    struct entry_format {
        virtual ValueType read_entry(std::istream &is) const = 0;
        virtual void write_entry(std::ostream &os,
                                 const ValueType &value) const = 0;
    };

    // maps entry format specification strings to algorithms
    std::map<std::string, const entry_format *> format_map;

    // the value is encoded as a decimal number
    struct : entry_format {
        ValueType read_entry(std::istream &is) const override
        {
            double result{};
            CHECK_STREAM(is >> result, "error while reading matrix entry");
            return static_cast<ValueType>(result);
        }

        void write_entry(std::ostream &os,
                         const ValueType &value) const override
        {
            write_entry_impl<ValueType>(os, value);
        }

    private:
        template <typename T>
        static xstd::enable_if_t<is_complex<T>()> write_entry_impl(
            std::ostream &, const T &)
        {
            throw STREAM_ERROR(
                "trying to write a complex matrix into a real entry format");
        }

        template <typename T>
        static xstd::enable_if_t<!is_complex<T>()> write_entry_impl(
            std::ostream &os, const T &value)
        {
            CHECK_STREAM(os << static_cast<double>(value),
                         "error while writing matrix entry");
        }

    } real_format{};

    // the value is encoded as a pair of decimal numbers
    struct : entry_format {
        ValueType read_entry(std::istream &is) const override
        {
            return read_entry_impl<ValueType>(is);
        }

        void write_entry(std::ostream &os,
                         const ValueType &value) const override
        {
            CHECK_STREAM(os << static_cast<double>(real(value)) << ' '
                            << static_cast<double>(imag(value)),
                         "error while writing matrix entry");
        }

    private:
        template <typename T>
        static xstd::enable_if_t<is_complex<T>(), T> read_entry_impl(
            std::istream &is)
        {
            using real_type = remove_complex<T>;
            double real{};
            double imag{};
            CHECK_STREAM(is >> real >> imag,
                         "error while reading matrix entry");
            return {static_cast<real_type>(real), static_cast<real_type>(imag)};
        }

        template <typename T>
        static xstd::enable_if_t<!is_complex<T>(), T> read_entry_impl(
            std::istream &)
        {
            throw STREAM_ERROR(
                "trying to read a complex matrix into a real storage type");
        }

    } complex_format{};

    // the value is not stored - it is implicitly set to "1"
    struct : entry_format {
        ValueType read_entry(std::istream &) const override
        {
            return one<ValueType>();
        }

        void write_entry(std::ostream &, const ValueType &) const override {}

    } pattern_format{};


    // storage modifier hierarchy provides algorithms for handling storage
    // modifiers (general, symetric, skew symetric, hermitian) and filling the
    // entire matrix from the stored parts
    struct storage_modifier {
        virtual size_type get_reservation_size(
            size_type num_rows, size_type num_cols,
            size_type num_nonzeros) const = 0;

        virtual void insert_entry(
            const IndexType &row, const IndexType &col, const ValueType &entry,
            matrix_data<ValueType, IndexType> &data) const = 0;

        virtual size_type get_row_start(size_type col) const = 0;
    };

    // maps storage modifier specification strings to algorithms
    std::map<std::string, const storage_modifier *> modifier_map;

    // all (nonzero) elements of the matrix are stored
    struct : storage_modifier {
        size_type get_reservation_size(size_type, size_type,
                                       size_type num_nonzeros) const override
        {
            return num_nonzeros;
        }

        void insert_entry(
            const IndexType &row, const IndexType &col, const ValueType &entry,
            matrix_data<ValueType, IndexType> &data) const override
        {
            data.nonzeros.emplace_back(row, col, entry);
        }

        size_type get_row_start(size_type) const override { return 0; }
    } general_modifier{};

    // the matrix is symmetric, only the lower triangle of the matrix is stored,
    // the upper part is obtained through transposition
    struct : storage_modifier {
        size_type get_reservation_size(size_type num_rows, size_type num_cols,
                                       size_type num_nonzeros) const override
        {
            return 2 * num_nonzeros - max(num_rows, num_cols);
        }

        void insert_entry(
            const IndexType &row, const IndexType &col, const ValueType &entry,
            matrix_data<ValueType, IndexType> &data) const override
        {
            data.nonzeros.emplace_back(row, col, entry);
            if (row != col) {
                data.nonzeros.emplace_back(col, row, entry);
            }
        }

        size_type get_row_start(size_type col) const override { return col; }
    } symmetric_modifier{};

    // the matrix is skew-symmetric, only the strict lower triangle of the
    // matrix is stored, the upper part is obtained through transposition, and
    // sign change
    struct : storage_modifier {
        size_type get_reservation_size(size_type, size_type,
                                       size_type num_nonzeros) const override
        {
            return 2 * num_nonzeros;
        }

        void insert_entry(
            const IndexType &row, const IndexType &col, const ValueType &entry,
            matrix_data<ValueType, IndexType> &data) const override
        {
            data.nonzeros.emplace_back(row, col, entry);
            data.nonzeros.emplace_back(col, row, -entry);
        }

        size_type get_row_start(size_type col) const override
        {
            return col + 1;
        }
    } skew_symmetric_modifier{};

    // the matrix is hermitian, only the lower triangle of the matrix is stored,
    // the upper part is obtained through conjugate transposition
    struct : storage_modifier {
        size_type get_reservation_size(size_type num_rows, size_type num_cols,
                                       size_type num_nonzeros) const override
        {
            return 2 * num_nonzeros - max(num_rows, num_cols);
        }

        void insert_entry(
            const IndexType &row, const IndexType &col, const ValueType &entry,
            matrix_data<ValueType, IndexType> &data) const override
        {
            data.nonzeros.emplace_back(row, col, entry);
            if (row != col) {
                data.nonzeros.emplace_back(col, row, conj(entry));
            }
        }

        size_type get_row_start(size_type col) const override { return col; }
    } hermitian_modifier{};


    // the storage layout hierarchy implements algorithms for reading/writing
    // the matrix based on its storage layout (column-major dense or coordinate
    // sparse)
    struct storage_layout {
        virtual matrix_data<ValueType, IndexType> read_data(
            std::istream &header, std::istream &content,
            const entry_format *entry_reader,
            const storage_modifier *modifier) const = 0;

        virtual void write_data(std::ostream &os,
                                const matrix_data<ValueType, IndexType> &data,
                                const entry_format *entry_writer,
                                const storage_modifier *modifier) const = 0;
    };

    // maps storage layout specification strings to algorithms
    std::map<std::string, const storage_layout *> layout_map;

    // the matrix is sparse, and every nonzero is stored together with its
    // coordinates
    struct : storage_layout {
        matrix_data<ValueType, IndexType> read_data(
            std::istream &header, std::istream &content,
            const entry_format *entry_reader,
            const storage_modifier *modifier) const override
        {
            size_type num_rows{};
            size_type num_cols{};
            size_type num_nonzeros{};
            CHECK_STREAM(
                header >> num_rows >> num_cols >> num_nonzeros,
                "error when determining matrix size, expected: rows cols nnz");
            matrix_data<ValueType, IndexType> data(dim<2>{num_rows, num_cols});
            data.nonzeros.reserve(modifier->get_reservation_size(
                num_rows, num_cols, num_nonzeros));
            for (size_type i = 0; i < num_nonzeros; ++i) {
                IndexType row{};
                IndexType col{};
                CHECK_STREAM(content >> row >> col,
                             "error when reading coordinates of matrix entry " +
                                 std::to_string(i));
                auto entry = entry_reader->read_entry(content);
                CHECK_STREAM(content, "error when reading matrix entry " +
                                          std::to_string(i));
                modifier->insert_entry(row - 1, col - 1, entry, data);
            }
            return data;
        }

        void write_data(std::ostream &os,
                        const matrix_data<ValueType, IndexType> &data,
                        const entry_format *entry_writer,
                        const storage_modifier *) const override
        {
            // TODO: use the storage modifier
            CHECK_STREAM(os << data.size[0] << ' ' << data.size[1] << ' '
                            << data.nonzeros.size() << '\n',
                         "error when writing size information");
            for (const auto &nonzero : data.nonzeros) {
                CHECK_STREAM(
                    os << nonzero.row + 1 << ' ' << nonzero.column + 1 << ' ',
                    "error when writing matrix index");
                entry_writer->write_entry(os, nonzero.value);
                CHECK_STREAM(os << '\n', "error when writing matrix data");
            }
        }

    } coordinate_layout{};

    // the matrix is dense, only the values are stored, without coordinates
    struct : storage_layout {
        matrix_data<ValueType, IndexType> read_data(
            std::istream &header, std::istream &content,
            const entry_format *entry_reader,
            const storage_modifier *modifier) const override
        {
            size_type num_rows{};
            size_type num_cols{};
            CHECK_STREAM(
                header >> num_rows >> num_cols,
                "error when determining matrix size, expected: rows cols nnz");
            matrix_data<ValueType, IndexType> data(dim<2>{num_rows, num_cols});
            data.nonzeros.reserve(modifier->get_reservation_size(
                num_rows, num_cols, num_rows * num_cols));
            for (size_type col = 0; col < num_cols; ++col) {
                for (size_type row = modifier->get_row_start(col);
                     row < num_rows; ++row) {
                    auto entry = entry_reader->read_entry(content);
                    CHECK_STREAM(content, "error when reading matrix entry " +
                                              std::to_string(row) + " ," +
                                              std::to_string(col));
                    modifier->insert_entry(row, col, entry, data);
                }
            }
            return data;
        }

        void write_data(std::ostream &os,
                        const matrix_data<ValueType, IndexType> &data,
                        const entry_format *entry_writer,
                        const storage_modifier *) const override
        {
            using nt = typename matrix_data<ValueType, IndexType>::nonzero_type;
            auto nonzeros = data.nonzeros;
            std::sort(begin(nonzeros), end(nonzeros), [](nt x, nt y) {
                return std::tie(x.column, x.row) < std::tie(y.column, y.row);
            });
            IndexType pos = 0;
            // TODO: use the storage modifier
            CHECK_STREAM(os << data.size[0] << ' ' << data.size[1] << '\n',
                         "error when writing size information");
            for (auto j = zero<IndexType>(); j < data.size[1]; ++j) {
                for (auto i = zero<IndexType>(); i < data.size[0]; ++i) {
                    if (pos >= nonzeros.size() ||
                        std::tie(nonzeros[pos].row, nonzeros[pos].column) !=
                            std::tie(i, j)) {
                        entry_writer->write_entry(os, zero<ValueType>());
                    } else {
                        entry_writer->write_entry(os, nonzeros[pos].value);
                        ++pos;
                    }
                    CHECK_STREAM(os << '\n', "error when writing matrix data");
                }
            }
        }
    } array_layout{};


    // the constructors establishes the mapping between specification strings to
    // classes representing algorithms
    mtx_io()
        : format_map{{"integer", &real_format},
                     {"real", &real_format},
                     {"complex", &complex_format},
                     {"pattern", &pattern_format}},
          modifier_map{{"general", &general_modifier},
                       {"symmetric", &symmetric_modifier},
                       {"skew-symmetric", &skew_symmetric_modifier},
                       {"hermitian", &hermitian_modifier}},
          layout_map{{"array", &array_layout},
                     {"coordinate", &coordinate_layout}}
    {}

    // represents the parsed header, whose components can then be used to
    // read/write the rest of the file
    struct header_data {
        const entry_format *entry{};
        const storage_modifier *modifier{};
        const storage_layout *layout{};
        std::string dimensions_line{};
    };

    // reads and parses the first line of the header
    header_data read_description_line(std::istream &is) const
    {
        header_data data{};

        std::string description_line{};
        do {
            CHECK_STREAM(getline(is, description_line),
                         "error when reading the header line");
        } while (description_line == "");
        transform(begin(description_line), end(description_line),
                  begin(description_line),
                  [](unsigned char c) { return std::tolower(c); });

        std::smatch match{};
        CHECK_MATCH(
            regex_match(
                description_line, match,
                std::regex("%%matrixmarket matrix "
                           "(coordinate|array) "
                           "(real|integer|complex|pattern) "
                           "(general|symmetric|skew-symmetric|hermitian)")),
            "error parsing the header line, the header format should have "
            "the following structure:\n"
            "    %%MatrixMarket matrix <LAYOUT-TYPE> <VALUE-TYPE> "
            "<LAYOUT-MODIFIER>\n"
            "where:\n"
            "    <LAYOUT-TYPE>     is one of: coordinate, array\n"
            "    <VALUE-TYPE>      is one of: real, integer, complex, pattern\n"
            "    <LAYOUT-MODIFIER> is one of: general, symmetric, "
            "skew-symmetric, hermitian\n");

        data.layout = layout_map.at(match[1]);
        data.entry = format_map.at(match[2]);
        data.modifier = modifier_map.at(match[3]);

        return data;
    }

    // reads and parses the header
    header_data read_header(std::istream &is) const
    {
        auto data = read_description_line(is);
        do {
            CHECK_STREAM(getline(is, data.dimensions_line),
                         "error when reading the dimensions line");
        } while (data.dimensions_line[0] == '%');
        return data;
    }
};


}  // namespace


template <typename ValueType, typename IndexType>
matrix_data<ValueType, IndexType> read_raw(std::istream &is)
{
    return mtx_io<ValueType, IndexType>::get().read(is);
}


template <typename ValueType, typename IndexType>
void write_raw(std::ostream &os, const matrix_data<ValueType, IndexType> &data,
               layout_type layout)
{
    // TODO: add support for all layout combinations
    mtx_io<ValueType, IndexType>::get().write(
        os, data,
        std::string("%%MatrixMarket matrix ") +
            (layout == layout_type::array ? "array" : "coordinate") + " " +
            (is_complex<ValueType>() ? "complex" : "real") + " general\n");
}


#define DECLARE_READ_RAW(ValueType, IndexType) \
    matrix_data<ValueType, IndexType> read_raw(std::istream &is)
#define DECLARE_WRITE_RAW(ValueType, IndexType)                   \
    void write_raw(std::ostream &os,                              \
                   const matrix_data<ValueType, IndexType> &data, \
                   layout_type layout);
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_READ_RAW);
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_WRITE_RAW);


}  // namespace gko
