// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/mtx_io.hpp>


#include <algorithm>
#include <cctype>
#include <cstring>
#include <limits>
#include <map>
#include <regex>
#include <string>
#include <type_traits>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>


namespace gko {
namespace {


// utilities for error checking
#define GKO_CHECK_STREAM(_stream, _message) \
    if ((_stream).fail()) {                 \
        throw GKO_STREAM_ERROR(_message);   \
    }


#define GKO_CHECK_MATCH(_result, _message) \
    if (!_result) {                        \
        throw GKO_STREAM_ERROR(_message);  \
    }


/**
 * The mtx_io class provides the functionality of reading and writing matrix
 * market format files.
 *
 * @ingroup mtx_io
 */
template <typename ValueType, typename IndexType>
class mtx_io {
public:
    /**
     * Returns an instance of the matrix.
     *
     * @return an instance of the matrix.
     */
    static const mtx_io& get()
    {
        static mtx_io instance;
        return instance;
    }

    /**
     * Reads a matrix from a stream.
     *
     * @param is  the input stream.
     *
     * @return the matrix data.
     */
    matrix_data<ValueType, IndexType> read(std::istream& is) const
    {
        auto parsed_header = this->read_header(is);
        std::istringstream dimensions_stream(parsed_header.dimensions_line);
        auto data = parsed_header.layout->read_data(
            dimensions_stream, is, parsed_header.entry, parsed_header.modifier);
        data.sort_row_major();
        return data;
    }

    /**
     * Writes a matrix to a stream.
     *
     * @param os  the output stream.
     * @param data  the matrix data to be written.
     * @param header  the header to be printed at the start of the file.
     */
    void write(std::ostream& os, const matrix_data<ValueType, IndexType>& data,
               const std::string& header) const
    {
        std::istringstream header_stream(header);
        auto parsed_header = this->read_description_line(header_stream);
        GKO_CHECK_STREAM(os << header,
                         "error when writing the matrix market header");
        parsed_header.layout->write_data(os, data, parsed_header.entry,
                                         parsed_header.modifier);
    }

private:
    /**
     * entry format hierarchy provides algorithms for reading/writing a single
     * entry of the matrix, depending on its storage scheme:
     */
    struct entry_format {
        virtual ValueType read_entry(std::istream& is) const = 0;
        virtual void write_entry(std::ostream& os,
                                 const ValueType& value) const = 0;
    };

    /**
     * maps entry format specification strings to algorithms
     */
    std::map<std::string, const entry_format*> format_map;

    /**
     * the value is encoded as a decimal number
     */
    struct : entry_format {
        /**
         * reads entry from the input stream
         *
         * @param  is the input stream
         *
         * @return the matrix entry.
         */
        ValueType read_entry(std::istream& is) const override
        {
            double result{};
            GKO_CHECK_STREAM(is >> result, "error while reading matrix entry");
            return static_cast<ValueType>(result);
        }

        /**
         * writes entry to the output stream
         *
         * @param  os the output stream
         * @param  value the matrix entry to be written
         */
        void write_entry(std::ostream& os,
                         const ValueType& value) const override
        {
            write_entry_impl<ValueType>(os, value);
        }

    private:
        template <typename T>
        static std::enable_if_t<is_complex_s<T>::value> write_entry_impl(
            std::ostream&, const T&)
        {
            throw GKO_STREAM_ERROR(
                "trying to write a complex matrix into a real entry format");
        }

        template <typename T>
        static std::enable_if_t<!is_complex_s<T>::value> write_entry_impl(
            std::ostream& os, const T& value)
        {
            GKO_CHECK_STREAM(os << static_cast<double>(value),
                             "error while writing matrix entry");
        }

    } real_format{};

    /**
     * the value is encoded as a pair of decimal numbers
     */
    struct : entry_format {
        /**
         * reads entry from the input stream
         *
         * @param  is the input stream
         *
         * @return the matrix entry.
         */
        ValueType read_entry(std::istream& is) const override
        {
            return read_entry_impl<ValueType>(is);
        }

        /**
         * writes entry to the output stream
         *
         * @param  os the output stream
         * @param  value the matrix entry to be written
         */
        void write_entry(std::ostream& os,
                         const ValueType& value) const override
        {
            GKO_CHECK_STREAM(os << static_cast<double>(real(value)) << ' '
                                << static_cast<double>(imag(value)),
                             "error while writing matrix entry");
        }

    private:
        template <typename T>
        static std::enable_if_t<is_complex_s<T>::value, T> read_entry_impl(
            std::istream& is)
        {
            using real_type = remove_complex<T>;
            double real{};
            double imag{};
            GKO_CHECK_STREAM(is >> real >> imag,
                             "error while reading matrix entry");
            return {static_cast<real_type>(real), static_cast<real_type>(imag)};
        }

        template <typename T>
        static std::enable_if_t<!is_complex_s<T>::value, T> read_entry_impl(
            std::istream&)
        {
            throw GKO_STREAM_ERROR(
                "trying to read a complex matrix into a real storage type");
        }

    } complex_format{};

    /**
     * the value is not stored - it is implicitly set to "1"
     */
    struct : entry_format {
        /**
         * reads entry from the input stream
         *
         * @param  dummy input stream
         *
         * @return the matrix entry(one).
         */
        ValueType read_entry(std::istream&) const override
        {
            return one<ValueType>();
        }

        /**
         * writes entry to the output stream
         *
         * @param  dummy output stream
         * @param  dummy matrix entry to be written
         */
        void write_entry(std::ostream&, const ValueType&) const override {}

    } pattern_format{};


    /**
     * storage modifier hierarchy provides algorithms for handling storage
     * modifiers (general, symmetric, skew symmetric, hermitian) and filling the
     * entire matrix from the stored parts
     */
    struct storage_modifier {
        virtual size_type get_reservation_size(
            size_type num_rows, size_type num_cols,
            size_type num_nonzeros) const = 0;

        virtual void insert_entry(
            const IndexType& row, const IndexType& col, const ValueType& entry,
            matrix_data<ValueType, IndexType>& data) const = 0;

        virtual size_type get_row_start(size_type col) const = 0;
    };

    /**
     * maps storage modifier specification strings to algorithms
     */
    std::map<std::string, const storage_modifier*> modifier_map;

    /**
     * all (nonzero) elements of the matrix are stored
     */
    struct : storage_modifier {
        /**
         * get the reservation size
         *
         * @param num_rows  the number of rows
         * @param num_cols  the number of columns
         * @param num_nonzeros  the number of non-zeros
         *
         * @return the reservation size
         */
        size_type get_reservation_size(size_type, size_type,
                                       size_type num_nonzeros) const override
        {
            return num_nonzeros;
        }

        /**
         * Insert an entry
         *
         * @param row  The row where the entry is to be inserted.
         * @param col  The column where the entry is to be inserted.
         * @param entry  the entry to be inserted.
         * @param data  the data holding the matrix.
         */
        void insert_entry(
            const IndexType& row, const IndexType& col, const ValueType& entry,
            matrix_data<ValueType, IndexType>& data) const override
        {
            data.nonzeros.emplace_back(row, col, entry);
        }

        /**
         * Get the start of the rows
         */
        size_type get_row_start(size_type) const override { return 0; }
    } general_modifier{};

    /**
     * the matrix is symmetric, only the lower triangle of the matrix is stored,
     * the upper part is obtained through transposition
     */
    struct : storage_modifier {
        /**
         * get the reservation size
         * @param num_rows
         * @param num_cols
         * @param num_nonzeros  the number of non-zeros
         *
         * @return the reservation size.
         */
        size_type get_reservation_size(size_type num_rows, size_type num_cols,
                                       size_type num_nonzeros) const override
        {
            return 2 * num_nonzeros -
                   min(2 * num_nonzeros, max(num_rows, num_cols));
        }

        /**
         * Insert an entry
         *
         * @param row  The row where the entry is to be inserted.
         * @param col  The column where the entry is to be inserted.
         * @param entry  the entry to be inserted.
         * @param data  the data holding the matrix.
         */
        void insert_entry(
            const IndexType& row, const IndexType& col, const ValueType& entry,
            matrix_data<ValueType, IndexType>& data) const override
        {
            data.nonzeros.emplace_back(row, col, entry);
            if (row != col) {
                data.nonzeros.emplace_back(col, row, entry);
            }
        }

        /**
         * Get the start of the rows
         */
        size_type get_row_start(size_type col) const override { return col; }
    } symmetric_modifier{};

    /**
     * the matrix is skew-symmetric, only the strict lower triangle of the
     * matrix is stored, the upper part is obtained through transposition, and
     * sign change
     */
    struct : storage_modifier {
        /**
         * get the reservation size
         *
         * @param num_rows
         * @param num_cols
         * @param num_nonzeros  the number of non-zeros
         *
         * @return the reservation size.
         */
        size_type get_reservation_size(size_type, size_type,
                                       size_type num_nonzeros) const override
        {
            return 2 * num_nonzeros;
        }

        /**
         * Insert an entry
         *
         * @param row  The row where the entry is to be inserted.
         * @param col  The column where the entry is to be inserted.
         * @param entry  the entry to be inserted.
         * @param data  the data holding the matrix.
         */
        void insert_entry(
            const IndexType& row, const IndexType& col, const ValueType& entry,
            matrix_data<ValueType, IndexType>& data) const override
        {
            data.nonzeros.emplace_back(row, col, entry);
            if (row != col) {
                data.nonzeros.emplace_back(col, row, -entry);
            }
        }

        /**
         * Get the start of the rows
         */
        size_type get_row_start(size_type col) const override
        {
            return col + 1;
        }
    } skew_symmetric_modifier{};

    /**
     * the matrix is hermitian, only the lower triangle of the matrix is stored,
     * the upper part is obtained through conjugate transposition
     */
    struct : storage_modifier {
        /**
         * get the reservation size
         *
         * @param num_rows
         * @param num_cols
         * @param num_nonzeros  the number of non-zeros
         *
         * @return the reservation size.
         */
        size_type get_reservation_size(size_type num_rows, size_type num_cols,
                                       size_type num_nonzeros) const override
        {
            return 2 * num_nonzeros -
                   min(2 * num_nonzeros, max(num_rows, num_cols));
        }

        /**
         * Insert an entry
         *
         * @param row  The row where the entry is to be inserted.
         * @param col  The column where the entry is to be inserted.
         * @param entry  the entry to be inserted.
         * @param data  the data holding the matrix.
         */
        void insert_entry(
            const IndexType& row, const IndexType& col, const ValueType& entry,
            matrix_data<ValueType, IndexType>& data) const override
        {
            data.nonzeros.emplace_back(row, col, entry);
            if (row != col) {
                data.nonzeros.emplace_back(col, row, conj(entry));
            }
        }

        /**
         * Get the start of the rows
         */
        size_type get_row_start(size_type col) const override { return col; }
    } hermitian_modifier{};


    /**
     * the storage layout hierarchy implements algorithms for reading/writing
     * the matrix based on its storage layout (column-major dense or coordinate
     * sparse)
     */
    struct storage_layout {
        /**
         * Read the matrix data
         *
         * @param header  The header in the matrix file
         * @param content  The content in the matrix file
         * @param entry_reader  The entry format in the matrix file
         * @param modifier  The storage modifier for the matrix file
         *
         * @return the matrix data
         */
        virtual matrix_data<ValueType, IndexType> read_data(
            std::istream& header, std::istream& content,
            const entry_format* entry_reader,
            const storage_modifier* modifier) const = 0;
        /**
         * Write the matrix data
         *
         * @param os  The output stream to write to
         * @param data  The matrix data to write
         * @param entry_writer  The entry format to write in.
         * @param modifier  The storage modifier
         */
        virtual void write_data(std::ostream& os,
                                const matrix_data<ValueType, IndexType>& data,
                                const entry_format* entry_writer,
                                const storage_modifier* modifier) const = 0;
    };

    /**
     * maps storage layout specification strings to algorithms
     */
    std::map<std::string, const storage_layout*> layout_map;

    /**
     * the matrix is sparse, and every nonzero is stored together with its
     * coordinates
     */
    struct : storage_layout {
        /**
         * Read the matrix data
         *
         * @param header  The header in the matrix file
         * @param content  The content in the matrix file
         * @param entry_reader  The entry format in the matrix file
         * @param modifier  The storage modifier for the matrix file
         *
         * @return the matrix data
         */
        matrix_data<ValueType, IndexType> read_data(
            std::istream& header, std::istream& content,
            const entry_format* entry_reader,
            const storage_modifier* modifier) const override
        {
            size_type num_rows{};
            size_type num_cols{};
            size_type num_nonzeros{};
            GKO_CHECK_STREAM(
                header >> num_rows >> num_cols >> num_nonzeros,
                "error when determining matrix size, expected: rows cols nnz");
            matrix_data<ValueType, IndexType> data(dim<2>{num_rows, num_cols});
            data.nonzeros.reserve(modifier->get_reservation_size(
                num_rows, num_cols, num_nonzeros));
            for (size_type i = 0; i < num_nonzeros; ++i) {
                IndexType row{};
                IndexType col{};
                GKO_CHECK_STREAM(
                    content >> row >> col,
                    "error when reading coordinates of matrix entry " +
                        std::to_string(i));
                auto entry = entry_reader->read_entry(content);
                GKO_CHECK_STREAM(content, "error when reading matrix entry " +
                                              std::to_string(i));
                modifier->insert_entry(row - 1, col - 1, entry, data);
            }
            return data;
        }

        /**
         * Write the matrix data
         *
         * @param os  The output stream to write to
         * @param data  The matrix data to write
         * @param entry_writer  The entry format to write in.
         * @param modifier  The storage modifier
         */
        void write_data(std::ostream& os,
                        const matrix_data<ValueType, IndexType>& data,
                        const entry_format* entry_writer,
                        const storage_modifier*) const override
        {
            // TODO: use the storage modifier
            GKO_CHECK_STREAM(os << data.size[0] << ' ' << data.size[1] << ' '
                                << data.nonzeros.size() << '\n',
                             "error when writing size information");
            for (const auto& nonzero : data.nonzeros) {
                GKO_CHECK_STREAM(
                    os << nonzero.row + 1 << ' ' << nonzero.column + 1 << ' ',
                    "error when writing matrix index");
                entry_writer->write_entry(os, nonzero.value);
                GKO_CHECK_STREAM(os << '\n', "error when writing matrix data");
            }
        }

    } coordinate_layout{};

    /**
     * the matrix is dense, only the values are stored, without coordinates
     */
    struct : storage_layout {
        /**
         * Read the matrix data
         *
         * @param header  The header in the matrix file
         * @param content  The content in the matrix file
         * @param entry_reader  The entry format in the matrix file
         * @param modifier  The storage modifier for the matrix file
         *
         * @return the matrix data
         */
        matrix_data<ValueType, IndexType> read_data(
            std::istream& header, std::istream& content,
            const entry_format* entry_reader,
            const storage_modifier* modifier) const override
        {
            size_type num_rows{};
            size_type num_cols{};
            GKO_CHECK_STREAM(
                header >> num_rows >> num_cols,
                "error when determining matrix size, expected: rows cols nnz");
            matrix_data<ValueType, IndexType> data(dim<2>{num_rows, num_cols});
            data.nonzeros.reserve(modifier->get_reservation_size(
                num_rows, num_cols, num_rows * num_cols));
            for (size_type col = 0; col < num_cols; ++col) {
                for (size_type row = modifier->get_row_start(col);
                     row < num_rows; ++row) {
                    auto entry = entry_reader->read_entry(content);
                    GKO_CHECK_STREAM(content,
                                     "error when reading matrix entry " +
                                         std::to_string(row) + " ," +
                                         std::to_string(col));
                    modifier->insert_entry(row, col, entry, data);
                }
            }
            return data;
        }

        /**
         * Write the matrix data
         *
         * @param os  The output stream to write to
         * @param data  The matrix data to write
         * @param entry_writer  The entry format to write in.
         * @param modifier  The storage modifier
         */
        void write_data(std::ostream& os,
                        const matrix_data<ValueType, IndexType>& data,
                        const entry_format* entry_writer,
                        const storage_modifier*) const override
        {
            using nt = typename matrix_data<ValueType, IndexType>::nonzero_type;
            auto nonzeros = data.nonzeros;
            std::sort(begin(nonzeros), end(nonzeros), [](nt x, nt y) {
                return std::tie(x.column, x.row) < std::tie(y.column, y.row);
            });
            IndexType pos = 0;
            // TODO: use the storage modifier
            GKO_CHECK_STREAM(os << data.size[0] << ' ' << data.size[1] << '\n',
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
                    GKO_CHECK_STREAM(os << '\n',
                                     "error when writing matrix data");
                }
            }
        }
    } array_layout{};


    /**
     * the constructors establishes the mapping between specification strings to
     * classes representing algorithms
     */
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

    /**
     * represents the parsed header, whose components can then be used to
     * read/write the rest of the file
     */
    struct header_data {
        const entry_format* entry{};
        const storage_modifier* modifier{};
        const storage_layout* layout{};
        std::string dimensions_line{};
    };

    /**
     * reads and parses the first line of the header
     *
     * @param is  the input stream
     *
     * @return the data containing the description
     */
    header_data read_description_line(std::istream& is) const
    {
        header_data data{};

        std::string description_line{};
        do {
            GKO_CHECK_STREAM(getline(is, description_line),
                             "error when reading the header line");
        } while (description_line == "");
        transform(begin(description_line), end(description_line),
                  begin(description_line),
                  [](unsigned char c) { return std::tolower(c); });

        std::smatch match{};
        GKO_CHECK_MATCH(
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

    /**
     * reads and parses the header
     *
     * @param is  The input stream to read the header from.
     *
     * @return the header data
     */
    header_data read_header(std::istream& is) const
    {
        auto data = read_description_line(is);
        do {
            GKO_CHECK_STREAM(getline(is, data.dimensions_line),
                             "error when reading the dimensions line");
        } while (data.dimensions_line[0] == '%');
        return data;
    }
};


}  // namespace


/**
 * Reads raw data from the stream.
 *
 * @param is  the input stream
 *
 * @return matrix_data  the matrix data.
 */
template <typename ValueType, typename IndexType>
matrix_data<ValueType, IndexType> read_raw(std::istream& is)
{
    return mtx_io<ValueType, IndexType>::get().read(is);
}


/**
 * Returns the magic number at the beginning of the binary format header for the
 * given type parameters.
 *
 * @tparam ValueType  the value type to be used for the binary storage
 * @tparam IndexType  the index type to be used for the binary storage
 */
template <typename ValueType, typename IndexType>
static constexpr uint64 binary_format_magic()
{
    constexpr auto is_int = std::is_same<IndexType, int32>::value;
    constexpr auto is_long = std::is_same<IndexType, int64>::value;
    constexpr auto is_double = std::is_same<ValueType, double>::value;
    constexpr auto is_float = std::is_same<ValueType, float>::value;
    constexpr auto is_complex_double =
        std::is_same<ValueType, std::complex<double>>::value;
    constexpr auto is_complex_float =
        std::is_same<ValueType, std::complex<float>>::value;
    static_assert(is_int || is_long, "invalid storage index type");
    static_assert(
        is_double || is_float || is_complex_double || is_complex_float,
        "invalid storage value type");
    constexpr auto index_bit = is_int ? 'I' : 'L';
    constexpr auto value_bit =
        is_double ? 'D' : (is_float ? 'S' : (is_complex_double ? 'Z' : 'C'));
    constexpr uint64 shift = 256;
    constexpr uint64 type_bits = index_bit * shift + value_bit;
    return 'G' +
           shift *
               ('I' +
                shift *
                    ('N' +
                     shift *
                         ('K' +
                          shift * ('G' + shift * ('O' + shift * type_bits)))));
}


namespace {


template <bool first>
struct select_helper {};

template <>
struct select_helper<true> {
    template <typename T1, typename T2>
    static T1 get(T1 val, T2)
    {
        return val;
    }
};

template <>
struct select_helper<false> {
    template <typename T1, typename T2>
    static T2 get(T1, T2 val)
    {
        return val;
    }
};


template <typename FileValueType, typename FileIndexType, typename ValueType,
          typename IndexType>
matrix_data<ValueType, IndexType> read_binary_convert(std::istream& is,
                                                      uint64 num_rows,
                                                      uint64 num_cols,
                                                      uint64 num_entries)
{
    if (num_rows > std::numeric_limits<IndexType>::max() ||
        num_cols > std::numeric_limits<IndexType>::max()) {
        throw GKO_STREAM_ERROR(
            "cannot read into this format, its index type would overflow");
    }
    if (is_complex<FileValueType>() && !is_complex<ValueType>()) {
        throw GKO_STREAM_ERROR(
            "cannot read into this format, would assign complex to real");
    }
    matrix_data<ValueType, IndexType> result(gko::dim<2>{
        static_cast<size_type>(num_rows), static_cast<size_type>(num_cols)});
    result.nonzeros.resize(num_entries);
    constexpr auto entry_binary_size =
        sizeof(FileValueType) + 2 * sizeof(FileIndexType);
    for (size_type i = 0; i < num_entries; i++) {
        std::array<char, entry_binary_size> block;
        GKO_CHECK_STREAM(is.read(block.data(), entry_binary_size),
                         "failed reading entry " + std::to_string(i));
        FileValueType value{};
        FileIndexType row{};
        FileIndexType column{};
        std::memcpy(&row, &block[0], sizeof(FileIndexType));
        std::memcpy(&column, &block[sizeof(FileIndexType)],
                    sizeof(FileIndexType));
        std::memcpy(&value, &block[2 * sizeof(FileIndexType)],
                    sizeof(FileValueType));
        result.nonzeros[i].value = static_cast<ValueType>(
            select_helper<is_complex<ValueType>()>::get(value, real(value)));
        result.nonzeros[i].row = row;
        result.nonzeros[i].column = column;
    }
    // sort the entries
    result.sort_row_major();
    return result;
}


}  // namespace


template <typename ValueType, typename IndexType>
matrix_data<ValueType, IndexType> read_binary_raw(std::istream& is)
{
    static_assert(sizeof(uint64) == 8, "c++ is broken");  // just to be sure
    std::array<char, 32> header{};
    GKO_CHECK_STREAM(is.read(header.data(), 32), "failed reading header");
    uint64 magic{};
    uint64 num_rows{};
    uint64 num_cols{};
    uint64 num_entries{};
    std::memcpy(&magic, &header[0], 8);
    std::memcpy(&num_rows, &header[8], 8);
    std::memcpy(&num_cols, &header[16], 8);
    std::memcpy(&num_entries, &header[24], 8);
#define DECLARE_OVERLOAD(_vtype, _itype)                                  \
    else if (magic == binary_format_magic<_vtype, _itype>())              \
    {                                                                     \
        return read_binary_convert<_vtype, _itype, ValueType, IndexType>( \
            is, num_rows, num_cols, num_entries);                         \
    }
    if (false) {
    }
    DECLARE_OVERLOAD(double, int32)
    DECLARE_OVERLOAD(float, int32)
    DECLARE_OVERLOAD(std::complex<double>, int32)
    DECLARE_OVERLOAD(std::complex<float>, int32)
    DECLARE_OVERLOAD(double, int64)
    DECLARE_OVERLOAD(float, int64)
    DECLARE_OVERLOAD(std::complex<double>, int64)
    DECLARE_OVERLOAD(std::complex<float>, int64)
#undef DECLARE_OVERLOAD
    else
    {
        throw GKO_STREAM_ERROR("invalid header magic number '" +
                               std::string(header.data(), 8) + "'");
    }
}


template <typename ValueType, typename IndexType>
matrix_data<ValueType, IndexType> read_generic_raw(std::istream& is)
{
    auto first_char = is.peek();
    GKO_CHECK_STREAM(is, "failed reading from stream");
    if (first_char == '%') {
        return read_raw<ValueType, IndexType>(is);
    } else {
        return read_binary_raw<ValueType, IndexType>(is);
    }
}


template <typename ValueType, typename IndexType>
void write_binary_raw(std::ostream& os,
                      const matrix_data<ValueType, IndexType>& mtx)
{
    uint64 magic = binary_format_magic<ValueType, IndexType>();
    uint64 num_rows = mtx.size[0];
    uint64 num_cols = mtx.size[1];
    uint64 num_entries = mtx.nonzeros.size();
    std::array<char, 32> header{};
    std::memcpy(&header[0], &magic, 8);
    std::memcpy(&header[8], &num_rows, 8);
    std::memcpy(&header[16], &num_cols, 8);
    std::memcpy(&header[24], &num_entries, 8);
    GKO_CHECK_STREAM(os.write(header.data(), 32), "failed writing header");
    constexpr auto entry_binary_size =
        sizeof(ValueType) + 2 * sizeof(IndexType);
    for (size_type i = 0; i < num_entries; i++) {
        std::array<char, entry_binary_size> block;
        std::memcpy(&block[0], &mtx.nonzeros[i].row, sizeof(IndexType));
        std::memcpy(&block[sizeof(IndexType)], &mtx.nonzeros[i].column,
                    sizeof(IndexType));
        std::memcpy(&block[2 * sizeof(IndexType)], &mtx.nonzeros[i].value,
                    sizeof(ValueType));
        GKO_CHECK_STREAM(os.write(block.data(), entry_binary_size),
                         "failed writing entry " + std::to_string(i));
    }
    os.flush();
}


/**
 * Writes raw data to the stream.
 *
 * @param os  the output stream.
 * @param data  the data to be written.
 * @param layout  the layout type which the data should be written in.
 */
template <typename ValueType, typename IndexType>
void write_raw(std::ostream& os, const matrix_data<ValueType, IndexType>& data,
               layout_type layout)
{
    // TODO: add support for all layout combinations
    mtx_io<ValueType, IndexType>::get().write(
        os, data,
        std::string("%%MatrixMarket matrix ") +
            (layout == layout_type::array ? "array" : "coordinate") + " " +
            (is_complex<ValueType>() ? "complex" : "real") + " general\n");
}


#define GKO_DECLARE_READ_RAW(ValueType, IndexType) \
    matrix_data<ValueType, IndexType> read_raw(std::istream& is)
#define GKO_DECLARE_WRITE_RAW(ValueType, IndexType)               \
    void write_raw(std::ostream& os,                              \
                   const matrix_data<ValueType, IndexType>& data, \
                   layout_type layout)
#define GKO_DECLARE_READ_BINARY_RAW(ValueType, IndexType) \
    matrix_data<ValueType, IndexType> read_binary_raw(std::istream& is)
#define GKO_DECLARE_WRITE_BINARY_RAW(ValueType, IndexType) \
    void write_binary_raw(std::ostream& os,                \
                          const matrix_data<ValueType, IndexType>& data)
#define GKO_DECLARE_READ_GENERIC_RAW(ValueType, IndexType) \
    matrix_data<ValueType, IndexType> read_generic_raw(std::istream& is)
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_READ_RAW);
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_WRITE_RAW);
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_READ_BINARY_RAW);
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_WRITE_BINARY_RAW);
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_READ_GENERIC_RAW);


}  // namespace gko
