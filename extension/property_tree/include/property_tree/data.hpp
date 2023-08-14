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

#ifndef GKO_PUBLIC_EXT_PROPERTY_TREE_DATA_HPP_
#define GKO_PUBLIC_EXT_PROPERTY_TREE_DATA_HPP_


#include <exception>
#include <string>
#include <type_traits>
#include <typeinfo>


namespace gko {
namespace extension {


/** @file data.hpp
 * It implements the base type for property tree. It only handles std::string,
 * double, long long int, bool, monostate(empty). Because it can be handled by
 * std::variant in C++17 directly, this file tries to use the same function
 * signature such that we can replace data_s without breaking public interface
 * when we decide to use C++17.
 */

class data_s;

// For empty state usage
struct monostate {};

/**
 * Check whether data_s holds type T data.
 *
 * @tparam T  type for checking
 *
 * @param d  the data_s data
 *
 * @return true if and only if data_s holds type T
 */
template <typename T>
bool holds_alternative(const data_s& d);

/**
 * Get the data with type T of data_s. If T is in the type list but not the type
 * held by data_s, it throws runtime error.
 *
 * @tparam T  type for checking
 *
 * @param d  the data_s data
 *
 * @return data with type T if data_s holds
 */
template <typename T>
T get(const data_s& d);

/**
 * The base data type for property tree. It only handles std::string,
 * double, long long int, bool, monostate(empty).
 */
class data_s {
    template <typename T>
    friend T get(const data_s& d);

    template <typename T>
    friend bool holds_alternative(const data_s& d);

public:
    /**
     * Default Constructor
     */
    data_s() : tag_(tag_type::empty_t) {}

    /**
     * Constructor for bool
     */
    data_s(bool bb) : tag_(tag_type::bool_t) { u_.bool_ = bb; }

    /**
     * Constructor for integer
     */
    data_s(long long int ii) : tag_(tag_type::int_t) { u_.int_ = ii; }

    /**
     * Constructor for integer with all integer type except for unsigned long
     * long int
     */
    template <typename T,
              typename = typename std::enable_if<
                  std::is_integral<T>::value &&
                  !std::is_same<T, unsigned long long int>::value>::type>
    data_s(T ii) : data_s(static_cast<long long int>(ii))
    {}

    /**
     * Constructor for string
     */
    data_s(const std::string& str) : tag_(tag_type::str_t) { str_ = str; };

    /**
     * Constructor for char (otherwise, it will use bool)
     */
    data_s(const char* s) : data_s(std::string(s)) {}

    /**
     * Constructor for double
     */
    data_s(double dd) : tag_(tag_type::double_t) { u_.double_ = dd; };

    /**
     * Constructor for float
     */
    data_s(float dd) : data_s(static_cast<double>(dd)) {}

protected:
    enum tag_type { str_t, int_t, double_t, bool_t, empty_t };

    template <tag_type>
    struct tag {
        using type = void;
    };

    /**
     * Get the data with Type T
     */
    template <typename T>
    T get() const
    {
        throw std::runtime_error("Not Supported");
    }

    template <tag_type T>
    typename tag<T>::type get() const;

    tag_type get_tag() const { return tag_; }

private:
    tag_type tag_;
    std::string str_;
    union {
        long long int int_;
        double double_;
        bool bool_;
    } u_;
};

template <>
long long int data_s::get<long long int>() const
{
    assert(tag_ == data_s::tag_type::int_t);
    return u_.int_;
}

template <>
std::string data_s::get<std::string>() const
{
    assert(tag_ == data_s::tag_type::str_t);
    return str_;
}

template <>
double data_s::get<double>() const
{
    assert(tag_ == data_s::tag_type::double_t);
    return u_.double_;
}

template <>
bool data_s::get<bool>() const
{
    assert(tag_ == data_s::tag_type::bool_t);
    return u_.bool_;
}


template <>
struct data_s::tag<data_s::tag_type::bool_t> {
    using type = bool;
};

template <>
struct data_s::tag<data_s::tag_type::int_t> {
    using type = long long int;
};

template <>
struct data_s::tag<data_s::tag_type::double_t> {
    using type = double;
};

template <>
struct data_s::tag<data_s::tag_type::str_t> {
    using type = std::string;
};


template <typename T>
T get(const data_s& d)
{
    static_assert(std::is_same<T, std::string>::value ||
                      std::is_same<T, long long int>::value ||
                      std::is_same<T, double>::value ||
                      std::is_same<T, bool>::value ||
                      std::is_same<T, monostate>::value,
                  "Not supported data type");
    if (holds_alternative<T>(d)) {
        return d.get<T>();
    }
    throw std::runtime_error(std::string("data does not holds the type ") +
                             typeid(T).name());
}


template <typename T>
bool holds_alternative(const data_s& d)
{
    static_assert(std::is_same<T, std::string>::value ||
                      std::is_same<T, long long int>::value ||
                      std::is_same<T, double>::value ||
                      std::is_same<T, bool>::value ||
                      std::is_same<T, monostate>::value,
                  "Not supported data type");
    if (std::is_same<T, std::string>::value) {
        return d.get_tag() == data_s::tag_type::str_t;
    } else if (std::is_same<T, long long int>::value) {
        return d.get_tag() == data_s::tag_type::int_t;
    } else if (std::is_same<T, double>::value) {
        return d.get_tag() == data_s::tag_type::double_t;
    } else if (std::is_same<T, bool>::value) {
        return d.get_tag() == data_s::tag_type::bool_t;
    } else if (std::is_same<T, monostate>::value) {
        return d.get_tag() == data_s::tag_type::empty_t;
    }
}


}  // namespace extension
}  // namespace gko

#endif  // GKO_PUBLIC_EXT_PROPERTY_TREE_DATA_HPP_
