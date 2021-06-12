/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#define ENUM_VALUE_(_name) _name
#define ENUM_VALUE_ASSIGN_(_name, _assign) _name = _assign
#define MACRO_OVERLOAD_(_1, _2, _NAME, ...) _NAME
#define ENUM_VALUE(...)                                                   \
    MACRO_OVERLOAD_(__VA_ARGS__, ENUM_VALUE_ASSIGN_, ENUM_VALUE_, UNUSED) \
    (__VA_ARGS__)


// clang-format off
#define ENUM_LAMBDA_(_name)                                            \
    {                                                                  \
        #_name, [&](arg_type item) {                                   \
            return this->build_item<enum_type, enum_type::_name>(item); \
        }                                                              \
    }
// clang-format on

#define ENUM_LAMBDA_ASSIGN_(_name, _assign) ENUM_LAMBDA_(_name)

#define ENUM_LAMBDA(...)                                                    \
    MACRO_OVERLOAD_(__VA_ARGS__, ENUM_LAMBDA_ASSIGN_, ENUM_LAMBDA_, UNUSED) \
    (__VA_ARGS__)

#define ENUM_CLASS(_enum_type, _type, _list) \
    enum class _enum_type : _type { _list(ENUM_VALUE) }

#define ENUM_MAP(_name, _enum_type, _return_type, _arg_type, _list, _keyword) \
    using enum_type = _enum_type;                                             \
    using arg_type = _arg_type;                                               \
    _keyword std::map<std::string, std::function<_return_type(arg_type)>>     \
        _name                                                                 \
    {                                                                         \
        _list(ENUM_LAMBDA)                                                    \
    }
