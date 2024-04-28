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

#include "python.hpp"

namespace py = pybind11;

#define GKO_MATRIX_BINDING(Name)                                               \
    py::class_<gko::matrix::Name<ValueType>,                                   \
               std::shared_ptr<gko::matrix::Name<ValueType>>, gko::LinOp>(     \
        module_matrix, "Name", py::buffer_protocol())                          \
        .def(py::init([](std::shared_ptr<gko::Executor> exec) {                \
            return gko::matrix::Name<ValueType>::create(exec);                 \
        }))                                                                    \
        .def(py::init([](std::shared_ptr<gko::Executor> exec, py::tuple dim,   \
                         gko::array<ValueType>& vals,                          \
                         gko::array<IndexType>& cols,                          \
                         gko::array<IndexType>& rows) {                        \
            return gko::share(gko::matrix::Name<ValueType>::create(            \
                exec,                                                          \
                gko::dim<2>{dim[0].cast<size_t>(), dim[1].cast<size_t>()},     \
                vals, cols, rows));                                            \
        }))                                                                    \
        .def(py::init([](std::shared_ptr<gko::Executor> exec, py::tuple dim,   \
                         py::buffer data, py::buffer rows, py::buffer cols) {  \
            /* Request a buffer descriptor from Python */                      \
            py::buffer_info data_info = data.request();                        \
            py::buffer_info rows_info = rows.request();                        \
            py::buffer_info cols_info = cols.request();                        \
                                                                               \
            if (data_info.format !=                                            \
                py::format_descriptor<ValueType>::format())                    \
                throw std::runtime_error("Incompatible dtype");                \
                                                                               \
            if (rows_info.format !=                                            \
                py::format_descriptor<IndexType>::format())                    \
                throw std::runtime_error("Incompatible dtype");                \
                                                                               \
            if (cols_info.format !=                                            \
                py::format_descriptor<IndexType>::format())                    \
                throw std::runtime_error("Incompatible dtype");                \
                                                                               \
            auto ref = gko::ReferenceExecutor::create();                       \
                                                                               \
            /* create a view into numpy data */                                \
            auto nnz = data_info.shape[0];                                     \
                                                                               \
            auto data_view =                                                   \
                gko::array<ValueType>(ref, nnz, (ValueType*)data_info.ptr);    \
            auto rows_view =                                                   \
                gko::array<IndexType>(ref, nnz, (IndexType*)rows_info.ptr);    \
            auto cols_view =                                                   \
                gko::array<IndexType>(ref, nnz, (IndexType*)cols_info.ptr);    \
                                                                               \
            return gko::share(gko::matrix::Name<ValueType>::create(            \
                exec,                                                          \
                gko::dim<2>{dim[0].cast<size_t>(), dim[1].cast<size_t>()},     \
                data_view, cols_view, rows_view));                             \
        }))                                                                    \
        .def("apply",                                                          \
             py::overload_cast<const gko::LinOp*, gko::LinOp*>(                \
                 &gko::matrix::Name<ValueType, IndexType>::apply),             \
             "")                                                               \
        .def("apply",                                                          \
             py::overload_cast<const gko::LinOp*, gko::LinOp*>(                \
                 &gko::matrix::Name<ValueType, IndexType>::apply, py::const_), \
             "")                                                               \
        .def("apply",                                                          \
             py::overload_cast<const gko::LinOp*, const gko::LinOp*,           \
                               const gko::LinOp*, gko::LinOp*>(                \
                 &gko::matrix::Name<ValueType, IndexType>::apply, py::const_), \
             "")                                                               \
        .def("__repr__",                                                       \
             [](const gko::matrix::Name<ValueType>& o) {                       \
                 auto str = std::string("pygko.matrix.Name object");           \
                 return str;                                                   \
             })                                                                \
        .def(                                                                  \
            "get_num_stored_elements",                                         \
            &gko::matrix::Name<ValueType>::get_num_stored_elements,            \
            "Applies Name matrix axpy to a vector (or a sequence of vectors)." \
            "Performs the operation x = Name * b + x")

// void init_coo(py::module_& module_matrix) { GKO_MATRIX_BINDING(Coo); }
