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

#include <pybind11/pybind11.h>
#include "ginkgo/ginkgo.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pygko, m)
{
    m.doc() = "Python bindings for the Ginkgo framework";

    py::class_<gko::Executor, std::shared_ptr<gko::Executor>> Executor(
        m, "Executor");

    py::class_<gko::OmpExecutor, std::shared_ptr<gko::OmpExecutor>>(
        m, "OmpExecutor", Executor)
        .def(py::init(&gko::OmpExecutor::create));

    py::class_<gko::ReferenceExecutor, std::shared_ptr<gko::ReferenceExecutor>>(
        m, "ReferenceExecutor", Executor)
        .def(py::init(&gko::ReferenceExecutor::create));

    py::class_<gko::array<double>>(m, "array")
        .def(py::init<std::shared_ptr<const gko::Executor>, int>())
        .def("fill", &gko::array<double>::fill,
             "Fill the array with the given value.")
        .def("get_num_elems", &gko::array<double>::get_num_elems);

    py::class_<gko::dim<2>>(m, "dim2").def(py::init<int, int>());

    py::class_<gko::LinOp, std::shared_ptr<gko::LinOp>> LinOp(m, "LinOp");

    // TODO wrap this for other data types like float, int ...
    py::class_<gko::matrix::Dense<double>,
               std::shared_ptr<gko::matrix::Dense<double>>, gko::LinOp>(
        m, "Dense", py::buffer_protocol())
        .def(py::init([](std::shared_ptr<gko::Executor> exec, py::buffer b) {
            /* Request a buffer descriptor from Python */
            py::buffer_info info = b.request();
            auto ref = gko::ReferenceExecutor::create();

            /* create a view into numpy data */
            auto elems = (info.ndim == 1) ? info.shape[0]
                                          : info.shape[0] * info.shape[1];

            auto view = gko::array<double>(ref, elems, (double*)info.ptr);

            // TODO fix dim<2>
            // TODO fix stride since the stride is given in bytes on the numpy
            // side
            return gko::matrix::Dense<double>::create(
                exec, gko::dim<2>{info.shape[0], 1}, view, 1);
        }))
        .def(py::init([](std::shared_ptr<gko::Executor> exec) {
            return gko::matrix::Dense<double>::create(exec);
        }))
        .def("__repr__",
             [](const gko::matrix::Dense<double>& o) {
                 auto str = std::string("pygko.matrix.Dense object of size ");
                 str += std::to_string(o.get_num_stored_elements());
                 return str;
             })
        .def_buffer([](gko::matrix::Dense<double>& m) -> py::buffer_info {
            size_t rows = m.get_num_stored_elements() / m.get_stride();
            size_t cols = m.get_stride();
            size_t dim = (m.get_stride() == 1) ? 1 : 2;

            // TODO implement for 2D matrix
            return py::buffer_info(
                m.get_values(), /* Pointer to buffer */
                sizeof(double), /* Size of one scalar */
                py::format_descriptor<double>::format(), /* Python struct-style
                                                            format descriptor */
                dim,             /* Number of dimensions */
                {rows},          /* Buffer dimensions */
                {sizeof(double)} /* Strides (in bytes) for each index */
            );
        })
        .def("get_stride", &gko::matrix::Dense<double>::get_stride,
             "Returns the stride of the matrix.")
        .def("scale", &gko::matrix::Dense<double>::scale,
             "Scales the matrix with a scalar (aka: BLAS scal).")
        .def("inv_scale", &gko::matrix::Dense<double>::inv_scale,
             "Scales the matrix with the inverse of a scalar.")
        .def("add_scaled", &gko::matrix::Dense<double>::add_scaled,
             "Adds `b` scaled by `alpha` to the matrix (aka: BLAS axpy).")
        .def(
            "sub_scaled", &gko::matrix::Dense<double>::sub_scaled,
            "Subtracts `b` scaled by `alpha` fron the matrix (aka: BLAS axpy).")
        .def("at",
             static_cast<double& (gko::matrix::Dense<double>::*)(size_t)>(
                 &gko::matrix::Dense<double>::at),
             "Returns an element using linearized index.")
        .def("at",
             static_cast<double& (gko::matrix::Dense<double>::*)(size_t,
                                                                 size_t)>(
                 &gko::matrix::Dense<double>::at),
             "Returns an element at row, column index.")
        .def("get_num_stored_elements",
             &gko::matrix::Dense<double>::get_num_stored_elements,
             "Returns the number of elements explicitly stored in the "
             "matrix.");
}
