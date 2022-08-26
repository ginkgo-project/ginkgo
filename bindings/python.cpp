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

std::shared_ptr<gko::matrix::Dense<double>> createDense(
    std::shared_ptr<gko::Executor> exec, gko::dim<2> dim)
{
    return gko::matrix::Dense<double>::create(exec, dim);
}


PYBIND11_MODULE(pygko, m)
{
    m.doc() = "Python bindings for the Ginkgo framework";

    py::class_<gko::Executor, std::shared_ptr<gko::Executor>> Executor(
        m, "Executor");

    m.def("createDense", &createDense, "A function that adds two numbers");


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

    py::class_<
        gko::EnableCreateMethod<gko::matrix::Dense<double>>,
        std::shared_ptr<gko::EnableCreateMethod<gko::matrix::Dense<double>>>>
        EnableCreateMethod(m, "EnableCreateMethod");

    py::class_<
        gko::ConvertibleTo<gko::matrix::Dense<gko::next_precision<double>>>,
        std::shared_ptr<gko::ConvertibleTo<
            gko::matrix::Dense<gko::next_precision<double>>>>>
        ConvertibleTo(m, "ConvertibleTo");

    py::class_<
        gko::matrix::Dense<double>, std::shared_ptr<gko::matrix::Dense<double>>,
        gko::LinOp,
        gko::ConvertibleTo<gko::matrix::Dense<gko::next_precision<double>>>,
        gko::EnableCreateMethod<gko::matrix::Dense<double>>>(m, "Dense")
        // .def(py::init<std::shared_ptr<gko::Executor>, gko::dim<2>>(
        //     &gko::matrix::Dense<double>::create))
        .def("scale", &gko::matrix::Dense<double>::scale,
             "Scales the matrix with a scalar (aka: BLAS scal).")
        .def("inv_scale", &gko::matrix::Dense<double>::inv_scale,
             "Scales the matrix with the inverse of a scalar.")
        .def("add_scaled", &gko::matrix::Dense<double>::add_scaled,
             "Adds `b` scaled by `alpha` to the matrix (aka: BLAS axpy).")
        .def(
            "sub_scaled", &gko::matrix::Dense<double>::sub_scaled,
            "Subtracts `b` scaled by `alpha` fron the matrix (aka: BLAS axpy).")
        // .def("compute_dot", &gko::matrix::Dense<double>::compute_dot,
        //      "Computes the column-wise dot product of this matrix and `b`.")
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
