// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "cerebras_handle.hpp"

#include <cmath>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>

#include <numpy/arrayobject.h>
#include <Python.h>

#include "python_inner_functions.hpp"


CerebrasHandle::CerebrasHandle(bool use_simulator)
{
    // start the interpreter
    Py_Initialize();

    // load the python module from string
    PyRun_SimpleString(PYTHON_CSL_INNER_FUNCTIONS);

    // load the module
    PythonObject pLocalModule(PyImport_AddModule("__main__"));
    pFuncDict_ =
        std::make_unique<PythonObject>(PyModule_GetDict(pLocalModule.get()));
    if (pFuncDict_->is_null()) {
        PyErr_Print();
        exit(1);
    }

    // prepare numpy
    init_numpy();

    start_device(use_simulator);
    PyErr_Print();
}

void CerebrasHandle::destroy()
{
    stop_device();
    PyErr_Print();
    pFuncDict_.reset();
    Py_Finalize();
}

void CerebrasHandle::copy_h2d(std::string target_var, float* vec,
                              size_t vec_size, int offset1, int offset2,
                              int size1, int size2, int elements_per_pe,
                              bool streaming, bool nonblocking)
{
    auto npy_array = array_to_numpy(vec, vec_size);
    PythonObject pFunc(PyDict_GetItemString(pFuncDict_->get(), "copy_h2d_f32"));
    if (!pFunc || !PyCallable_Check(pFunc.get())) {
        PyErr_Print();
        exit(1);
    }
    PythonObject pArgs(
        PyTuple_Pack(9, Py_BuildValue("s", target_var.c_str()), npy_array.get(),
                     Py_BuildValue("i", offset1), Py_BuildValue("i", offset2),
                     Py_BuildValue("i", size1), Py_BuildValue("i", size2),
                     Py_BuildValue("i", elements_per_pe),
                     PyBool_FromLong(streaming), PyBool_FromLong(nonblocking)));
    PyObject_CallObject(pFunc.get(), pArgs.get());
    PyErr_Print();
}

void CerebrasHandle::copy_d2h(std::string target_var, float* vec,
                              size_t vec_size, int offset1, int offset2,
                              int size1, int size2, int elements_per_pe,
                              bool streaming, bool nonblocking)
{
    auto npy_array = array_to_numpy(vec, vec_size);
    PythonObject pFunc(PyDict_GetItemString(pFuncDict_->get(), "copy_d2h_f32"));
    if (!pFunc || !PyCallable_Check(pFunc.get())) {
        PyErr_Print();
        exit(1);
    }
    PythonObject pArgs(
        PyTuple_Pack(9, Py_BuildValue("s", target_var.c_str()), npy_array.get(),
                     Py_BuildValue("i", offset1), Py_BuildValue("i", offset2),
                     Py_BuildValue("i", size1), Py_BuildValue("i", size2),
                     Py_BuildValue("i", elements_per_pe),
                     PyBool_FromLong(streaming), PyBool_FromLong(nonblocking)));
    PyObject_CallObject(pFunc.get(), pArgs.get());
    PyErr_Print();
}

void CerebrasHandle::call_func(std::string func_name, bool nonblocking)
{
    PythonObject pFunc(
        PyDict_GetItemString(pFuncDict_->get(), "call_cerebras_func"));
    if (!pFunc || !PyCallable_Check(pFunc.get())) {
        PyErr_Print();
        exit(1);
    }
    PythonObject pArgs(PyTuple_Pack(2, Py_BuildValue("s", func_name.c_str()),
                                    PyBool_FromLong(nonblocking)));
    PyObject_CallObject(pFunc.get(), pArgs.get());
    PyErr_Print();
}

// hacky numpy stuff - cannot be inlined as this is a macro that in
// some cases expands to "return NULL;"
void* CerebrasHandle::init_numpy(void)
{
    import_array();
    return NULL;
}

// TODO: does this return a view on the array or a copy of the array?
// I believe view, but not sure...
PythonObject CerebrasHandle::array_to_numpy(float* vec, size_t length)
{
    npy_intp dims[1] = {static_cast<npy_intp>(length)};
    return PythonObject(
        PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, const_cast<float*>(vec)));
}

void CerebrasHandle::start_device(bool use_simulator)
{
    PythonObject pFunc(PyDict_GetItemString(pFuncDict_->get(), "start_device"));
    PythonObject pArgs(PyTuple_Pack(1, PyBool_FromLong(use_simulator)));
    PyObject_CallObject(pFunc.get(), pArgs.get());
}

void CerebrasHandle::stop_device()
{
    PythonObject pFunc(PyDict_GetItemString(pFuncDict_->get(), "stop_device"));
    PythonObject pArgs(PyTuple_Pack(0));
    PyObject_CallObject(pFunc.get(), pArgs.get());
}
