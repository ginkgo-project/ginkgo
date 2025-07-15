// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <cmath>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <vector>

#include <numpy/arrayobject.h>
#include <Python.h>

#include "pythonhandler.hpp"

#ifndef EXECDIR
#define EXECDIR "."
#endif


class CerebrasInterface {
public:
    CerebrasInterface(std::string module, bool use_simulator = false)
    {
        // start the interpreter
        Py_Initialize();

        // enhance sys.path
        PyRun_SimpleString("import sys; sys.path.append(\".\")");

        // load the module
        PythonObject pName(PyUnicode_FromString(module.c_str()));
        pModule_ = std::make_unique<PythonObject>(PyImport_Import(pName.get()));
        if (pModule_->is_null()) {
            PyErr_Print();
            exit(1);
        }

        // prepare numpy
        init_numpy();

        // load code onto the accelerator
        // INFO:    code is loaded from the path specified
        //          in "artifact.json"
        start_device(std::string(EXECDIR) + std::string("/artifact_path.json"),
                     "artifact_path", use_simulator);
        PyErr_Print();
    }

    ~CerebrasInterface()
    {
        stop_device();
        PyErr_Print();
        pModule_.reset();
        Py_Finalize();
    }

    void copy_h2d(std::string target_var, std::vector<float>& vec, int offset1,
                  int offset2, int size1, int size2, int elements_per_pe,
                  bool streaming, bool nonblocking)
    {
        auto npy_array = vector_to_numpy(vec);
        PythonObject pFunc(
            PyObject_GetAttrString(pModule_->get(), "copy_h2d_f32"));
        PythonObject pArgs(PyTuple_Pack(
            9, Py_BuildValue("s", target_var.c_str()), npy_array.get(),
            Py_BuildValue("i", offset1), Py_BuildValue("i", offset2),
            Py_BuildValue("i", size1), Py_BuildValue("i", size2),
            Py_BuildValue("i", elements_per_pe), PyBool_FromLong(streaming),
            PyBool_FromLong(nonblocking)));
        PyObject_CallObject(pFunc.get(), pArgs.get());
        PyErr_Print();
    }

    void copy_d2h(std::string target_var, std::vector<float>& vec, int offset1,
                  int offset2, int size1, int size2, int elements_per_pe,
                  bool streaming, bool nonblocking)
    {
        auto npy_array = vector_to_numpy(vec);
        PythonObject pFunc(
            PyObject_GetAttrString(pModule_->get(), "copy_d2h_f32"));
        PythonObject pArgs(PyTuple_Pack(
            9, Py_BuildValue("s", target_var.c_str()), npy_array.get(),
            Py_BuildValue("i", offset1), Py_BuildValue("i", offset2),
            Py_BuildValue("i", size1), Py_BuildValue("i", size2),
            Py_BuildValue("i", elements_per_pe), PyBool_FromLong(streaming),
            PyBool_FromLong(nonblocking)));
        PyObject_CallObject(pFunc.get(), pArgs.get());
        PyErr_Print();
    }

    void call_func(std::string func_name, bool nonblocking = false)
    {
        PythonObject pFunc(
            PyObject_GetAttrString(pModule_->get(), "call_cerebras_func"));
        PythonObject pArgs(PyTuple_Pack(2,
                                        Py_BuildValue("s", func_name.c_str()),
                                        PyBool_FromLong(nonblocking)));
        PyObject_CallObject(pFunc.get(), pArgs.get());
        PyErr_Print();
    }

private:
    std::unique_ptr<PythonObject> pModule_;

    // hacky numpy stuff - cannot be inlined as this is a macro that in
    // some cases expands to "return NULL;"
    void* init_numpy(void)
    {
        import_array();
        return NULL;
    }

    void numpy_to_vector(PythonObject& numpy_array, std::vector<float>& vector)
    {
        PyArrayObject* array =
            reinterpret_cast<PyArrayObject*>(numpy_array.get());
        float* data = static_cast<float*>(PyArray_DATA(array));
        vector.assign(data, data + PyArray_SIZE(array));
    }

    // TODO: does this return a view on the array or a copy of the array?
    // I believe view, but not sure...
    PythonObject vector_to_numpy(std::vector<float>& vector)
    {
        npy_intp dims[1] = {static_cast<npy_intp>(vector.size())};
        return PythonObject(PyArray_SimpleNewFromData(
            1, dims, NPY_FLOAT, const_cast<float*>(vector.data())));
    }

    void start_device(std::string artifact_json_path,
                      std::string artifact_json_key, bool use_simulator)
    {
        PythonObject pFunc(
            PyObject_GetAttrString(pModule_->get(), "start_device"));
        PythonObject pArgs(
            PyTuple_Pack(3, Py_BuildValue("s", artifact_json_path.c_str()),
                         Py_BuildValue("s", artifact_json_key.c_str()),
                         PyBool_FromLong(use_simulator)));
        PyObject_CallObject(pFunc.get(), pArgs.get());
    }

    void stop_device()
    {
        PythonObject pFunc(
            PyObject_GetAttrString(pModule_->get(), "stop_device"));
        PythonObject pArgs(PyTuple_Pack(0));
        PyObject_CallObject(pFunc.get(), pArgs.get());
    }
};
