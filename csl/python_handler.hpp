// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <stdexcept>

#include <Python.h>


class PythonObject {
public:
    // constructor
    PythonObject(PyObject* obj = nullptr) : obj_(obj) {}

    // copy constructor
    PythonObject(const PythonObject& other) : obj_(other.obj_)
    {
        Py_XINCREF(obj_);
    }

    // move constructor
    PythonObject(PythonObject&& other) noexcept : obj_(other.obj_)
    {
        other.obj_ = nullptr;
    }

    // destructor
    ~PythonObject() { Py_XDECREF(obj_); }

    // copy assignment
    PythonObject operator=(const PythonObject& other)
    {
        if (this != &other) {
            Py_XINCREF(other.obj_);
            Py_XDECREF(obj_);
            obj_ = other.obj_;
        }
        return *this;
    }

    // move assignment
    PythonObject operator=(PythonObject&& other) noexcept
    {
        if (this != &other) {
            Py_XDECREF(obj_);
            obj_ = other.obj_;
            other.obj_ = nullptr;
        }
        return *this;
    }

    // raw access
    PyObject* get() const { return obj_; }

    // dereference convenience
    PythonObject operator->() const { return obj_; }
    PythonObject operator*() const { return obj_; }

    // not operator
    bool operator!() const { return is_null(); }

    // check for nullptr
    bool is_null() const { return obj_ == nullptr; }

    // assert validity
    void assert_valid(const char* msg = "Invalid PyObject") const
    {
        if (!obj_) throw std::runtime_error(msg);
    }

private:
    PyObject* obj_;
};
