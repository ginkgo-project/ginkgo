// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_EXPORT_DEVICE_HPP_
#define GKO_PUBLIC_CORE_BASE_EXPORT_DEVICE_HPP_

#include <ginkgo/config.hpp>


// extract the necessary part from CMake's generate_export_header and adapt for
// different platform.
#if defined(GKO_DEVICE_STATIC_DEFINE) || GINKGO_EXPORT_ALL_SYMBOLS


#define GKO_DEVICE_EXPORT


#elif defined(_WIN32) || defined(__CYGWIN__)


#ifdef ginkgo_device_EXPORTS
/* We are building this library */
#define GKO_DEVICE_EXPORT __declspec(dllexport)
#else
/* We are using this library */
#define GKO_DEVICE_EXPORT __declspec(dllimport)
#endif


#else  // GCC/CLANG


#ifdef ginkgo_device_EXPORTS
/* We are building this library */
#define GKO_DEVICE_EXPORT __attribute__((visibility("default")))
#else
/* We are using this library */
#define GKO_DEVICE_EXPORT __attribute__((visibility("default")))
#endif


#endif


#endif  // GKO_PUBLIC_CORE_BASE_EXPORT_DEVICE_HPP_
