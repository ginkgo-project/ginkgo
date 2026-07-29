// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_EXPORT_HPP_
#define GKO_PUBLIC_CORE_BASE_EXPORT_HPP_

// extract the necessary part from CMake's generate_export_header and adapt for
// different platform.
#ifdef GKO_STATIC_DEFINE


#define GKO_EXPORT


#elif defined(_WIN32) || defined(__CYGWIN__)


#ifdef ginkgo_EXPORTS
/* We are building this library */
#define GKO_EXPORT __declspec(dllexport)
#else
/* We are using this library */
#define GKO_EXPORT __declspec(dllimport)
#endif


#else  // GCC/CLANG


#ifdef ginkgo_EXPORTS
/* We are building this library */
#define GKO_EXPORT __attribute__((visibility("default")))
#else
/* We are using this library */
#define GKO_EXPORT __attribute__((visibility("default")))
#endif


#endif

// only have export class in GCC/CLANG
#if defined(_WIN32) || defined(__CYGWIN__)
#define GKO_EXPORT_CLASS
#else
#define GKO_EXPORT_CLASS GKO_EXPORT
#endif


#endif  // GKO_PUBLIC_CORE_BASE_EXPORT_HPP_
