// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GINKGO_EXPORT_H
#define GINKGO_EXPORT_H

// modified from cmake generate header
#ifdef GINKGO_STATIC_DEFINE
#define GINKGO_EXPORT
#define GINKGO_NO_EXPORT
#else
#ifndef GINKGO_EXPORT
#ifdef ginkgo_EXPORTS
/* We are building this library */
#if defined _WIN32 || defined __CYGWIN__ || defined __MSYS__
#define GINKGO_EXPORT __declspec(dllexport)
#else
#define GINKGO_EXPORT __attribute__((visibility("default")))
#endif
#else
/* We are using this library */
#if defined _WIN32 || defined __CYGWIN__ || defined __MSYS__
#define GINKGO_EXPORT __declspec(dllimport)
#else
#define GINKGO_EXPORT __attribute__((visibility("default")))
#endif
#endif

#ifndef GINKGO_NO_EXPORT
#define GINKGO_NO_EXPORT __attribute__((visibility("hidden")))
#endif
#endif

#ifndef GINKGO_DEPRECATED
#define GINKGO_DEPRECATED __attribute__((__deprecated__))
#endif

#ifndef GINKGO_DEPRECATED_EXPORT
#define GINKGO_DEPRECATED_EXPORT GINKGO_EXPORT GINKGO_DEPRECATED
#endif

#ifndef GINKGO_DEPRECATED_NO_EXPORT
#define GINKGO_DEPRECATED_NO_EXPORT GINKGO_NO_EXPORT GINKGO_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#ifndef GINKGO_NO_DEPRECATED
#define GINKGO_NO_DEPRECATED
#endif
#endif

#endif /* GINKGO_EXPORT_H */
