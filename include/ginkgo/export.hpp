// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_EXPORT_HPP_
#define GKO_PUBLIC_EXPORT_HPP_

#include <ginkgo/export_.hpp>

#ifdef _MSC_VER
#define GKO_EXPORT_CLASS
#else
#define GKO_EXPORT_CLASS GKO_EXPORT
#endif


#endif  // GKO_PUBLIC_EXPORT_HPP_
