// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_CONFIG_WRITE_CONFIG_HPP_
#define GKO_PUBLIC_CORE_CONFIG_WRITE_CONFIG_HPP_


#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/config/property_tree.hpp>
#include <ginkgo/core/stop/criterion.hpp>


namespace gko {
namespace config {


/**
 * Writes the parameters of a LinOpFactory as the property tree that parse
 * reads, so that parsing it creates a factory with the same parameters.
 *
 * The index type is not written, since parse takes it from its
 * type_descriptor, and neither are function-valued parameters like the
 * selectors of Multigrid, which parse does not read. LinOp parameters are
 * written as the placeholder "<object>".
 *
 * @param factory  the factory to write
 *
 * @return the configuration of the factory
 *
 * @throw NotSupported  if the factory or one of its nested factories has no
 *                      writer, or has a parameter that parse can not read
 */
pnode write_config(const LinOpFactory& factory);


/**
 * Writes the parameters of a stopping criterion factory as the property tree
 * that parse reads for an entry of "criteria". A combined criterion is written
 * as the array of the criteria it combines.
 *
 * @param factory  the criterion factory to write
 *
 * @return the configuration of the criterion factory
 *
 * @throw NotSupported  if the type of the factory has no writer
 */
pnode write_config(const stop::CriterionFactory& factory);


}  // namespace config
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_CONFIG_WRITE_CONFIG_HPP_
