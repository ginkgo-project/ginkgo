/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_CORE_STOP_COMBINED_HPP_
#define GKO_CORE_STOP_COMBINED_HPP_


#include "core/stop/criterion.hpp"


#include <vector>


namespace gko {
namespace stop {

/**
 * The Combined class is used to combine multiple criterions together through a
 * OR operation. The typical use case is to consider convergence has happened
 * when any one of multiple criterions are satisfied, e.g. a number of
 * iterations, the relative residual norm has reached a threshold, etc.
 */
class Combined : public Criterion {
public:
    struct Factory : public Criterion::Factory {
        using t = std::vector<std::unique_ptr<Criterion::Factory>>;

        /**
         * Instantiates a Combined::Factory object by using any number of
         * unique_ptrs to Criterion::Factory.
         * @param v  any number of unique_ptr to criterion factories
         *
         * @internal In order to allow any number of arguments to be passed, the
         * combined class relies on C++ Variadic Templates, or parameter packs.
         */
        template <class... V>
        Factory(V... v)
        {
            emplace(std::move(v)...);
        }

        /**
         * @internal Recursive Variadic Templated helper function to push
         * to the vector of unique_ptr to Criterion::Factory, v_ the element `V
         * v` of the pack, one by one. A recursion is called on the rest of the
         * templates, through the pack `R`.
         *
         * @tparam V  the first element of the variadic template (pack)
         * @tparam R  the rest of the pack
         * @param v  the first element which will be pushed back
         * @param rest  the rest of the pack which we recurse upon
         */
        template <class V, class... R>
        void emplace(V v, R... rest)
        {
            v_.emplace_back(std::move(v));
            emplace(std::move(rest)...);
        }

        /**
         * @internal the stopping condition for the recursion: when no arguments
         * are found in the pack "R", stop the recursion.
         */
        void emplace() {}

        template <class... V>
        static std::unique_ptr<Factory> create(V... v)
        {
            return std::unique_ptr<Factory>(new Factory(std::move(v)...));
        }

        std::unique_ptr<Criterion> create_criterion(
            std::shared_ptr<const LinOp> system_matrix,
            std::shared_ptr<const LinOp> b, const LinOp *x) const override;
        t v_{};
    };


    /**
     * Helper function which allows to add subcriterions in order to properly
     * build the Combined class with pointers to all subcriterions.
     *
     * @param c  the subcriterion to add to the Combined class
     */
    void add_subcriterion(std::unique_ptr<Criterion> c);

    bool check(uint8 stoppingId, bool setFinalized,
               Array<stopping_status> *stop_status, bool *one_changed,
               const Updater &) override;

private:
    std::vector<std::unique_ptr<Criterion>> criterions_{};
};


}  // namespace stop
}  // namespace gko


#endif  // GKO_CORE_STOP_COMBINED_HPP_
