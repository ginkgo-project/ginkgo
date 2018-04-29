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

#ifndef GKO_CORE_BASE_ABSTRACT_FACTORY_HPP_
#define GKO_CORE_BASE_ABSTRACT_FACTORY_HPP_


#include "core/base/polymorphic_object.hpp"


namespace gko {


template <typename AbstractProductType, typename ComponentsType>
class AbstractFactory
    : public EnableAbstractPolymorphicObject<
          AbstractFactory<AbstractProductType, ComponentsType>> {
public:
    using abstrct_product_type = AbstractProductType;
    using components_type = ComponentsType;

    template <typename... Args>
    std::unique_ptr<AbstractProductType> generate(Args &&... args) const
    {
        return this->generate_impl({std::forward<Args>(args)...});
    }

protected:
    AbstractFactory(std::shared_ptr<const Executor> exec)
        : EnableAbstractPolymorphicObject<AbstractFactory>(std::move(exec))
    {}

    virtual std::unique_ptr<AbstractProductType> generate_impl(
        ComponentsType args) const = 0;
};


template <typename ProductType, typename AbstractProductType,
          typename ComponentsType, typename ParametersType>
class DefaultFactory
    : public EnablePolymorphicObject<
          DefaultFactory<ProductType, AbstractProductType, ComponentsType,
                         ParametersType>,
          AbstractFactory<AbstractProductType, ComponentsType>>,
      public EnablePolymorphicAssignment<DefaultFactory<
          ProductType, AbstractProductType, ComponentsType, ParametersType>>,
      public EnableCreateMethod<DefaultFactory<
          ProductType, AbstractProductType, ComponentsType, ParametersType>> {
    friend class EnablePolymorphicObject<
        DefaultFactory, AbstractFactory<AbstractProductType, ComponentsType>>;
    friend class EnableCreateMethod<DefaultFactory>;

public:
    using product_type = ProductType;
    using abstract_product_type = AbstractProductType;
    using components_type = ComponentsType;
    using parameters_type = ParametersType;

    template <typename... Args>
    std::unique_ptr<ProductType> generate(Args &&... args) const
    {
        return std::unique_ptr<ProductType>(static_cast<ProductType *>(
            this->generate_impl({std::forward<Args>(args)...}).release()));
    }

protected:
    template <typename... Args>
    explicit DefaultFactory(std::shared_ptr<const Executor> exec,
                            Args &&... args)
        : EnablePolymorphicObject<
              DefaultFactory,
              AbstractFactory<AbstractProductType, ComponentsType>>(
              std::move(exec)),
          parameters{std::forward<Args>(args)...}
    {}

    virtual std::unique_ptr<AbstractProductType> generate_impl(
        ComponentsType args) const
    {
        return new ProductType(this->get_executor(), parameters, args);
    }

private:
    ParametersType parameters;
};


}  // namespace gko


#endif  // GKO_CORE_BASE_ABSTRACT_FACTORY_HPP_
