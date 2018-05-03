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


template <typename ConcreteFactory, typename ProductType,
          typename AbstractProductType, typename ComponentsType,
          typename ParametersType,
          typename PolymorphicBase =
              AbstractFactory<AbstractProductType, ComponentsType>>
class EnableDefaultFactory
    : public EnablePolymorphicObject<ConcreteFactory, PolymorphicBase>,
      public EnablePolymorphicAssignment<ConcreteFactory> {
    friend class EnablePolymorphicObject<ConcreteFactory, PolymorphicBase>;

public:
    using product_type = ProductType;
    using abstract_product_type = AbstractProductType;
    using components_type = ComponentsType;
    using parameters_type = ParametersType;
    using polymorphic_base = PolymorphicBase;

    template <typename... Args>
    std::unique_ptr<ProductType> generate(Args &&... args) const
    {
        return std::unique_ptr<ProductType>(static_cast<ProductType *>(
            this->generate_impl({std::forward<Args>(args)...}).release()));
    }

    const parameters_type &get_parameters() const { return parameters_; };

    static parameters_type create() { return {}; }

protected:
    explicit EnableDefaultFactory(std::shared_ptr<const Executor> exec,
                                  const parameters_type &parameters = {})
        : EnablePolymorphicObject<ConcreteFactory, PolymorphicBase>(
              std::move(exec)),
          parameters_{parameters}
    {}

    std::unique_ptr<AbstractProductType> generate_impl(
        ComponentsType args) const override
    {
        return std::unique_ptr<AbstractProductType>(
            new ProductType(self(), args));
    }

private:
    GKO_ENABLE_SELF(ConcreteFactory);

    ParametersType parameters_;
};


template <typename ConcreteParametersType, typename Factory>
struct parameters_type_base {
    std::unique_ptr<Factory> on_executor(
        std::shared_ptr<const Executor> exec) const
    {
        return std::unique_ptr<Factory>(new Factory(exec, *self()));
    }

protected:
    GKO_ENABLE_SELF(ConcreteParametersType);
};


}  // namespace gko


#endif  // GKO_CORE_BASE_ABSTRACT_FACTORY_HPP_
