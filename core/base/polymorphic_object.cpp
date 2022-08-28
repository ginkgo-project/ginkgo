/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <ginkgo/core/base/polymorphic_object.hpp>


namespace gko {


PolymorphicObject::~PolymorphicObject()
{
    this->template log<log::Logger::polymorphic_object_deleted>(exec_.get(),
                                                                this);
}


std::unique_ptr<PolymorphicObject> PolymorphicObject::create_default(
    std::shared_ptr<const Executor> exec) const
{
    this->template log<log::Logger::polymorphic_object_create_started>(
        exec_.get(), this);
    auto created = this->create_default_impl(std::move(exec));
    this->template log<log::Logger::polymorphic_object_create_completed>(
        exec_.get(), this, created.get());
    return created;
}


std::unique_ptr<PolymorphicObject> PolymorphicObject::create_default() const
{
    return this->create_default(exec_);
}


std::unique_ptr<PolymorphicObject> PolymorphicObject::clone(
    std::shared_ptr<const Executor> exec) const
{
    auto new_op = this->create_default(exec);
    new_op->copy_from(this);
    return new_op;
}


std::unique_ptr<PolymorphicObject> PolymorphicObject::clone() const
{
    return this->clone(exec_);
}


PolymorphicObject* PolymorphicObject::copy_from(const PolymorphicObject* other)
{
    this->template log<log::Logger::polymorphic_object_copy_started>(
        exec_.get(), other, this);
    auto copied = this->copy_from_impl(other);
    this->template log<log::Logger::polymorphic_object_copy_completed>(
        exec_.get(), other, this);
    return copied;
}


PolymorphicObject* PolymorphicObject::copy_from(
    std::unique_ptr<PolymorphicObject> other)
{
    this->template log<log::Logger::polymorphic_object_move_started>(
        exec_.get(), other.get(), this);
    auto copied = this->copy_from_impl(std::move(other));
    this->template log<log::Logger::polymorphic_object_move_completed>(
        exec_.get(), other.get(), this);
    return copied;
}


PolymorphicObject* PolymorphicObject::move_from(PolymorphicObject* other)
{
    this->template log<log::Logger::polymorphic_object_move_started>(
        exec_.get(), other, this);
    auto moved = this->move_from_impl(other);
    this->template log<log::Logger::polymorphic_object_move_completed>(
        exec_.get(), other, this);
    return moved;
}


PolymorphicObject* PolymorphicObject::move_from(
    std::unique_ptr<PolymorphicObject> other)
{
    this->template log<log::Logger::polymorphic_object_move_started>(
        exec_.get(), other.get(), this);
    auto copied = this->copy_from_impl(std::move(other));
    this->template log<log::Logger::polymorphic_object_move_completed>(
        exec_.get(), other.get(), this);
    return copied;
}


PolymorphicObject* PolymorphicObject::clear() { return this->clear_impl(); }


std::shared_ptr<const Executor> PolymorphicObject::get_executor() const noexcept
{
    return exec_;
}


PolymorphicObject::PolymorphicObject(std::shared_ptr<const Executor> exec)
    : exec_{std::move(exec)}
{}


PolymorphicObject::PolymorphicObject(const PolymorphicObject& other)
{
    // preserve the executor of the object
    *this = other;
}


PolymorphicObject& PolymorphicObject::operator=(const PolymorphicObject&)
{
    // preserve the executor of the object
    return *this;
}


}  // namespace gko
