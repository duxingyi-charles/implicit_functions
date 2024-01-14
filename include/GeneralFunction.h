#pragma once

#include "ImplicitFunction.h"
#include <functional>

template <typename Scalar>
class GeneralFunction : public ImplicitFunction<Scalar>
{
public:
    GeneralFunction(
        std::function<Scalar(Scalar, Scalar, Scalar, Scalar &, Scalar &, Scalar &)> func_grad) : func_grad_(func_grad) {}

    Scalar evaluate(Scalar x, Scalar y, Scalar z) const override
    {
        Scalar gx, gy, gz;
        return func_grad_(x, y, z, gx, gy, gz);
    }

    Scalar evaluate_gradient(Scalar x, Scalar y, Scalar z, Scalar &gx, Scalar &gy, Scalar &gz) const override
    {
        return func_grad_(x, y, z, gx, gy, gz);
    }

private:
    std::function<Scalar(Scalar, Scalar, Scalar, Scalar &, Scalar &, Scalar &)> func_grad_;
};