#pragma once

#if IMPLICIT_FUNCTIONS_WITH_SHADER_SUPPORT

#include "ImplicitFunction.h"
#include <implicit_shader/Application.h>

#include <utility>
#include <type_traits>
#include <array>

template <typename Scalar>
class ImplicitShader : public ImplicitFunction<Scalar> {
public:
    ImplicitShader(std::string name, Scalar delta) : name_(std::move(name)), delta_(delta),
                                                           func_app_(), func_grad_app_() {
        func_app_.onInit(SHADER_DIR "/" + name_ + ".wgsl", 1);
        func_grad_app_.onInit(SHADER_DIR "/" + name_ + ".wgsl", 7);
    }

    [[nodiscard]] Scalar evaluate(Scalar x, Scalar y, Scalar z) const override {
        std::array<Scalar, 4> buffer{x, y, z, 0};
        if constexpr (std::is_same<Scalar, float>::value) {
            func_app_.onCompute(buffer);
        } else {
            std::array<float, 4> buffer_float;
            for (int i = 0; i < 4; ++i) {
                buffer_float[i] = static_cast<float>(buffer[i]);
            }
            func_app_.onCompute(buffer_float);
            buffer[3] = static_cast<Scalar>(buffer_float[3]);
        }
        return buffer[3];
    }

    Scalar evaluate_gradient(Scalar x, Scalar y, Scalar z, Scalar &gx, Scalar &gy, Scalar &gz) const override {
        std::array<Scalar, 28> buffer{x, y, z, 0,
                                  x - delta_, y, z, 0,
                                  x + delta_, y, z, 0,
                                  x, y - delta_, z, 0,
                                  x, y + delta_, z, 0,
                                  x, y, z - delta_, 0,
                                  x, y, z + delta_, 0};
        if constexpr (std::is_same<Scalar, float>::value) {
            func_grad_app_.onCompute(buffer);
        } else {
            std::array<float, 28> buffer_float;
            for (int i = 0; i < 28; ++i) {
                buffer_float[i] = static_cast<float>(buffer[i]);
            }
            func_grad_app_.onCompute(buffer_float);
            for (int i = 3; i < 28; i+=4) {
                buffer[i] = static_cast<Scalar>(buffer_float[i]);
            }
        }
        gx = (buffer[11] - buffer[7]) / (2 * delta_);
        gy = (buffer[19] - buffer[15]) / (2 * delta_);
        gz = (buffer[27] - buffer[23]) / (2 * delta_);
        return buffer[3];
    }

    ~ImplicitShader() override {
        func_app_.onFinish();
        func_grad_app_.onFinish();
    }

private:
    mutable implicit_shader::Application func_app_;
    mutable implicit_shader::Application func_grad_app_;
    std::string name_;
    // delta used for finite difference
    Scalar delta_;
};

#endif
