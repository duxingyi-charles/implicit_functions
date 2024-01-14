#pragma once

#include <string>
#include <Eigen/Core>
#include "ImplicitFunction.h"
#include "GeneralFunction.h"
#include "primitives.h"
//#include "Hermite_RBF.h"

bool load_functions(const std::string &filename,
                    const std::vector<std::array<double, 3>> &pts,
                    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &funcVals);

bool load_functions(const std::string &filename, std::vector<std::unique_ptr<ImplicitFunction<double>>> &functions);