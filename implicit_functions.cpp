#include "implicit_functions.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <cmath>

typedef std::function<double(double, double, double, double &, double &, double &)> FuncGrad;

bool load_functions(const std::string &filename,
                    const std::vector<std::array<double, 3>> &pts,
                    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &funcVals)
{
    using json = nlohmann::json;
    std::ifstream fin(filename.c_str());
    if (!fin)
    {
        std::cout << "function file not exist!" << std::endl;
        return false;
    }
    json data;
    fin >> data;
    fin.close();
    //
    size_t n_pts = pts.size();
    size_t n_func = data.size();
    funcVals.resize(n_pts, n_func);
    for (int j = 0; j < n_func; ++j)
    {
        std::string type = data[j]["type"].get<std::string>();
        if (type == "plane")
        {
            std::array<double, 3> point;
            for (int i = 0; i < 3; ++i)
            {
                point[i] = data[j]["point"][i].get<double>();
            }
            std::array<double, 3> normal;
            for (int i = 0; i < 3; ++i)
            {
                normal[i] = data[j]["normal"][i].get<double>();
            }
            //
            PlaneDistanceFunction<double> plane(point, normal);
            for (int i = 0; i < n_pts; i++)
            {
                funcVals(i, j) = plane.evaluate(pts[i][0], pts[i][1], pts[i][2]);
            }
        }
        else if (type == "line")
        {
            std::array<double, 3> point;
            for (int i = 0; i < 3; ++i)
            {
                point[i] = data[j]["point"][i].get<double>();
            }
            std::array<double, 3> unit_vector;
            for (int i = 0; i < 3; ++i)
            {
                unit_vector[i] = data[j]["unit_vector"][i].get<double>();
            }
            //
            CylinderDistanceFunction<double> line(point, unit_vector, 0);
            for (int i = 0; i < n_pts; i++)
            {
                funcVals(i, j) = line.evaluate(pts[i][0], pts[i][1], pts[i][2]);
            }
        }
        else if (type == "cylinder")
        {
            std::array<double, 3> axis_point;
            for (int i = 0; i < 3; ++i)
            {
                axis_point[i] = data[j]["axis_point"][i].get<double>();
            }
            std::array<double, 3> axis_unit_vector;
            for (int i = 0; i < 3; ++i)
            {
                axis_unit_vector[i] = data[j]["axis_vector"][i].get<double>();
            }
            double radius = data[j]["radius"].get<double>();
            //
            CylinderDistanceFunction<double> cylinder(axis_point, axis_unit_vector, radius);
            for (int i = 0; i < n_pts; i++)
            {
                funcVals(i, j) = cylinder.evaluate(pts[i][0], pts[i][1], pts[i][2]);
            }
        }
        else if (type == "sphere")
        {
            std::array<double, 3> center;
            for (int i = 0; i < 3; ++i)
            {
                center[i] = data[j]["center"][i].get<double>();
            }
            double radius = data[j]["radius"].get<double>();
            //
            std::unique_ptr<ImplicitFunction<double>> sphere;
            if (radius >= 0)
            {
                sphere = std::make_unique<SphereDistanceFunction<double>>(center, radius);
            }
            else
            {
                sphere = std::make_unique<SphereUnsignedDistanceFunction<double>>(center, -radius);
            }
            for (int i = 0; i < n_pts; i++)
            {
                funcVals(i, j) = sphere->evaluate(pts[i][0], pts[i][1], pts[i][2]);
            }
        }
        else if (type == "torus")
        {
            std::array<double, 3> center;
            for (int i = 0; i < 3; ++i)
            {
                center[i] = data[j]["center"][i].get<double>();
            }
            std::array<double, 3> axis_unit_vector;
            for (int i = 0; i < 3; ++i)
            {
                axis_unit_vector[i] = data[j]["axis_vector"][i].get<double>();
            }
            double major_radius = data[j]["major_radius"].get<double>();
            double minor_radius = data[j]["minor_radius"].get<double>();
            //
            TorusDistanceFunction<double> torus(center, axis_unit_vector, major_radius, minor_radius);
            for (int i = 0; i < n_pts; i++)
            {
                funcVals(i, j) = torus.evaluate(pts[i][0], pts[i][1], pts[i][2]);
            }
        }
        else if (type == "circle")
        {
            std::array<double, 3> center;
            for (int i = 0; i < 3; ++i)
            {
                center[i] = data[j]["center"][i].get<double>();
            }
            std::array<double, 3> axis_unit_vector;
            for (int i = 0; i < 3; ++i)
            {
                axis_unit_vector[i] = data[j]["axis_vector"][i].get<double>();
            }
            double radius = data[j]["radius"].get<double>();
            //
            TorusDistanceFunction<double> circle(center, axis_unit_vector, radius, 0);
            for (int i = 0; i < n_pts; i++)
            {
                funcVals(i, j) = circle.evaluate(pts[i][0], pts[i][1], pts[i][2]);
            }
        }
        else if (type == "cone")
        {
            std::array<double, 3> apex;
            for (int i = 0; i < 3; ++i)
            {
                apex[i] = data[j]["apex"][i].get<double>();
            }
            std::array<double, 3> axis_unit_vector;
            for (int i = 0; i < 3; ++i)
            {
                axis_unit_vector[i] = data[j]["axis_vector"][i].get<double>();
            }
            double apex_angle = data[j]["apex_angle"].get<double>();
            //
            ConeDistanceFunction<double> cone(apex, axis_unit_vector, apex_angle);
            for (int i = 0; i < n_pts; i++)
            {
                funcVals(i, j) = cone.evaluate(pts[i][0], pts[i][1], pts[i][2]);
            }
        }
        else if (type == "zero")
        {
            for (int i = 0; i < n_pts; i++)
            {
                funcVals(i, j) = 0;
            }
        }
        else
        {
            std::cout << "undefined type: " << type << std::endl;
            return false;
        }
    }
    return true;
}

bool load_functions(const std::string &filename, std::vector<std::unique_ptr<ImplicitFunction<double>>> &functions)
{
    using json = nlohmann::json;
    std::ifstream fin(filename.c_str());
    if (!fin)
    {
        std::cout << "function file not exist!" << std::endl;
        return false;
    }
    json data;
    fin >> data;
    fin.close();
    //
    size_t n_func = data.size();
    functions.resize(n_func);
    for (int j = 0; j < n_func; ++j)
    {
        std::string type = data[j]["type"].get<std::string>();
        if (type == "plane")
        {
            std::array<double, 3> point;
            for (int i = 0; i < 3; ++i)
            {
                point[i] = data[j]["point"][i].get<double>();
            }
            std::array<double, 3> normal;
            for (int i = 0; i < 3; ++i)
            {
                normal[i] = data[j]["normal"][i].get<double>();
            }
            //
            functions[j] = std::make_unique<PlaneDistanceFunction<double>>(point, normal);
        }
        else if (type == "line")
        {
            std::array<double, 3> point;
            for (int i = 0; i < 3; ++i)
            {
                point[i] = data[j]["point"][i].get<double>();
            }
            std::array<double, 3> unit_vector;
            for (int i = 0; i < 3; ++i)
            {
                unit_vector[i] = data[j]["unit_vector"][i].get<double>();
            }
            bool squared = false;
            if (data[j].find("squared") != data[j].end())
            {
                squared = data[j]["squared"].get<bool>();
            }
            //
            if (squared)
                functions[j] = std::make_unique<CylinderSquaredDistanceFunction<double>>(point, unit_vector, 0);
            else
                functions[j] = std::make_unique<CylinderDistanceFunction<double>>(point, unit_vector, 0);
        }
        else if (type == "cylinder")
        {
            std::array<double, 3> axis_point;
            for (int i = 0; i < 3; ++i)
            {
                axis_point[i] = data[j]["axis_point"][i].get<double>();
            }
            std::array<double, 3> axis_unit_vector;
            for (int i = 0; i < 3; ++i)
            {
                axis_unit_vector[i] = data[j]["axis_vector"][i].get<double>();
            }
            double radius = data[j]["radius"].get<double>();
            bool squared = false;
            if (data[j].find("squared") != data[j].end())
            {
                squared = data[j]["squared"].get<bool>();
            }
            //
            if (squared)
                functions[j] = std::make_unique<CylinderSquaredDistanceFunction<double>>(axis_point, axis_unit_vector, radius);
            else
                functions[j] = std::make_unique<CylinderDistanceFunction<double>>(axis_point, axis_unit_vector, radius);
        }
        else if (type == "sphere")
        {
            std::array<double, 3> center;
            for (int i = 0; i < 3; ++i)
            {
                center[i] = data[j]["center"][i].get<double>();
            }
            double radius = data[j]["radius"].get<double>();
            bool squared = false;
            if (data[j].find("squared") != data[j].end())
            {
                squared = data[j]["squared"].get<bool>();
            }
            //
            if (squared)
            {
                functions[j] = std::make_unique<SphereSquaredDistanceFunction<double>>(center, radius);
            }
            else
            {
                if (radius >= 0)
                {
                    functions[j] = std::make_unique<SphereDistanceFunction<double>>(center, radius);
                }
                else
                {
                    functions[j] = std::make_unique<SphereUnsignedDistanceFunction<double>>(center, -radius);
                }
            }
        }
        else if (type == "torus")
        {
            std::array<double, 3> center;
            for (int i = 0; i < 3; ++i)
            {
                center[i] = data[j]["center"][i].get<double>();
            }
            std::array<double, 3> axis_unit_vector;
            for (int i = 0; i < 3; ++i)
            {
                axis_unit_vector[i] = data[j]["axis_vector"][i].get<double>();
            }
            double major_radius = data[j]["major_radius"].get<double>();
            double minor_radius = data[j]["minor_radius"].get<double>();
            //
            functions[j] = std::make_unique<TorusDistanceFunction<double>>(center, axis_unit_vector, major_radius, minor_radius);
        }
        else if (type == "circle")
        {
            std::array<double, 3> center;
            for (int i = 0; i < 3; ++i)
            {
                center[i] = data[j]["center"][i].get<double>();
            }
            std::array<double, 3> axis_unit_vector;
            for (int i = 0; i < 3; ++i)
            {
                axis_unit_vector[i] = data[j]["axis_vector"][i].get<double>();
            }
            double radius = data[j]["radius"].get<double>();
            //
            functions[j] = std::make_unique<TorusDistanceFunction<double>>(center, axis_unit_vector, radius, 0);
        }
        else if (type == "cone")
        {
            std::array<double, 3> apex;
            for (int i = 0; i < 3; ++i)
            {
                apex[i] = data[j]["apex"][i].get<double>();
            }
            std::array<double, 3> axis_unit_vector;
            for (int i = 0; i < 3; ++i)
            {
                axis_unit_vector[i] = data[j]["axis_vector"][i].get<double>();
            }
            double apex_angle = data[j]["apex_angle"].get<double>();
            //
            functions[j] = std::make_unique<ConeDistanceFunction<double>>(apex, axis_unit_vector, apex_angle);
        }
        else if (type == "wave")
        {
            FuncGrad f = [](double x, double y, double z,
                            double &gx, double &gy, double &gz)
            {
                double val = 2 - 4 * z + 0.3333333333333333 * (sin(15.343274134071653 - 9.21125890435288 * x - 18.2340614919869 * y) + sin(4.801594243303487 + 7.985828093490334 * x - 10.314294288432725 * y) - sin(6.54696301939958 - 18.46419127184428 * x - 1.2648908198491549 * y));
                gx = -3.070419634784293 * cos(15.343274134071653 - 9.21125890435288 * x -
                                              18.2340614919869 * y) +
                2.6619426978301113 *
                cos(4.801594243303487 + 7.985828093490334 * x - 10.314294288432725 * y) +
                6.154730423948093 * cos(6.54696301939958 - 18.46419127184428 * x - 1.2648908198491549 * y);
                gy = -6.078020497328966 * cos(15.343274134071653 - 9.21125890435288 * x - 18.2340614919869 * y) -
                3.4380980961442416 * cos(4.801594243303487 + 7.985828093490334 * x -
                                         10.314294288432725 * y) +
                0.4216302732830516 *
                cos(6.54696301939958 - 18.46419127184428 * x - 1.2648908198491549 * y);
                gz = -4;
                return val;
            };
            functions[j] = std::make_unique<GeneralFunction<double>>(f);
        }
        else if (type == "Gaussian")
        {
            FuncGrad f = [](double x, double y, double z,
                            double &gx, double &gy, double &gz)
            {
                double e_power = exp(-21.33333333333333 * x * x - 21.33333333333333 * y * y +
                                     (-7.105427357601002e-15 - 21.33333333333333 * z) * z +
                                     y * (-7.105427357601002e-15 + 21.33333333333334 * z) +
                                     x * (-7.105427357601002e-15 + 21.33333333333334 * y + 21.33333333333334 * z));
                double val = 3.4641016151377553 + 2.0000000000000213 * e_power -
                2.3094010767585034 * x - 2.3094010767585034 * y - 2.3094010767585034 * z;
                gx = -2.3094010767585034 - 85.33333333333422 *
                e_power * (1.6653345369377353e-16 + x - 0.5000000000000002 * y - 0.5000000000000002 * z);
                gy = -2.3094010767585034 + 42.66666666666713 *
                e_power * (-3.3306690738754686e-16 + x - 1.9999999999999991 * y + z);
                gz = -2.3094010767585034 + 42.66666666666713 *
                e_power * (-3.3306690738754686e-16 + x + y - 1.9999999999999991 * z);
                return val;
            };
            functions[j] = std::make_unique<GeneralFunction<double>>(f);
        }
        else if (type == "smile")
        {
            FuncGrad f = [](double x, double y, double z,
                            double &gx, double &gy, double &gz)
            {
                double val = std::pow((2 * (y - 0.51)) - std::pow( 2* (x - 0.51), 2) - std::pow(2 *(y - 0.51), 2) + 1, 4)
                +std::pow(std::pow(2 * (x - 0.51), 2) + std::pow( 2* (y - 0.51), 2) + std::pow(2 *(z - 0.51), 2), 4) - 1;
                gx = -32 * (-0.51 + x)*std::pow(1 - 4 * std::pow((-0.51 + x), 2) + 2*(-0.51 + y) - 4*std::pow(-0.51 + y, 2), 3)
                +32 * (-0.51 + x) * std::pow(4 * std::pow(-0.51 + x, 2) + 4 * std::pow(-0.51 + y, 2) + 4 * std::pow(-0.51 + z, 2), 3);
                gy = 4 * (2 - 8 * (-0.51 + y)) * std::pow(1 - 4 * std::pow(-0.51 + x, 2) + 2 * (-0.51 + y) - 4 * std::pow(-0.51 + y, 2), 3)
                + 32 * (-0.51 + y) * std::pow(4 * std::pow(-0.51 + x, 2) + 4 * std::pow(-0.51 + y, 2) + 4 * std::pow(-0.51 + z, 2), 3);
                gz = 32 * std::pow(4 * std::pow(-0.51 + x, 2) + 4 * std::pow(-0.51 + y, 2) + 4 * std::pow(-0.51 + z, 2), 3) * (-0.51 + z);
                return val;
            };
            functions[j] = std::make_unique<GeneralFunction<double>>(f);
        }
        else if (type == "teardrop")
        {
            FuncGrad f = [](double x, double y, double z,
                            double &gx, double &gy, double &gz)
            {
                double val = 0.5 * std::pow(2 * (x - 0.5), 5) +
                0.5 * std::pow(2 * (x - 0.5), 4) - std::pow((2 * (y - 0.5)), 2) - std::pow(2 * (z - 0.5), 2);
                gx = 32 * std::pow(-0.5 + x, 3) + 80 * std::pow(-0.5 + x, 4);
                gy = -8 * (-0.5 + y);
                gz = -8 * (-0.5 + z);
                return val;
            };
            functions[j] = std::make_unique<GeneralFunction<double>>(f);
        }
        else if (type == "cyclide")
        {
            FuncGrad f = [](double x, double y, double z,
                            double &gx, double &gy, double &gz)
            {
                double val = 8816 - 208 * (4 + 900 * std::pow(-0.5 + x, 2)) + 7200 * (-0.5 + x) -
                19200 * (9 * std::pow(-0.5 + y, 2) - 9 * std::pow(-0.5 + z, 2)) +
                10000 * std::pow(9 * std::pow(-0.5 + x, 2) + 9 * std::pow(-0.5 + y, 2) + 9 * std::pow(-0.5 + z, 2), 2);
                gx = 7200 - 374400 * (-0.5 + x) +
                360000 * (-0.5 + x) * (9 * std::pow(-0.5 + x, 2) + 9 * std::pow(-0.5 + y, 2) + 9 * std::pow(-0.5 + z, 2));
                gy = -345600 * (-0.5 + y) +
                360000 * (-0.5 + y) * (9 * std::pow(-0.5 + x, 2) + 9 * std::pow(-0.5 + y, 2) + 9 * std::pow(-0.5 + z, 2));
                gz = 345600 * (-0.5 + z) +
                360000 * (9 * std::pow(-0.5 + x, 2) + 9 * std::pow(-0.5 + y, 2) + 9 * std::pow(-0.5 + z, 2)) * (-0.5 + z);
                return val;
            };
            functions[j] = std::make_unique<GeneralFunction<double>>(f);
        }
        else if (type == "zero")
        {
            functions[j] = std::make_unique<ConstantFunction<double>>(0);
        }
        else
        {
            std::cout << "undefined type: " << type << std::endl;
            return false;
        }
    }
    return true;
}
