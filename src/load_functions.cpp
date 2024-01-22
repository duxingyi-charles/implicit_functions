#include "load_functions.h"
#include "implicit_functions.h"

#include "nlohmann/json.hpp"
#include <fstream>
#include <iostream>
#include <cmath>

typedef std::function<double(double, double, double, double &, double &, double &)> FuncGrad;

bool import_xyz(const std::string &filename, std::vector<Eigen::Vector3d> &pts) {

    std::ifstream reader(filename.data(), std::ofstream::in);

    if (!reader.good()) {
        std::cout << "Can not open the file " << filename << std::endl;
        return false;
    }else {
        std::cout << "Reading: "<<filename<< std::endl;
    }

    // first number: dimension (2 or 3)
    int dim;
    reader >> dim;
    // read point coordinates
    if (dim != 3) {
        std::cout << "Can't handle non-3D points." << std::endl;
        reader.close();
        return false;
    }
    pts.clear();
    double x,y,z;
    while ((reader >> x >> y >> z)) {
        pts.emplace_back(x,y,z);
    }

    reader.close();
    return true;
}

bool import_RBF_coeff(const std::string &filename, Eigen::VectorXd &a, Eigen::Vector4d &b) {
    std::ifstream reader(filename.data(), std::ofstream::in);
    if (!reader.good()) {
        std::cout << "Can not open the file " << filename << std::endl;
        return false;
    }else {
        std::cout << "Reading: "<<filename<< std::endl;
    }

    // first line: coefficient a
    std::string line;
    std::getline(reader, line);
    std::istringstream iss(line);
    double val;
    std::vector<double> tmp_a;
    while (iss >> val) tmp_a.push_back(val);
    a.resize(tmp_a.size());
    for (size_t i = 0; i < tmp_a.size(); ++i) {
        a(i) = tmp_a[i];
    }

    // second line: coefficient b (d,c0,c1,c2)
    std::getline(reader, line);
    std::istringstream iss2(line);
    double d,c0,c1,c2;
    if (!(iss2 >> d >> c0 >> c1 >> c2)) {
        std::cout << "coeff_b should have 4 elements (in 3D)." << std::endl;
        reader.close();
        return false;
    }
    b << d, c0, c1, c2;

    reader.close();
    return true;
}

bool import_Hermite_RBF(const std::string &pts_file, const std::string &coeff_file, std::vector<Eigen::Vector3d> &control_pts,
                        Eigen::VectorXd &coeff_a, Eigen::Vector4d &coeff_b)
{
    // import control points
    bool succeed = import_xyz(pts_file, control_pts);
    if (!succeed) {
        std::cout << "Fail to import RBF control points." << std::endl;
        return false;
    }

    // import RBF coefficients
    succeed = import_RBF_coeff(coeff_file, coeff_a, coeff_b);
    if (!succeed) {
        std::cout << "Fail to import RBF coefficients." << std::endl;
        return false;
    }

    return true;
}

std::unique_ptr<ImplicitFunction<double>> load_Hermite_RBF(const nlohmann::json& entry,
                                                   const std::string& path_name) {
    assert(entry.contains("points"));
    auto point_file = path_name + entry["points"].get<std::string>();
    assert(entry.contains("rbf_coeffs"));
    auto coeff_file = path_name + entry["rbf_coeffs"].get<std::string>();

    std::vector<Eigen::Vector3d> control_pts;
    Eigen::VectorXd coeff_a;
    Eigen::Vector4d coeff_b;

    if (import_Hermite_RBF(point_file, coeff_file, control_pts, coeff_a, coeff_b)) {
        auto fn = std::make_unique<Hermite_RBF<double>>(control_pts, coeff_a, coeff_b);
        return fn;
    }
    else {
        std::cout << "Failed to load Hermite RBF function!" << std::endl;
        return nullptr;
    }
}

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
    // compatible with Boundary Sample Halfspaces (BSH) config files
    if (data.contains("input")) {
        data = data["input"];
    }
    //
    auto n_pts = static_cast<Eigen::Index>(pts.size());
    auto n_func = static_cast<Eigen::Index>(data.size());
    funcVals.resize(n_pts, n_func);
    for (int j = 0; j < n_func; ++j)
    {
        auto type = data[j]["type"].get<std::string>();
        if (type == "plane")
        {
            std::array<double, 3> point{};
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
            std::array<double, 3> axis_unit_vector;
            // if data has 'axis_point1' and 'axis_point2', then use them to define the axis
            if (data[j].contains("axis_point1") && data[j].contains("axis_point2"))
            {
                for (int i = 0; i < 3; ++i)
                {
                    axis_point[i] = data[j]["axis_point1"][i].get<double>();
                }
                std::array<double, 3> axis_point2;
                for (int i = 0; i < 3; ++i)
                {
                    axis_point2[i] = data[j]["axis_point2"][i].get<double>();
                }
                for (int i = 0; i < 3; ++i)
                {
                    axis_unit_vector[i] = axis_point2[i] - axis_point[i];
                }
            }
            else
            {
                for (int i = 0; i < 3; ++i)
                {
                    axis_point[i] = data[j]["axis_point"][i].get<double>();
                }
                for (int i = 0; i < 3; ++i)
                {
                    axis_unit_vector[i] = data[j]["axis_vector"][i].get<double>();
                }
            }
            auto radius = data[j]["radius"].get<double>();
            bool squared = false;
            if (data[j].contains("squared"))
            {
                squared = data[j]["squared"].get<bool>();
            }
            //
            std::unique_ptr<ImplicitFunction<double>> cylinder;
            if (squared)
                cylinder = std::make_unique<CylinderSquaredDistanceFunction<double>>(axis_point, axis_unit_vector, radius);
            else
                cylinder = std::make_unique<CylinderDistanceFunction<double>>(axis_point, axis_unit_vector, radius);
            for (int i = 0; i < n_pts; i++)
            {
                funcVals(i, j) = cylinder->evaluate(pts[i][0], pts[i][1], pts[i][2]);
            }
        }
        else if (type == "sphere")
        {
            std::array<double, 3> center;
            for (int i = 0; i < 3; ++i)
            {
                center[i] = data[j]["center"][i].get<double>();
            }
            auto radius = data[j]["radius"].get<double>();
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
            auto major_radius = data[j]["major_radius"].get<double>();
            auto minor_radius = data[j]["minor_radius"].get<double>();
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
            auto radius = data[j]["radius"].get<double>();
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
            auto apex_angle = data[j]["apex_angle"].get<double>();
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
        else if (type == "rbf") {
            auto pos = filename.find_last_of("/\\");
            auto path_name = filename.substr(0, pos + 1);
            auto rbf = load_Hermite_RBF(data[j], path_name);
            for (int i = 0; i < n_pts; i++)
            {
                funcVals(i, j) = rbf->evaluate(pts[i][0], pts[i][1], pts[i][2]);
            }
        }
        else if (type == "shader") {
            auto name = data[j]["name"].get<std::string>();
            auto delta = data[j]["delta"].get<float>();
            ImplicitShader<float> shader(name, delta);
            for (int i = 0; i < n_pts; i++)
            {
                funcVals(i, j) = shader.evaluate(pts[i][0], pts[i][1], pts[i][2]);
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
    // compatible with Boundary Sample Halfspaces (BSH) config files
    if (data.contains("input")) {
        data = data["input"];
    }
    //
    size_t n_func = data.size();
    functions.resize(n_func);
    for (int j = 0; j < n_func; ++j)
    {
        auto type = data[j]["type"].get<std::string>();
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
            std::array<double, 3> axis_unit_vector;
            // if data has 'axis_point1' and 'axis_point2', then use them to define the axis
            if (data[j].contains("axis_point1") && data[j].contains("axis_point2"))
            {
                for (int i = 0; i < 3; ++i)
                {
                    axis_point[i] = data[j]["axis_point1"][i].get<double>();
                }
                std::array<double, 3> axis_point2;
                for (int i = 0; i < 3; ++i)
                {
                    axis_point2[i] = data[j]["axis_point2"][i].get<double>();
                }
                for (int i = 0; i < 3; ++i)
                {
                    axis_unit_vector[i] = axis_point2[i] - axis_point[i];
                }
            }
            else
            {
                for (int i = 0; i < 3; ++i)
                {
                    axis_point[i] = data[j]["axis_point"][i].get<double>();
                }
                for (int i = 0; i < 3; ++i)
                {
                    axis_unit_vector[i] = data[j]["axis_vector"][i].get<double>();
                }
            }
            auto radius = data[j]["radius"].get<double>();
            bool squared = false;
            if (data[j].contains("squared"))
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
            auto radius = data[j]["radius"].get<double>();
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
            auto major_radius = data[j]["major_radius"].get<double>();
            auto minor_radius = data[j]["minor_radius"].get<double>();
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
            auto radius = data[j]["radius"].get<double>();
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
            auto apex_angle = data[j]["apex_angle"].get<double>();
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
                double e_power = exp(-21.3333 * std::pow(x, 2) - 21.3333 * std::pow(y, 2) + (-7.10543*std::pow(10,-15) - 21.3333 * z) * z + y * (-7.10543*std::pow(10,-15) + 21.3333 * z) + x * (-7.10543*std::pow(10,-15) + 21.3333 * y + 21.3333 * z));
                double val = 3.4641 + 2 * e_power - 2.3094 * x - 2.3094 * y - 2.3094 * z;
                gx = -2.3094 + e_power *  (-1.42109* std::pow(10, -14) - 85.3333 * x + 42.6667 * y + 42.6667 * z);
                gy = -2.3094 + e_power *  (-1.42109* std::pow(10, -14) + 42.6667 * x - 85.3333 * y + 42.6667 * z);
                gz = -2.3094 + e_power *  (-1.42109* std::pow(10, -14) + 42.6667 * x + 42.6667 * y - 85.3333 * z);
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
        else if (type == "teardrop_shifted")
        {
            FuncGrad f = [](double x, double y, double z,
                            double &gx, double &gy, double &gz)
            {
                double val = 16 * std::pow(-0.6 + 0.995004 * x + 0.0998334 * y, 5) +
                8* std::pow(-0.6 + 0.995004 * x + 0.0998334 * y, 4) - 4 * std::pow(-0.5 - 0.0998334 * x + 0.995004 * y, 2) - std::pow(2 * (z - 0.5), 2);
                gx = 31.8401 * std::pow(-0.6 + 0.995004 * x + 0.0998334 * y, 3) +
                79.6003 * std::pow(-0.6 + 0.995004 * x + 0.0998334 * y, 4) +
                0.798667 * (-0.5 - 0.0998334 * x + 0.995004 * y);
                gy = 3.19467 * std::pow(-0.6 + 0.995004 * x + 0.0998334 * y, 3) +
                7.98667 * std::pow(-0.6 + 0.995004 * x + 0.0998334 * y, 4) -
                7.96003 * (-0.5 - 0.0998334 * x + 0.995004 * y);
                gz = -8 * (-0.5 + z);
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
        else if (type == "rbf") {
            auto pos = filename.find_last_of("/\\");
            auto path_name = filename.substr(0, pos + 1);
            functions[j] = load_Hermite_RBF(data[j], path_name);
        }
        else if (type == "shader") {
            auto name = data[j]["name"].get<std::string>();
            auto delta = data[j]["delta"].get<double>();
            functions[j] = std::make_unique<ImplicitShader<double>>(name, delta);
        }
        else
        {
            std::cout << "undefined type: " << type << std::endl;
            return false;
        }
    }
    return true;
}
