#include "implicit_functions.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

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
            double radius = data[j]["radius"].get<double>();
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
            double radius = data[j]["radius"].get<double>();
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