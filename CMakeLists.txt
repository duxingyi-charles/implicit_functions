cmake_minimum_required(VERSION 3.14)

project(implicit_functions LANGUAGES CXX)

option(IMPLICIT_FUNCTIONS_WITH_SHADER_SUPPORT "Enable shader support" OFF)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
include(nlohmann-json)
include(Eigen3)

add_library(implicit_functions STATIC
        "include/implicit_functions.h"
        "src/load_functions.cpp")

target_link_libraries(implicit_functions PUBLIC nlohmann_json::nlohmann_json Eigen3::Eigen)

target_compile_features(implicit_functions PRIVATE cxx_std_20)

target_include_directories(implicit_functions PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(implicit_functions::implicit_functions ALIAS
        implicit_functions)

if (IMPLICIT_FUNCTIONS_WITH_SHADER_SUPPORT)
    include(implicit_shader)
    target_link_libraries(implicit_functions PUBLIC implicit_shader::implicit_shader)
    target_compile_definitions(implicit_functions PUBLIC IMPLICIT_FUNCTIONS_WITH_SHADER_SUPPORT)
endif()
