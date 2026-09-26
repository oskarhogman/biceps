/*
 * Copyright (C) 2025 André Löfgren
 *
 * This file is part of Biceps.
 *
 * Biceps is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Biceps is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Biceps. If not, see <https://www.gnu.org/licenses/>.
 */
#pragma once
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Stokes

#include <boost/test/unit_test.hpp>
#include <pstokes_fem.hpp>
#include <enums.hpp>

using namespace MESH2D;

inline FloatType ux_func(FloatType x, FloatType z) {
    return SIN_FUNC(4.0*PI_CONST*x)*COS_FUNC(4.0*PI_CONST*z);
}

inline FloatType uz_func(FloatType x, FloatType z) {
    return -COS_FUNC(4.0*PI_CONST*x)*SIN_FUNC(4.0*PI_CONST*z);
}

inline FloatType uz_func_bed(FloatType x, FloatType z) {
    return COS_FUNC(4.0*PI_CONST*x)*SIN_FUNC(4.0*PI_CONST*z);
}

inline FloatType p_func(FloatType x, FloatType z) {
    return PI_CONST*COS_FUNC(4.0*PI_CONST*x)*COS_FUNC(4.0*PI_CONST*z);
}

inline FloatType force_x(FloatType x, FloatType z) {
    return 28.0*PI_CONST*PI_CONST*SIN_FUNC(4.0*PI_CONST*x)*COS_FUNC(4.0*PI_CONST*z);
}

inline FloatType force_z(FloatType x, FloatType z) {
    return -36.0*PI_CONST*PI_CONST*COS_FUNC(4.0*PI_CONST*x)*SIN_FUNC(4.0*PI_CONST*z);
}

static std::vector<std::tuple<int, int, int, FloatType, FloatType>> test_manufactured_sol_cases;

inline void populate_manufactured_sol_cases() {
    test_manufactured_sol_cases.push_back(std::tuple<int, int, int, FloatType, FloatType>(
        8, 8, TRIANGLE_RIGHT, 0.05, 0.08
    ));
    test_manufactured_sol_cases.push_back(std::tuple<int, int, int, FloatType, FloatType>(
        16, 8, TRIANGLE_RIGHT, 0.04, 0.03
    ));
    test_manufactured_sol_cases.push_back(std::tuple<int, int, int, FloatType, FloatType>(
        8, 16, TRIANGLE_RIGHT, 0.015, 0.05
    ));
    test_manufactured_sol_cases.push_back(std::tuple<int, int, int, FloatType, FloatType>(
        16, 16, TRIANGLE_RIGHT, 0.075, 0.075
    ));
    test_manufactured_sol_cases.push_back(std::tuple<int, int, int, FloatType, FloatType>(
        32, 32, TRIANGLE_RIGHT, 0.005, 0.005
    ));

    test_manufactured_sol_cases.push_back(std::tuple<int, int, int, FloatType, FloatType>(
        8, 8, TRIANGLE_LEFT, 0.05, 0.08
    ));
    test_manufactured_sol_cases.push_back(std::tuple<int, int, int, FloatType, FloatType>(
        16, 8, TRIANGLE_LEFT, 0.04, 0.03
    ));
    test_manufactured_sol_cases.push_back(std::tuple<int, int, int, FloatType, FloatType>(
        8, 16, TRIANGLE_LEFT, 0.015, 0.05
    ));
    test_manufactured_sol_cases.push_back(std::tuple<int, int, int, FloatType, FloatType>(
        16, 16, TRIANGLE_LEFT, 0.075, 0.075
    ));
    test_manufactured_sol_cases.push_back(std::tuple<int, int, int, FloatType, FloatType>(
        32, 32, TRIANGLE_LEFT, 0.005, 0.005
    ));

    test_manufactured_sol_cases.push_back(std::tuple<int, int, int, FloatType, FloatType>(
        8, 8, QUADRILATERAL, 0.05, 0.08
    ));
    test_manufactured_sol_cases.push_back(std::tuple<int, int, int, FloatType, FloatType>(
        16, 8, QUADRILATERAL, 0.04, 0.03
    ));
    test_manufactured_sol_cases.push_back(std::tuple<int, int, int, FloatType, FloatType>(
        8, 16, QUADRILATERAL, 0.015, 0.05
    ));
    test_manufactured_sol_cases.push_back(std::tuple<int, int, int, FloatType, FloatType>(
        16, 16, QUADRILATERAL, 0.075, 0.075
    ));
    test_manufactured_sol_cases.push_back(std::tuple<int, int, int, FloatType, FloatType>(
        32, 32, QUADRILATERAL, 0.005, 0.005
    ));
}
