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
#define BOOST_TEST_MODULE FEM2D

#include <boost/test/unit_test.hpp>
#include <Eigen/Dense>
#include <float_type.hpp>

static Eigen::MatrixX<FloatType> points_tri_deg_1_ref = (
    Eigen::MatrixX<FloatType>(3, 2) <<
        0.0, 0.0,
        1.0, 0.0,
        0.0, 1.0
).finished();

static Eigen::MatrixX<FloatType> points_tri_deg_2_ref = (
    Eigen::MatrixX<FloatType>(6, 2) <<
        0.0, 0.0,
        0.5, 0.0,
        1.0, 0.0,
        0.5, 0.5,
        0.0, 1.0,
        0.0, 0.5
).finished();

static Eigen::MatrixX<FloatType> points_quad_deg_1_ref = (
    Eigen::MatrixX<FloatType>(4, 2) <<
        0.0, 0.0,
        1.0, 0.0,
        1.0, 1.0,
        0.0, 1.0
).finished();

static Eigen::MatrixX<FloatType> points_quad_deg_2_ref = (
    Eigen::MatrixX<FloatType>(9, 2) <<
        0.0, 0.0,
        0.5, 0.0,
        1.0, 0.0,
        1.0, 0.5,
        1.0, 1.0,
        0.5, 1.0,
        0.0, 1.0,
        0.0, 0.5,
        0.5, 0.5
).finished();

static Eigen::MatrixX<FloatType> M_1x1_ref = (
    Eigen::MatrixX<FloatType>(4, 4) <<
         0.1944444444444444, 0.0833333333333333, 0.0972222222222222, 0.0416666666666667,
         0.0833333333333333, 0.1388888888888889, 0.0416666666666667, 0.0694444444444444,
         0.0972222222222222, 0.0416666666666667, 0.1944444444444444, 0.0833333333333333,
         0.0416666666666667, 0.0694444444444444, 0.0833333333333333, 0.1388888888888889
).finished();

static Eigen::MatrixX<FloatType> A_1x1_ref = (
    Eigen::MatrixX<FloatType>(4, 4) <<
         0.924196240746594, -0.348392481493187,  0.0758037592534063, -0.651607518506813,
        -0.348392481493187,  0.696784962986375, -0.1516075185068130, -0.196784962986375,
         0.075803759253406, -0.151607518506813,  0.4241962407465940, -0.348392481493187,
        -0.651607518506813, -0.196784962986375, -0.3483924814931870,  1.196784962986370
).finished();

static Eigen::MatrixX<FloatType> A_xx_1x1_ref = (
    Eigen::MatrixX<FloatType>(4, 4) <<
         0.731049060186648, -0.462098120373297,  0.268950939813352, -0.537901879626703,
        -0.462098120373297,  0.424196240746594, -0.037901879626703,  0.075803759253406,
         0.268950939813352, -0.037901879626703,  0.231049060186648, -0.462098120373297,
        -0.537901879626703,  0.075803759253406, -0.462098120373297,  0.924196240746594
).finished();

static Eigen::MatrixX<FloatType> A_zz_1x1_ref = (
    Eigen::MatrixX<FloatType>(4, 4) <<
         0.193147180559945,  0.113705638880109, -0.193147180559945, -0.113705638880109,
         0.113705638880109,  0.272588722239781, -0.113705638880109, -0.272588722239781,
        -0.193147180559945, -0.113705638880109,  0.193147180559945,  0.113705638880109,
        -0.113705638880109, -0.272588722239781,  0.113705638880109,  0.272588722239781
).finished();
