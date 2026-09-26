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
#define BOOST_TEST_MODULE StructuredMesh

#include <boost/test/unit_test.hpp>
#include <Eigen/Dense>
#include <float_type.hpp>

// pmat for nx=3, nz=2, d=1
static Eigen::MatrixX<FloatType> p1_mesh_pmat_ref = (
    Eigen::MatrixX<FloatType>(12, 2) <<
    0.0,     0.0,
    1.0/3.0, 0.0,
    2.0/3.0, 0.0,
    1.0,     0.0,

    0.0,     0.5,
    1.0/3.0, 0.5,
    2.0/3.0, 0.5,
    1.0,     0.5,

    0.0,     1.0,
    1.0/3.0, 1.0,
    2.0/3.0, 1.0,
    1.0,     1.0
).finished();

// pmat for nx=3, nz=2, d=2
static Eigen::MatrixX<FloatType> p2_mesh_pmat_ref = (
    Eigen::MatrixX<FloatType>(35, 2) <<
        0.0,     0.0,
        1.0/6.0, 0.0,
        2.0/6.0, 0.0,
        3.0/6.0, 0.0,
        4.0/6.0, 0.0,
        5.0/6.0, 0.0,
        1.0,     0.0,

        0.0,     0.25,
        1.0/6.0, 0.25,
        2.0/6.0, 0.25,
        3.0/6.0, 0.25,
        4.0/6.0, 0.25,
        5.0/6.0, 0.25,
        1.0,     0.25,

        0.0,     0.5,
        1.0/6.0, 0.5,
        2.0/6.0, 0.5,
        3.0/6.0, 0.5,
        4.0/6.0, 0.5,
        5.0/6.0, 0.5,
        1.0,     0.5,

        0.0,     0.75,
        1.0/6.0, 0.75,
        2.0/6.0, 0.75,
        3.0/6.0, 0.75,
        4.0/6.0, 0.75,
        5.0/6.0, 0.75,
        1.0,     0.75,

        0.0,     1.0,
        1.0/6.0, 1.0,
        2.0/6.0, 1.0,
        3.0/6.0, 1.0,
        4.0/6.0, 1.0,
        5.0/6.0, 1.0,
        1.0,     1.0
).finished();

// cmat for nx=3, nz=2, d=1, QUADRILATERAL
static Eigen::MatrixXi p1_mesh_cmat_quad_ref = (
    Eigen::MatrixXi(6, 4) <<
        0, 1, 5, 4,
        1, 2, 6, 5,
        2, 3, 7, 6,
        4, 5, 9, 8,
        5, 6, 10, 9,
        6, 7, 11, 10	
).finished();

// cmat for nx=3, nz=2, d=1, TRIANGLE_RIGHT
static Eigen::MatrixXi p1_mesh_cmat_tri_right_ref = (
    Eigen::MatrixXi(12, 3) <<
        0, 1, 4,
        5, 4, 1,
        1, 2, 5,
        6, 5, 2,
        2, 3, 6,
        7, 6, 3,

        4, 5, 8,
        9, 8, 5,
        5, 6, 9,
        10, 9, 6,
        6, 7, 10,
        11, 10, 7
).finished();

// cmat for nx=3, nz=2, d=1, TRIANGLE_LEFT
static Eigen::MatrixXi p1_mesh_cmat_tri_left_ref = (
    Eigen::MatrixXi(12, 3) <<
        4, 0, 5,
        1, 5, 0,
        5, 1, 6,
        2, 6, 1,
        6, 2, 7,
        3, 7, 2,

        8, 4, 9,
        5, 9, 4,
        9, 5, 10,
        6, 10, 5,
        10, 6, 11,
        7, 11, 6
).finished();

// cmat for nx=3, nz=2, d=2, QUADRILATERAL
static Eigen::MatrixXi p2_mesh_cmat_quad_ref = (
    Eigen::MatrixXi(6, 9) <<
        0, 1, 2, 9, 16, 15, 14, 7, 8,
        2, 3, 4, 11, 18, 17, 16, 9, 10,
        4, 5, 6, 13, 20, 19, 18, 11, 12,

        14, 15, 16, 23, 30, 29, 28, 21, 22,
        16, 17, 18, 25, 32, 31, 30, 23, 24,
        18, 19, 20, 27, 34, 33, 32, 25, 26
).finished();

// cmat for nx=3, nz=2, d=2, TRIANGLE_RIGHT
static Eigen::MatrixXi p2_mesh_cmat_tri_right_ref = (
    Eigen::MatrixXi(12, 6) <<
        0, 1, 2, 8, 14, 7,
        16, 15, 14, 8, 2, 9,
        2, 3, 4, 10, 16, 9,
        18, 17, 16, 10, 4, 11,
        4, 5, 6, 12, 18, 11,
        20, 19, 18, 12, 6, 13,

        14, 15, 16, 22, 28, 21,
        30, 29, 28, 22, 16, 23,
        16, 17, 18, 24, 30, 23,
        32, 31, 30, 24, 18, 25,
        18, 19, 20, 26, 32, 25,
        34, 33, 32, 26, 20, 27
).finished();

// cmat for nx=3, nz=2, d=2, TRIANGLE_LEFT
static Eigen::MatrixXi p2_mesh_cmat_tri_left_ref = (
    Eigen::MatrixXi(12, 6) <<
        14, 7, 0, 8, 16, 15,
        2, 9, 16, 8, 0, 1,
        16, 9, 2, 10, 18, 17,
        4, 11, 18, 10, 2, 3,
        18, 11, 4, 12, 20, 19,
        6, 13, 20, 12, 4, 5,

        28, 21, 14, 22, 30, 29,
        16, 23, 30, 22, 14, 15,
        30, 23, 16, 24, 32, 31,
        18, 25, 32, 24, 16, 17,
        32, 25, 18, 26, 34, 33,
        20, 27, 34, 26, 18, 19
).finished();

// dimat for nx=3, nz=2, d=2, TRIANGLE_LEFT
static Eigen::VectorXi p1_mesh_dimat_ref = (
    Eigen::VectorXi(12) <<
        16, 32, 32, 64,
        8, 1, 1, 128,
        4, 2, 2, 256
).finished();

// dimat for nx=3, nz=2, d=2, TRIANGLE_LEFT
static Eigen::VectorXi p2_mesh_dimat_ref = (
    Eigen::VectorXi(35) <<
        16, 32, 32, 32, 32, 32, 64,
        8, 1, 1, 1, 1, 1, 128,
        8, 1, 1, 1, 1, 1, 128,
        8, 1, 1, 1, 1, 1, 128,
        4, 2, 2, 2, 2, 2, 256
).finished();

/**********************
 * DOFS REFERENCE IDS *
 **********************/
// North boundary dofs for nx=2, nz=2, d=2
static Eigen::VectorXi north_boundary_dofs_ref = (
    Eigen::VectorXi(3) << 21, 22, 23
).finished();

// West boundary dofs for nx=2, nz=2, d=2
static Eigen::VectorXi west_boundary_dofs_ref = (
    Eigen::VectorXi(3) << 5, 10, 15
).finished();

// South boundary dofs for nx=2, nz=2, d=2
static Eigen::VectorXi south_boundary_dofs_ref = (
    Eigen::VectorXi(3) << 1, 2, 3
).finished();

// West boundary dofs for nx=2, nz=2, d=2
static Eigen::VectorXi east_boundary_dofs_ref = (
    Eigen::VectorXi(3) << 9, 14, 19
).finished();

// Interior dofs for nx=2, nz=2, d=2
static Eigen::VectorXi interior_dofs_ref = (
    Eigen::VectorXi(9) <<
        6, 7, 8,
        11, 12, 13,
        16, 17, 18
).finished();

// Domain dofs for nx=2, nz=2, d=2
static Eigen::VectorXi boundary_dofs_ref = (
    Eigen::VectorXi(16) << 
        0, 1, 2, 3, 4,
        5, 9, 10, 14, 15, 19,
        20, 21, 22, 23, 24
).finished();

// Domain dofs for nx=2, nz=2, d=2
static Eigen::VectorXi domain_dofs_ref = (
    Eigen::VectorXi(25) << 
        0, 1, 2, 3, 4,
        5, 6, 7, 8, 9,
        10, 11, 12, 13, 14,
        15, 16, 17, 18, 19,
        20, 21, 22, 23, 24
).finished();

// Domain dofs for nx=2, nz=2, d=2
static Eigen::VectorXi i_or_nb_dofs_ref = (
    Eigen::VectorXi(12) << 
        6, 7, 8,
        11, 12, 13,
        16, 17, 18,
        21, 22, 23
).finished();

// Domain dofs for nx=2, nz=2, d=2
static Eigen::VectorXi b_not_sb_not_wb_dofs_ref = (
    Eigen::VectorXi(10) << 
        0, 4, 9, 14, 19, 20, 21, 22, 23, 24
).finished();

/*****************************
 * VERTEX DOFS REFERENCE IDS *
 *****************************/
// North boundary dofs for nx=2, nz=2, d=2
static Eigen::VectorXi north_boundary_vdofs_ref = (
    Eigen::VectorXi(1) << 22
).finished();

// West boundary dofs for nx=2, nz=2, d=2
static Eigen::VectorXi west_boundary_vdofs_ref = (
    Eigen::VectorXi(1) << 10
).finished();

// South boundary dofs for nx=2, nz=2, d=2
static Eigen::VectorXi south_boundary_vdofs_ref = (
    Eigen::VectorXi(1) << 2
).finished();

// West boundary dofs for nx=2, nz=2, d=2
static Eigen::VectorXi east_boundary_vdofs_ref = (
    Eigen::VectorXi(1) << 14
).finished();

// Interior dofs for nx=2, nz=2, d=2
static Eigen::VectorXi interior_vdofs_ref = (
    Eigen::VectorXi(1) << 12
).finished();

// Domain dofs for nx=2, nz=2, d=2
static Eigen::VectorXi boundary_vdofs_ref = (
    Eigen::VectorXi(8) << 0, 2, 4, 10, 14, 20, 22, 24
).finished();

// Domain dofs for nx=2, nz=2, d=2
static Eigen::VectorXi domain_vdofs_ref = (
    Eigen::VectorXi(9) << 0, 2, 4, 10, 12, 14, 20, 22, 24
).finished();

// Domain dofs for nx=2, nz=2, d=2
static Eigen::VectorXi i_or_nb_vdofs_ref = (
    Eigen::VectorXi(2) << 12, 22
).finished();

// Domain dofs for nx=2, nz=2, d=2
static Eigen::VectorXi nb_or_eb_vdofs_ref = (
    Eigen::VectorXi(2) << 14, 22 
).finished();


/**********************
 * CELL REFERENCE IDS *
 **********************/
static Eigen::VectorXi cimat_3x4_quad_ref = (
    Eigen::VectorXi(12) << 16, 32, 64, 8, 1, 128, 8, 1, 128, 4, 2, 256 
).finished();
static Eigen::MatrixXi cmat_3x3_quad_p1_ref = (
    Eigen::MatrixXi(3, 4) <<
    8, 9, 13, 12,
    9, 10, 14, 13,
    10, 11, 15, 14
).finished();
