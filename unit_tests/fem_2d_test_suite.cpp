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
#include <fem_2d_test_suite.hpp>
#include <fem_2d.hpp>
#include <boost/format.hpp>
#include <enums.hpp>

using namespace FEM2D;
using namespace MESH2D;

BOOST_AUTO_TEST_CASE(test_reference_element_points, *boost::unit_test::tolerance(1e-14))
{
    Eigen::MatrixX<FloatType> points_tri_right_deg_1 = FEM2D::reference_element_points_rs(TRIANGLE_RIGHT, 1);
    Eigen::MatrixX<FloatType> points_tri_right_deg_2 = FEM2D::reference_element_points_rs(TRIANGLE_RIGHT, 2);
    Eigen::MatrixX<FloatType> points_tri_left_deg_1 = FEM2D::reference_element_points_rs(TRIANGLE_LEFT, 1);
    Eigen::MatrixX<FloatType> points_tri_left_deg_2 = FEM2D::reference_element_points_rs(TRIANGLE_LEFT, 2);
    Eigen::MatrixX<FloatType> points_quad_deg_1 = FEM2D::reference_element_points_rs(QUADRILATERAL, 1);
    Eigen::MatrixX<FloatType> points_quad_deg_2 = FEM2D::reference_element_points_rs(QUADRILATERAL, 2);

    BOOST_CHECK_EQUAL(points_tri_right_deg_1, points_tri_deg_1_ref);
    BOOST_CHECK_EQUAL(points_tri_right_deg_2, points_tri_deg_2_ref);
    BOOST_CHECK_EQUAL(points_tri_left_deg_1, points_tri_deg_1_ref);
    BOOST_CHECK_EQUAL(points_tri_left_deg_2, points_tri_deg_2_ref);
    BOOST_CHECK_EQUAL(points_quad_deg_1, points_quad_deg_1_ref);
    BOOST_CHECK_EQUAL(points_quad_deg_2, points_quad_deg_2_ref);
}

BOOST_AUTO_TEST_CASE(test_assemble_mass_matrix)
{
    std::vector<FloatType> tol_vec;
    tol_vec.push_back(0.5);
    tol_vec.push_back(1e-14);
    tol_vec.push_back(1e-14);
    tol_vec.push_back(1e-14);
    tol_vec.push_back(1e-14);
    Eigen::VectorX<FloatType> svec_p1(2);
    svec_p1(0) = 2.0;
    svec_p1(1) = 1.0;
    StructuredMesh mesh(1, 1, 1, QUADRILATERAL);
    mesh.extrude_z(svec_p1);
    for (int gp = 1; gp <= 5; ++gp) {
        Eigen::MatrixX<FloatType> M_1x1 = (
            FEM2D::assemble_mass_matrix(mesh, gp)
        );
        FloatType error = (M_1x1_ref - M_1x1).cwiseAbs().maxCoeff();
        BOOST_TEST_CHECK(
             error < tol_vec[gp-1],
            (boost::format{"FEM2D::assemble_mass_matrix(mesh=StructuredMesh(nx=%d, nz=%d, degree=%d, cell_type=%s), gp=%d) failed: ||error_A||_inf = %g > %g"} %1 %1 %1 %QUADRILATERAL %gp %error %tol_vec[gp-1])
        );
    }
}

BOOST_AUTO_TEST_CASE(test_assemble_stiffness_matrix)
{
    std::vector<FloatType> tol_vec;
    tol_vec.push_back(0.5);
    tol_vec.push_back(5e-3);
    tol_vec.push_back(5e-4);
    tol_vec.push_back(5e-6);
    tol_vec.push_back(5e-7);
    Eigen::VectorX<FloatType> svec_p1(2);
    svec_p1(0) = 2.0;
    svec_p1(1) = 1.0;
    StructuredMesh mesh(1, 1, 1, QUADRILATERAL);
    mesh.extrude_z(svec_p1);
    for (int gp = 1; gp <= 5; ++gp) {
        Eigen::MatrixX<FloatType> A_1x1 = Eigen::MatrixX<FloatType>(
            FEM2D::assemble_stiffness_matrix(mesh, gp)
        );
        FloatType error = (A_1x1_ref - A_1x1).cwiseAbs().maxCoeff();
        BOOST_TEST_CHECK(
             error < tol_vec[gp-1],
            (boost::format{"FEM2D::assemble_stiffness_matrix(mesh=StructuredMesh(nx=%d, nz=%d, degree=%d, cell_type=%s), gp=%d) failed: ||error_A||_inf = %g > %g"} %1 %1 %1 %QUADRILATERAL %gp %error %tol_vec[gp-1])
        );
    }
}

BOOST_AUTO_TEST_CASE(test_assemble_stiffness_xx_matrix)
{
    std::vector<FloatType> tol_vec;
    tol_vec.push_back(0.5);
    tol_vec.push_back(5e-3);
    tol_vec.push_back(5e-5);
    tol_vec.push_back(5e-6);
    tol_vec.push_back(5e-8);
    Eigen::VectorX<FloatType> svec_p1(2);
    svec_p1(0) = 2.0;
    svec_p1(1) = 1.0;
    StructuredMesh mesh(1, 1, 1, QUADRILATERAL);
    mesh.extrude_z(svec_p1);
    for (int gp = 1; gp <= 5; ++gp) {
        Eigen::MatrixX<FloatType> A_xx_1x1 = Eigen::MatrixX<FloatType>(
            FEM2D::assemble_stiffness_xx_matrix(mesh, gp)
        );
        FloatType error = (A_xx_1x1_ref - A_xx_1x1).cwiseAbs().maxCoeff();
        BOOST_TEST_CHECK(
             error < tol_vec[gp-1],
            (boost::format{"FEM2D::assemble_stiffness_xx_matrix(mesh=StructuredMesh(nx=%d, nz=%d, degree=%d, cell_type=%s), gp=%d) failed: ||error_A||_inf = %g > %g"} %1 %1 %1 %QUADRILATERAL %gp %error %tol_vec[gp-1])
        );
    }
}

BOOST_AUTO_TEST_CASE(test_assemble_stiffness_zz_matrix)
{
    std::vector<FloatType> tol_vec;
    tol_vec.push_back(0.5);
    tol_vec.push_back(5e-3);
    tol_vec.push_back(5e-4);
    tol_vec.push_back(5e-6);
    tol_vec.push_back(5e-7);
    Eigen::VectorX<FloatType> svec_p1(2);
    svec_p1(0) = 2.0;
    svec_p1(1) = 1.0;
    StructuredMesh mesh(1, 1, 1, QUADRILATERAL);
    mesh.extrude_z(svec_p1);
    for (int gp = 1; gp <= 5; ++gp) {
        Eigen::MatrixX<FloatType> A_zz_1x1 = Eigen::MatrixX<FloatType>(
            FEM2D::assemble_stiffness_zz_matrix(mesh, gp)
        );
        FloatType error = (A_zz_1x1_ref - A_zz_1x1).cwiseAbs().maxCoeff();
        BOOST_TEST_CHECK(
             error < tol_vec[gp-1],
            (boost::format{"FEM2D::assemble_stiffness_zz_matrix(mesh=StructuredMesh(nx=%d, nz=%d, degree=%d, cell_type=%s), gp=%d) failed: ||error_A||_inf = %g > %g"} %1 %1 %1 %QUADRILATERAL %gp %error %tol_vec[gp-1])
        );
    }
}

BOOST_AUTO_TEST_CASE(test_diff_x_interp)
{
}

BOOST_AUTO_TEST_CASE(test_diff_z_interp)
{
}

BOOST_AUTO_TEST_CASE(test_diff_x_proj)
{
}

BOOST_AUTO_TEST_CASE(test_diff_z_proj)
{
}
