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
#include <mesh_test_suite.hpp>
#include <structured_mesh.hpp>
#include <boost/format.hpp>
#include <enums.hpp>

using namespace MESH2D;

BOOST_AUTO_TEST_CASE(test_mesh_pmat, *boost::unit_test::tolerance(1e-14))
{
    int nx = 3;
    int nz = 2;
    std::vector<int> element_ids;
    element_ids.push_back(QUADRILATERAL);
    element_ids.push_back(TRIANGLE_RIGHT);
    element_ids.push_back(TRIANGLE_LEFT);

    int deg = 1;
    for (int id: element_ids) {
        std::string element = id == QUADRILATERAL ? "QUADRILATERAL" : id == TRIANGLE_RIGHT ? "TRIANGLE_RIGHT": "TRIANGLE_LEFT";
        StructuredMesh sm(nx, nz, deg, id);
        Eigen::ArrayXX<FloatType> pmat_adiff = (sm.pmat - p1_mesh_pmat_ref).array().abs();
        BOOST_REQUIRE_EQUAL(sm.pmat.rows(), p1_mesh_pmat_ref.rows());
        BOOST_REQUIRE_EQUAL(sm.pmat.cols(), p1_mesh_pmat_ref.cols());
        BOOST_TEST_CHECK(
            fabsl(pmat_adiff.maxCoeff()) < 1e-14,
            (boost::format{"cmat test for StructuredMesh(nx=%d, nz=%d, degree=%d, cell_type=%s) failed:"}
            % nx % nz % deg % element).str()
        );
    }

    deg = 2;
    for (int id: element_ids) {
        std::string element = id == QUADRILATERAL ? "QUADRILATERAL" : id == TRIANGLE_RIGHT ? "TRIANGLE_RIGHT": "TRIANGLE_LEFT";
        StructuredMesh sm(nx, nz, deg, id);
        Eigen::ArrayXX<FloatType> pmat_adiff = (sm.pmat - p2_mesh_pmat_ref).array().abs();
        BOOST_REQUIRE_EQUAL(sm.pmat.rows(), p2_mesh_pmat_ref.rows());
        BOOST_REQUIRE_EQUAL(sm.pmat.cols(), p2_mesh_pmat_ref.cols());
        BOOST_TEST_CHECK(
            fabsl(pmat_adiff.maxCoeff()) < 1e-14,
            (boost::format{"cmat test for StructuredMesh(nx=%d, nz=%d, degree=%d, cell_type=%s) failed:"}
            % nx % nz % deg % element).str()
        );
    }
}

BOOST_AUTO_TEST_CASE(test_mesh_cmat)
{
    int nx = 3;
    int nz = 2;
    std::vector<int> element_ids;
    element_ids.push_back(QUADRILATERAL);
    element_ids.push_back(TRIANGLE_RIGHT);
    element_ids.push_back(TRIANGLE_LEFT);

    int deg = 1;
    Eigen::MatrixXi p1_mesh_cmat_ref;
    for (int id: element_ids) {
        if (id == QUADRILATERAL)
            p1_mesh_cmat_ref = p1_mesh_cmat_quad_ref;
        else if (id == TRIANGLE_RIGHT)
            p1_mesh_cmat_ref = p1_mesh_cmat_tri_right_ref;
        else
            p1_mesh_cmat_ref = p1_mesh_cmat_tri_left_ref;
        std::string element = id == QUADRILATERAL ? "QUADRILATERAL" : id == TRIANGLE_RIGHT ? "TRIANGLE_RIGHT": "TRIANGLE_LEFT";
        StructuredMesh sm(nx, nz, deg, id);
        Eigen::ArrayXXi cmat_adiff = (sm.cmat - p1_mesh_cmat_ref).array().abs();
        BOOST_REQUIRE_EQUAL(sm.cmat.rows(), p1_mesh_cmat_ref.rows());
        BOOST_REQUIRE_EQUAL(sm.cmat.cols(), p1_mesh_cmat_ref.cols());
        BOOST_TEST_CHECK(
            cmat_adiff.maxCoeff() == 0,
            (boost::format{"cmat test for StructuredMesh(nx=%d, nz=%d, degree=%d, cell_type=%s) failed:"}
            % nx % nz % deg % element).str()
        );
    }

    deg = 2;
    Eigen::MatrixXi p2_mesh_cmat_ref;
    for (int id: element_ids) {
        if (id == QUADRILATERAL)
            p2_mesh_cmat_ref = p2_mesh_cmat_quad_ref;
        else if (id == TRIANGLE_RIGHT)
            p2_mesh_cmat_ref = p2_mesh_cmat_tri_right_ref;
        else
            p2_mesh_cmat_ref = p2_mesh_cmat_tri_left_ref;
        std::string element = id == QUADRILATERAL ? "QUADRILATERAL" : id == TRIANGLE_RIGHT ? "TRIANGLE_RIGHT": "TRIANGLE_LEFT";
        StructuredMesh sm(nx, nz, deg, id);
        Eigen::ArrayXXi cmat_adiff = (sm.cmat - p2_mesh_cmat_ref).array().abs();
        BOOST_REQUIRE_EQUAL(sm.cmat.rows(), p2_mesh_cmat_ref.rows());
        BOOST_REQUIRE_EQUAL(sm.cmat.cols(), p2_mesh_cmat_ref.cols());
        BOOST_TEST_CHECK(
            cmat_adiff.maxCoeff() == 0,
            (boost::format{"cmat test for StructuredMesh(nx=%d, nz=%d, degree=%d, cell_type=%s) failed:"}
            % nx % nz % deg % element).str()
        );
    }
}

BOOST_AUTO_TEST_CASE(test_mesh_cimat)
{
    int nx = 3;
    int nz = 4;
    StructuredMesh sm_p1(nx, nz, 1, QUADRILATERAL);
    Eigen::ArrayXXi cimat_adiff_p1 = (sm_p1.cimat - cimat_3x4_quad_ref).array().abs();
    BOOST_REQUIRE_EQUAL(sm_p1.cimat.rows(), cimat_3x4_quad_ref.rows());
    BOOST_REQUIRE_EQUAL(sm_p1.cimat.cols(), cimat_3x4_quad_ref.cols());
    BOOST_TEST_CHECK(
        cimat_adiff_p1.maxCoeff() == 0,
        (boost::format{"cimat test for StructuredMesh(nx=%d, nz=%d, degree=%d, cell_type=%s) failed:"}
        % nx % nz % 1 % QUADRILATERAL).str()
    );

    StructuredMesh sm_p2(nx, nz, 2, QUADRILATERAL);
    Eigen::ArrayXXi cimat_adiff_p2 = (sm_p1.cimat - cimat_3x4_quad_ref).array().abs();
    BOOST_REQUIRE_EQUAL(sm_p2.cimat.rows(), cimat_3x4_quad_ref.rows());
    BOOST_REQUIRE_EQUAL(sm_p2.cimat.cols(), cimat_3x4_quad_ref.cols());
    BOOST_TEST_CHECK(
        cimat_adiff_p2.maxCoeff() == 0,
        (boost::format{"cimat test for StructuredMesh(nx=%d, nz=%d, degree=%d, cell_type=%s) failed:"}
        % nx % nz % 2 % QUADRILATERAL).str()
    );
}

BOOST_AUTO_TEST_CASE(test_mesh_extract_cell_inds)
{
    int nx = 3;
    int nz = 3;
    StructuredMesh mesh(nx, nz, 1, QUADRILATERAL);
    std::vector<int> cinds = mesh.extract_cell_inds(NORTH_WEST_ID | NORTH_ID | NORTH_EAST_ID);
    Eigen::MatrixXi cmat_3x3_quad_p1 = mesh.cmat(cinds, Eigen::all);

    Eigen::ArrayXXi cmat_adiff_p1 = (cmat_3x3_quad_p1 - cmat_3x3_quad_p1_ref).array().abs();
    BOOST_REQUIRE_EQUAL(cmat_3x3_quad_p1.rows(), cmat_3x3_quad_p1_ref.rows());
    BOOST_REQUIRE_EQUAL(cmat_3x3_quad_p1.cols(), cmat_3x3_quad_p1_ref.cols());
    BOOST_TEST_CHECK(
        cmat_adiff_p1.maxCoeff() == 0,
        (boost::format{"StructuredMesh(nx=%d, nz=%d, degree=%d, cell_type=%s).extract_cell_inds(NORTH_WEST_ID | NORTH_ID | NORTH_EAST_ID) failed:"}
        % nx % nz % 1 % QUADRILATERAL).str()
    );
}

BOOST_AUTO_TEST_CASE(test_mesh_dimat)
{
    int nx = 3;
    int nz = 2;
    std::vector<int> element_ids;
    element_ids.push_back(QUADRILATERAL);
    element_ids.push_back(TRIANGLE_RIGHT);
    element_ids.push_back(TRIANGLE_LEFT);

    int deg = 1;
    for (int id: element_ids) {
        std::string element = id == QUADRILATERAL ? "QUADRILATERAL" : id == TRIANGLE_RIGHT ? "TRIANGLE_RIGHT": "TRIANGLE_LEFT";
        StructuredMesh sm(nx, nz, deg, id);
        Eigen::ArrayXXi dimat_adiff = (sm.dimat - p1_mesh_dimat_ref).array().abs();
        BOOST_REQUIRE_EQUAL(sm.dimat.rows(), p1_mesh_dimat_ref.rows());
        BOOST_REQUIRE_EQUAL(sm.dimat.cols(), p1_mesh_dimat_ref.cols());
        BOOST_TEST_CHECK(
            dimat_adiff.maxCoeff() == 0,
            (boost::format{"dimat test for StructuredMesh(nx=%d, nz=%d, degree=%d, cell_type=%s) failed:"}
            % nx % nz % deg % element).str()
        );
    }

    deg = 2;
    for (int id: element_ids) {
        std::string element = id == QUADRILATERAL ? "QUADRILATERAL" : id == TRIANGLE_RIGHT ? "TRIANGLE_RIGHT": "TRIANGLE_LEFT";
        StructuredMesh sm(nx, nz, deg, id);
        Eigen::ArrayXXi dimat_adiff = (sm.dimat - p2_mesh_dimat_ref).array().abs();
        BOOST_REQUIRE_EQUAL(sm.dimat.rows(), p2_mesh_dimat_ref.rows());
        BOOST_REQUIRE_EQUAL(sm.dimat.cols(), p2_mesh_dimat_ref.cols());
        BOOST_TEST_CHECK(
            dimat_adiff.maxCoeff() == 0,
            (boost::format{"dimat test for StructuredMesh(nx=%d, nz=%d, degree=%d, cell_type=%s) failed:"}
            % nx % nz % deg % element).str()
        );
    }
}

BOOST_AUTO_TEST_CASE(test_domain_ids) {
    BOOST_TEST_CHECK(
        (NORTH_ID | NORTH_WEST_ID | WEST_ID | SOUTH_WEST_ID | SOUTH_ID | SOUTH_EAST_ID | EAST_ID | NORTH_EAST_ID | INTERIOR_ID) == DOMAIN_ID,
        "(NORTH_ID | NORTH_WEST_ID | WEST_ID | SOUTH_WEST_ID | SOUTH_ID | SOUTH_EAST_ID | EAST_ID | NORTH_EAST_ID | INTERIOR_ID) != DOMAIN_ID"
    );
    BOOST_TEST_CHECK(
        (DOMAIN_ID & NORTH_ID) == NORTH_ID,
        "(DOMAIN_ID & NORTH_ID) != NORTH_ID"
    );
    BOOST_TEST_CHECK(
        (DOMAIN_ID & WEST_ID) == WEST_ID,
        "(DOMAIN_ID & WEST_ID) != WEST_ID"
    );
    BOOST_TEST_CHECK(
        (DOMAIN_ID & SOUTH_ID) == SOUTH_ID,
        "(DOMAIN_ID & SOUTH_ID) != SOUTH_ID"
    );
    BOOST_TEST_CHECK(
        (DOMAIN_ID & EAST_ID) == EAST_ID,
        "(DOMAIN_ID & EAST_ID) != EAST_ID"
    );
    BOOST_TEST_CHECK(
        (DOMAIN_ID & INTERIOR_ID) == INTERIOR_ID,
        "(DOMAIN_ID & INTERIOR_ID) != INTERIOR_ID"
    );
    BOOST_TEST_CHECK(
        (INTERIOR_ID & BOUNDARY_ID) == EMPTY_ID,
        "(INTERIOR_ID & BOUNDARY_ID) != EMPTY_ID"
    );
    BOOST_TEST_CHECK(
        (DOMAIN_ID & ~BOUNDARY_ID) == INTERIOR_ID,
        "(DOMAIN_ID & ~BOUNDARY_ID) != INTERIOR_ID"
    );
    BOOST_TEST_CHECK(
        (BOUNDARY_ID & ~(NORTH_ID | NORTH_WEST_ID | WEST_ID | SOUTH_WEST_ID | SOUTH_ID | SOUTH_EAST_ID | EAST_ID | NORTH_EAST_ID)) == EMPTY_ID,
        "(BOUNDARY_ID & ~(NORTH_ID | NORTH_WEST_ID | WEST_ID | SOUTH_WEST_ID | SOUTH_ID | SOUTH_EAST | EAST_ID | NORTH_EAST_ID)) != EMPTY_ID"
    );
    BOOST_TEST_CHECK(
        ((BOUNDARY_ID & ~NORTH_ID) | NORTH_ID) == BOUNDARY_ID,
        "((BOUNDARY_ID & ~NORTH_ID) | NORTH_ID) != BOUNDARY_ID"
    );
    BOOST_TEST_CHECK(
        (BOUNDARY_ID & ~WEST_ID) == NORTH_ID | NORTH_WEST_ID | SOUTH_WEST_ID | SOUTH_ID | SOUTH_EAST_ID | EAST_ID | NORTH_EAST_ID,
        "(BOUNDARY_ID & ~WEST_ID) != NORTH_ID | NORTH_WEST_ID | SOUTH_WEST_ID | SOUTH_ID | SOUTH_EAST_ID | EAST_ID | NORTH_EAST_ID"
    );

    BOOST_TEST_CHECK(
        (DOMAIN_ID & ~INTERIOR_ID & ~(SOUTH_ID | WEST_ID)) == NORTH_ID | NORTH_WEST_ID | SOUTH_WEST_ID | SOUTH_EAST_ID | EAST_ID | NORTH_EAST_ID,
        "(DOMAIN_ID & ~INTERIOR_ID & ~(SOUTH_ID | WEST_ID)) != NORTH_ID | NORTH_WEST_ID | SOUTH_WEST_ID | SOUTH_EAST_ID | EAST_ID | NORTH_EAST_ID"
    );
}

BOOST_AUTO_TEST_CASE(test_dof_extraction)
{
    int nx = 2;
    int nz = 2;
    int deg = 2;
    int cell_type = QUADRILATERAL;
    StructuredMesh sm(nx, nz, deg, cell_type);
    std::vector<int> extracted_dofs;
    extracted_dofs = sm.extract_dof_inds(NORTH_ID);
    Eigen::Map<Eigen::VectorXi> north_boundary_dofs(extracted_dofs.data(), extracted_dofs.size());
    BOOST_TEST_CHECK(
        north_boundary_dofs == north_boundary_dofs_ref,
        (boost::format{"StructuredMesh(nx=%d, nz=%d, degree=%d, cell_type=%s).extract_dof_inds(NORTH_ID) failed:"}
        % nx % nz % deg % cell_type).str()
    );

    extracted_dofs = sm.extract_dof_inds(WEST_ID);
    Eigen::Map<Eigen::VectorXi> west_boundary_dofs(extracted_dofs.data(), extracted_dofs.size());
    BOOST_TEST_CHECK(
        west_boundary_dofs == west_boundary_dofs_ref,
        (boost::format{"StructuredMesh(nx=%d, nz=%d, degree=%d, cell_type=%s).extract_dof_inds(WEST_ID) failed:"}
        % nx % nz % deg % cell_type).str()
    );

    extracted_dofs = sm.extract_dof_inds(SOUTH_ID);
    Eigen::Map<Eigen::VectorXi> south_boundary_dofs(extracted_dofs.data(), extracted_dofs.size());
    BOOST_TEST_CHECK(
        south_boundary_dofs == south_boundary_dofs_ref,
        (boost::format{"StructuredMesh(nx=%d, nz=%d, degree=%d, cell_type=%s).extract_dof_inds(SOUTH_ID) failed:"}
        % nx % nz % deg % cell_type).str()
    );

    extracted_dofs = sm.extract_dof_inds(EAST_ID);
    Eigen::Map<Eigen::VectorXi> east_boundary_dofs(extracted_dofs.data(), extracted_dofs.size());
    BOOST_TEST_CHECK(
        east_boundary_dofs == east_boundary_dofs_ref,
        (boost::format{"StructuredMesh(nx=%d, nz=%d, degree=%d, cell_type=%s).extract_dof_inds(EAST_ID) failed:"}
        % nx % nz % deg % cell_type).str()
    );

    extracted_dofs = sm.extract_dof_inds(BOUNDARY_ID);
    Eigen::Map<Eigen::VectorXi> boundary_dofs(extracted_dofs.data(), extracted_dofs.size());
    BOOST_TEST_CHECK(
        boundary_dofs == boundary_dofs_ref,
        (boost::format{"StructuredMesh(nx=%d, nz=%d, degree=%d, cell_type=%s).extract_dof_inds(BOUNDARY_ID) failed:"}
        % nx % nz % deg % cell_type).str()
    );

    extracted_dofs = sm.extract_dof_inds(INTERIOR_ID);
    Eigen::Map<Eigen::VectorXi> interior_dofs(extracted_dofs.data(), extracted_dofs.size());
    BOOST_TEST_CHECK(
        interior_dofs == interior_dofs_ref,
        (boost::format{"StructuredMesh(nx=%d, nz=%d, degree=%d, cell_type=%s).extract_dof_inds(INTERIOR_ID) failed:"}
        % nx % nz % deg % cell_type).str()
    );

    extracted_dofs = sm.extract_dof_inds(DOMAIN_ID);
    Eigen::Map<Eigen::VectorXi> domain_dofs(extracted_dofs.data(), extracted_dofs.size());
    BOOST_TEST_CHECK(
        domain_dofs == domain_dofs_ref,
        (boost::format{"StructuredMesh(nx=%d, nz=%d, degree=%d, cell_type=%s).extract_dof_inds(DOMAIN_ID) failed:"}
        % nx % nz % deg % cell_type).str()
    );

    extracted_dofs = sm.extract_dof_inds(INTERIOR_ID | NORTH_ID);
    Eigen::Map<Eigen::VectorXi> i_or_nb_dofs(extracted_dofs.data(), extracted_dofs.size());
    BOOST_TEST_CHECK(
        i_or_nb_dofs == i_or_nb_dofs_ref,
        (boost::format{"StructuredMesh(nx=%d, nz=%d, degree=%d, cell_type=%s).extract_dof_inds(INTERIOR_ID | NORTH_ID) failed:"}
        % nx % nz % deg % cell_type).str()
    );
    extracted_dofs = sm.extract_dof_inds(DOMAIN_ID & ~INTERIOR_ID & ~(SOUTH_ID | WEST_ID));
    Eigen::Map<Eigen::VectorXi> b_not_sb_not_wb_dofs(extracted_dofs.data(), extracted_dofs.size());
    BOOST_TEST_CHECK(
        b_not_sb_not_wb_dofs == b_not_sb_not_wb_dofs_ref,
        (boost::format{"StructuredMesh(nx=%d, nz=%d, degree=%d, cell_type=%s).extract_dof_inds(DOMAIN_ID & ~INTERIOR_ID & ~(SOUTH_ID | WEST_ID)) failed:"}
        % nx % nz % deg % cell_type).str()
    );
}

BOOST_AUTO_TEST_CASE(test_vertex_dof_extraction)
{
    int nx = 2;
    int nz = 2;
    int deg = 2;
    int cell_type = QUADRILATERAL;
    StructuredMesh sm(nx, nz, deg, cell_type);
    std::vector<int> extracted_dofs;

    extracted_dofs = sm.extract_vertex_dof_inds(NORTH_ID);
    Eigen::Map<Eigen::VectorXi> north_boundary_vdofs(extracted_dofs.data(), extracted_dofs.size());
    BOOST_TEST_CHECK(
        north_boundary_vdofs == north_boundary_vdofs_ref,
        (boost::format{"StructuredMesh(nx=%d, nz=%d, degree=%d, cell_type=%s).extract_vertex_dof_inds(NORTH_ID) failed:"}
        % nx % nz % deg % cell_type).str()
    );

    extracted_dofs = sm.extract_vertex_dof_inds(SOUTH_ID);
    Eigen::Map<Eigen::VectorXi> south_boundary_vdofs(extracted_dofs.data(), extracted_dofs.size());
    BOOST_TEST_CHECK(
        south_boundary_vdofs == south_boundary_vdofs_ref,
        (boost::format{"StructuredMesh(nx=%d, nz=%d, degree=%d, cell_type=%s).extract_vertex_dof_inds(SOUTH_ID) failed:"}
        % nx % nz % deg % cell_type).str()
    );

    extracted_dofs = sm.extract_vertex_dof_inds(EAST_ID);
    Eigen::Map<Eigen::VectorXi> east_boundary_vdofs(extracted_dofs.data(), extracted_dofs.size());
    BOOST_TEST_CHECK(
        east_boundary_vdofs == east_boundary_vdofs_ref,
        (boost::format{"StructuredMesh(nx=%d, nz=%d, degree=%d, cell_type=%s).extract_vertex_dof_inds(EAST_ID) failed:"}
        % nx % nz % deg % cell_type).str()
    );

    extracted_dofs = sm.extract_vertex_dof_inds(BOUNDARY_ID);
    Eigen::Map<Eigen::VectorXi> boundary_vdofs(extracted_dofs.data(), extracted_dofs.size());
    BOOST_TEST_CHECK(
        boundary_vdofs == boundary_vdofs_ref,
        (boost::format{"StructuredMesh(nx=%d, nz=%d, degree=%d, cell_type=%s).extract_vertex_dof_inds(BOUNDARY_ID) failed:"}
        % nx % nz % deg % cell_type).str()
    );

    extracted_dofs = sm.extract_vertex_dof_inds(INTERIOR_ID);
    Eigen::Map<Eigen::VectorXi> interior_vdofs(extracted_dofs.data(), extracted_dofs.size());
    BOOST_TEST_CHECK(
        interior_vdofs == interior_vdofs_ref,
        (boost::format{"StructuredMesh(nx=%d, nz=%d, degree=%d, cell_type=%s).extract_vertex_dof_inds(INTERIOR_ID) failed:"}
        % nx % nz % deg % cell_type).str()
    );

    extracted_dofs = sm.extract_vertex_dof_inds(DOMAIN_ID);
    Eigen::Map<Eigen::VectorXi> domain_vdofs(extracted_dofs.data(), extracted_dofs.size());
    BOOST_TEST_CHECK(
        domain_vdofs == domain_vdofs_ref,
        (boost::format{"StructuredMesh(nx=%d, nz=%d, degree=%d, cell_type=%s).extract_vertex_dof_inds(DOMAIN_ID) failed:"}
        % nx % nz % deg % cell_type).str()
    );

    extracted_dofs = sm.extract_vertex_dof_inds(INTERIOR_ID | NORTH_ID);
    Eigen::Map<Eigen::VectorXi> i_or_nb_vdofs(extracted_dofs.data(), extracted_dofs.size());
    BOOST_TEST_CHECK(
        i_or_nb_vdofs == i_or_nb_vdofs_ref,
        (boost::format{"StructuredMesh(nx=%d, nz=%d, degree=%d, cell_type=%s).extract_vertex_dof_inds(INTERIOR_ID | NORTH_ID) failed:"}
        % nx % nz % deg % cell_type).str()
    );
    extracted_dofs = sm.extract_vertex_dof_inds(DOMAIN_ID & ~INTERIOR_ID & ~(NORTH_WEST_ID | WEST_ID | SOUTH_WEST_ID | SOUTH_ID | SOUTH_EAST_ID | NORTH_EAST_ID));
    Eigen::Map<Eigen::VectorXi> nb_or_eb_vdofs(extracted_dofs.data(), extracted_dofs.size());
    BOOST_TEST_CHECK(
        nb_or_eb_vdofs == nb_or_eb_vdofs_ref,
        (boost::format{"StructuredMesh(nx=%d, nz=%d, degree=%d, cell_type=%s).extract_vertex_dof_inds(DOMAIN_ID & ~INTERIOR_ID & ~(NORTH_WEST_ID | WEST_ID | SOUTH_WEST_ID | SOUTH_ID | SOUTH_EAST_ID | NORTH_EAST_ID)) failed:"}
        % nx % nz % deg % cell_type).str()
    );
}
