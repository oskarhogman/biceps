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
#include <pstokes_fem.hpp>
#include <stokes_test_suite.hpp>
#include <cmath>
#include <boost/format.hpp>

BOOST_AUTO_TEST_CASE(test_fluctuating_flow)
{
    FloatType eta_cavity = 1.0;
    FloatType A = 1.0/(2.0*eta_cavity);
    FloatType n_i = 1.0;
    FloatType eps_reg_2 = 1e-10;
    Eigen::MatrixX<FloatType> xz_ux_mat, xz_uz_mat, xz_p_mat;
    Eigen::VectorX<FloatType> X_vec, Z_vec, UX_vec, UZ_vec,
        P_vec, UX_vec_ref, UZ_vec_ref;

    populate_manufactured_sol_cases();

    for (
        std::tuple<int, int, int, FloatType, FloatType> test_case:
        test_manufactured_sol_cases
    ) {
        int nx = std::get<0>(test_case);
        int nz = std::get<1>(test_case);
        int cell_type = std::get<2>(test_case);
        FloatType error_ux_tol = std::get<3>(test_case);
        FloatType error_uz_tol = std::get<4>(test_case);
        StructuredMesh u_mesh(nx, nz, 2, cell_type);
        StructuredMesh p_mesh(nx, nz, 1, cell_type);
        pStokesProblem psp(A, n_i, eps_reg_2, force_x, force_z, u_mesh, p_mesh);
        psp.assemble_stress_block();
        psp.assemble_incomp_block();
        psp.assemble_rhs_vec();
        psp.commit_lhs_mat();

        psp.apply_dirichlet_bc(NORTH_ID, HORIZONTAL, &ux_func);
        psp.apply_dirichlet_bc(NORTH_WEST_ID, HORIZONTAL, &ux_func);
        psp.apply_dirichlet_bc(WEST_ID, HORIZONTAL, &ux_func);
        psp.apply_dirichlet_bc(SOUTH_WEST_ID, HORIZONTAL, &ux_func);
        psp.apply_dirichlet_bc(SOUTH_ID, HORIZONTAL, &ux_func);
        psp.apply_dirichlet_bc(SOUTH_EAST_ID, HORIZONTAL, &ux_func);
        psp.apply_dirichlet_bc(EAST_ID, HORIZONTAL, &ux_func);
        psp.apply_dirichlet_bc(NORTH_EAST_ID, HORIZONTAL, &ux_func);

        psp.apply_dirichlet_bc(NORTH_ID, VERTICAL, &uz_func);
        psp.apply_dirichlet_bc(NORTH_WEST_ID, VERTICAL, &uz_func);
        psp.apply_dirichlet_bc(WEST_ID, VERTICAL, &uz_func);
        psp.apply_dirichlet_bc(SOUTH_WEST_ID, VERTICAL, &uz_func_bed);
        psp.apply_dirichlet_bc(SOUTH_ID, VERTICAL, &uz_func_bed);
        psp.apply_dirichlet_bc(SOUTH_EAST_ID, VERTICAL, &uz_func_bed);
        psp.apply_dirichlet_bc(EAST_ID, VERTICAL, &uz_func);
        psp.apply_dirichlet_bc(NORTH_EAST_ID, VERTICAL, &uz_func);
        psp.prune_lhs(1e-12);

        psp.solve_linear_system();

        xz_ux_mat = psp.velocity_x().extract_vertex_subvec(DOMAIN_ID);
        xz_uz_mat = psp.velocity_z().extract_vertex_subvec(DOMAIN_ID);
        xz_p_mat = psp.pressure().extract_vertex_subvec(DOMAIN_ID);

        X_vec = xz_ux_mat(Eigen::all, 0);
        Z_vec = xz_ux_mat(Eigen::all, 1);
        UX_vec = xz_ux_mat(Eigen::all, 2);
        UZ_vec = xz_uz_mat(Eigen::all, 2);
        P_vec = xz_p_mat(Eigen::all, 2);

        UX_vec_ref = Eigen::VectorX<FloatType>::Zero(UX_vec.size());
        UZ_vec_ref = Eigen::VectorX<FloatType>::Zero(UZ_vec.size());
        for (int i = 0; i < X_vec.size(); ++i) {
            UX_vec_ref(i) = ux_func(X_vec(i), Z_vec(i));
            UZ_vec_ref(i) = uz_func(X_vec(i), Z_vec(i));
        }

        FloatType error_ux_inf_norm = (UX_vec_ref - UX_vec).cwiseAbs().maxCoeff();
        FloatType error_uz_inf_norm = (UZ_vec_ref - UZ_vec).cwiseAbs().maxCoeff();
        BOOST_TEST_CHECK(
            error_ux_inf_norm < error_ux_tol,
            (boost::format{"Stokes velocity X error test for StructuredMesh(nx=%d, nz=%d, degree=P2P1, cell_type=%s) failed: ||error_ux||_inf = %g > %g"}
            % nx % nz % cell_type %error_ux_inf_norm %error_ux_tol).str()
        );
        BOOST_TEST_CHECK(
            error_uz_inf_norm < error_uz_tol,
            (boost::format{"Stokes velocity Z error test for StructuredMesh(nx=%d, nz=%d, degree=P2P1, cell_type=%s) failed: ||error_uz||_inf = %g > %g"}
            % nx % nz % cell_type %error_uz_inf_norm %error_uz_tol).str()
        );
    }
}
