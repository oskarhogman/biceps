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
#include <tuple>
#include <boost/format.hpp>
#include <poisson_test_suite.hpp>

BOOST_AUTO_TEST_CASE(test_manufactured_solution_case_1)
{
    populate_manufactured_sol_cases();	
    for (std::tuple<int, int, int, int, FloatType> test_case: test_manufactured_sol_cases) {
        int nx = std::get<0>(test_case);
        int nz = std::get<1>(test_case);
        int deg = std::get<2>(test_case);
        int cell_type = std::get<3>(test_case);
        FloatType error_tol = std::get<4>(test_case);
        StructuredMesh mesh(nx, nz, deg, cell_type);
        PoissonProblem pp = PoissonProblem(mesh);
        pp.assemble_stiffness_block(&alpha_manufactured_case_2, &beta_manufactured_case_2, 3);
        pp.commit_lhs_mat();
        pp.assemble_force_rhs(&force_manufactured_case_1, 2);
        pp.apply_dirichlet_bc(&bc_func_manufactured_case_1, BOUNDARY_ID);
        pp.solve_linear_system();
        Eigen::VectorX<FloatType> x_vec = pp.mesh.pmat(Eigen::all, 0);
        Eigen::VectorX<FloatType> z_vec = pp.mesh.pmat(Eigen::all, 1);

        Eigen::VectorX<FloatType> ue_vec = Eigen::VectorX<FloatType>::Zero(x_vec.size());
        for (int dof = 0; dof < x_vec.size(); ++dof) {
            ue_vec(dof) = exact_sol_func_manufactured_case_1(
                x_vec(dof), z_vec(dof)
            );
        }
        Eigen::VectorX<FloatType> uh_vec = pp.sol_vec;
        FloatType error_inf_norm = (ue_vec - uh_vec).cwiseAbs().maxCoeff();
        BOOST_TEST_CHECK(
            error_inf_norm < error_tol,
            (boost::format{"Poisson solution error test with (alpha=1, beta=1, gamma=0) for StructuredMesh(nx=%d, nz=%d, degree=%d, cell_type=%s) failed: ||error_u||_inf = %g > %g"}
            % nx % nz % deg % cell_type %error_inf_norm %error_tol).str()
        );
    }
}

BOOST_AUTO_TEST_CASE(test_manufactured_solution_case_2)
{
    populate_manufactured_sol_cases();
    for (std::tuple<int, int, int, int, FloatType> test_case: test_manufactured_sol_cases) {
        int nx = std::get<0>(test_case);
        int nz = std::get<1>(test_case);
        int deg = std::get<2>(test_case);
        int cell_type = std::get<3>(test_case);
        FloatType error_tol = std::get<4>(test_case);
        StructuredMesh mesh(nx, nz, deg, cell_type);
        PoissonProblem pp = PoissonProblem(mesh);
        pp.assemble_stiffness_block(&alpha_manufactured_case_2, &beta_manufactured_case_2, 2);
        pp.assemble_mass_block(&gamma_manufactured_case_2, 3);
        pp.commit_lhs_mat();
        pp.assemble_force_rhs(&force_manufactured_case_2, 2);
        pp.apply_dirichlet_bc(&bc_func_manufactured_case_2, BOUNDARY_ID);
        pp.solve_linear_system();
        Eigen::VectorX<FloatType> x_vec = pp.mesh.pmat(Eigen::all, 0);
        Eigen::VectorX<FloatType> z_vec = pp.mesh.pmat(Eigen::all, 1);

        Eigen::VectorX<FloatType> ue_vec = Eigen::VectorX<FloatType>::Zero(x_vec.size());
        for (int dof = 0; dof < x_vec.size(); ++dof) {
            ue_vec(dof) = exact_sol_func_manufactured_case_2(
                x_vec(dof), z_vec(dof)
            );
        }
        Eigen::VectorX<FloatType> uh_vec = pp.sol_vec;
        FloatType error_inf_norm = (ue_vec - uh_vec).cwiseAbs().maxCoeff();
        BOOST_TEST_CHECK(
            error_inf_norm < error_tol,
            (boost::format{"Poisson solution error test with (alpha=1, beta=1, gamma=1) for StructuredMesh(nx=%d, nz=%d, degree=%d, cell_type=%s) failed: ||error_u||_inf = %g > %g"}
            % nx % nz % deg % cell_type %error_inf_norm %error_tol).str()
        );
    }
}

BOOST_AUTO_TEST_CASE(test_manufactured_solution_case_3)
{
    populate_manufactured_sol_cases();
    for (std::tuple<int, int, int, int, FloatType> test_case: test_manufactured_sol_cases) {
        int nx = std::get<0>(test_case);
        int nz = std::get<1>(test_case);
        int deg = std::get<2>(test_case);
        int cell_type = std::get<3>(test_case);
        FloatType error_tol = std::get<4>(test_case);
        StructuredMesh mesh(nx, nz, deg, cell_type);
        PoissonProblem pp = PoissonProblem(mesh);
        pp.assemble_stiffness_block(&alpha_manufactured_case_3, &beta_manufactured_case_3, 2);
        pp.assemble_mass_block(&gamma_manufactured_case_3, 3);
        pp.commit_lhs_mat();
        pp.assemble_force_rhs(&force_manufactured_case_3, 2);
        pp.apply_dirichlet_bc(&bc_func_manufactured_case_3, BOUNDARY_ID);
        pp.solve_linear_system();
        Eigen::VectorX<FloatType> x_vec = pp.mesh.pmat(Eigen::all, 0);
        Eigen::VectorX<FloatType> z_vec = pp.mesh.pmat(Eigen::all, 1);

        Eigen::VectorX<FloatType> ue_vec = Eigen::VectorX<FloatType>::Zero(x_vec.size());
        for (int dof = 0; dof < x_vec.size(); ++dof) {
            ue_vec(dof) = exact_sol_func_manufactured_case_3(
                x_vec(dof), z_vec(dof)
            );
        }
        Eigen::VectorX<FloatType> uh_vec = pp.sol_vec;
        FloatType error_inf_norm = (ue_vec - uh_vec).cwiseAbs().maxCoeff();
        BOOST_TEST_CHECK(
            error_inf_norm < error_tol,
            (boost::format{"Poisson solution error test (alpha=5, beta=3, gamma=-2) for StructuredMesh(nx=%d, nz=%d, degree=%d, cell_type=%s) failed: ||error_u||_inf = %g > %g"}
            % nx % nz % deg % cell_type %error_inf_norm %error_tol).str()
        );
    }
}
