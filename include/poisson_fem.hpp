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
#define EIGEN_SPARSEMATRIX_PLUGIN <eigen_spmat_addons.hpp>

#include <vector>
#include <functional>
#include <fem_function_2d.hpp>
#include <Eigen/Sparse>

/**
 * @class PoissonProblem
 * @brief Solves the Poisson equation using the finite element method (FEM).
 *
 * This class constructs and solves the weak form of the Poisson equation on a given structured mesh.
 * It supports various boundary conditions and allows assembly of stiffness, mass, Neumann, and Robin terms.
 */
class PoissonProblem {

private:
    int n_dofs; ///< Number of degrees of freedom in the system.

public:
    std::vector<int> free_dofs; ///< Indices of free degrees of freedom.
    std::vector<int> fixed_dofs; ///< Indices of fixed degrees of freedom (Dirichlet boundary conditions).

    /**
     * @brief Constructs a PoissonProblem instance on a given structured mesh.
     * @param mesh The finite element mesh to be used for solving the problem.
     */
    explicit PoissonProblem(StructuredMesh &mesh);

    StructuredMesh &mesh; ///< Reference to the structured mesh.

    std::vector<Eigen::Triplet<FloatType>> lhs_coeffs; ///< Storage for triplet coefficients of the system matrix.

    Eigen::SparseMatrix<FloatType> lhs_mat; ///< Global left-hand-side (LHS) matrix.
    Eigen::SparseMatrix<FloatType> lhs_mat_free; ///< LHS matrix for free degrees of freedom.
    Eigen::SparseMatrix<FloatType> lhs_mat_fixed; ///< LHS matrix for fixed degrees of freedom.

    Eigen::VectorX<FloatType> bc_vec; ///< Vector containing boundary condition values.
    Eigen::VectorX<FloatType> rhs_vec; ///< Right-hand-side (RHS) vector of the system.
    Eigen::VectorX<FloatType> rhs_vec_free; ///< RHS vector restricted to free DOFs.
    Eigen::VectorX<FloatType> sol_vec_free; ///< Solution vector for free DOFs.
    Eigen::VectorX<FloatType> sol_vec; ///< Complete solution vector.

    /**
     * @brief Assembles the stiffness matrix using given material properties.
     * @param alpha Diffusion coefficient function.
     * @param beta Reaction coefficient function.
     * @param gp Number of Gauss points used for integration.
     */
    void assemble_stiffness_block(
        std::function<FloatType(FloatType, FloatType)> alpha,
        std::function<FloatType(FloatType, FloatType)> beta,
        int gp
    );

    /**
     * @brief Assembles the mass matrix using a given density function.
     * @param gamma Mass coefficient function.
     * @param gauss_precision Number of Gauss points used for integration.
     */
    void assemble_mass_block(
        std::function<FloatType(FloatType, FloatType)> gamma,
        int gauss_precision
    );

    /**
     * @brief Assembles the Robin boundary condition contribution to the system matrix.
     * @param a_robin Coefficient for the Robin term.
     * @param b_robin Coefficient for the boundary term.
     * @param g_robin Function representing the boundary condition.
     * @param boundary_id Identifier for the boundary segment.
     * @param gauss_precision Number of Gauss points used for integration.
     */
    void assemble_robin_block(
        std::function<FloatType(FloatType, FloatType)> a_robin,
        std::function<FloatType(FloatType, FloatType)> b_robin,
        std::function<FloatType(FloatType, FloatType)> g_robin,
        int boundary_id,
        int gauss_precision
    );

    /**
     * @brief Converts the assembled triplet list into a sparse matrix format.
     */
    void commit_lhs_mat();

    /**
     * @brief Assembles the force term in the right-hand-side vector.
     * @param f Function representing the source term.
     * @param gauss_precision Number of Gauss points used for integration.
     */
    void assemble_force_rhs(
        std::function<FloatType(FloatType, FloatType)> f,
        int gauss_precision
    );

    /**
     * @brief Overloaded function for assembling the force term using a FEMFunction2D.
     * @param f FEMFunction2D object representing the source term.
     * @param gauss_precision Number of Gauss points used for integration.
     */
    void assemble_force_rhs(FEMFunction2D &f, int gauss_precision);

    /**
     * @brief Assembles the Neumann boundary condition contribution to the RHS vector.
     * @param g_neumann Function representing the Neumann condition.
     * @param boundary_id Identifier for the boundary segment.
     * @param gauss_precision Number of Gauss points used for integration.
     */
    void assemble_neumann_rhs(
        std::function<FloatType(FloatType, FloatType)> g_neumann,
        int boundary_id,
        int gauss_precision
    );

    /**
     * @brief Assembles the Robin boundary condition contribution to the RHS vector.
     * @param b_robin Coefficient function for the Robin term.
     * @param g_robin Function representing the boundary condition.
     * @param boundary_id Identifier for the boundary segment.
     * @param gauss_precision Number of Gauss points used for integration.
     */
    void assemble_robin_rhs(
        std::function<FloatType(FloatType, FloatType)> b_robin,
        std::function<FloatType(FloatType, FloatType)> g_robin,
        int boundary_id,
        int gauss_precision
    );

    /**
     * @brief Applies a Dirichlet boundary condition to a specified boundary.
     * @param g_dirichlet Function representing the Dirichlet condition.
     * @param boundary_id Identifier for the boundary segment.
     */
    void apply_dirichlet_bc(
        std::function<FloatType(FloatType, FloatType)> g_dirichlet,
        int boundary_id
    );

    /**
     * @brief Solves the linear system of equations for the FEM solution.
     */
    void solve_linear_system();

    /**
     * @brief Retrieves the computed solution as a FEM function.
     * @return FEMFunction2D representing the solution.
     */
    FEMFunction2D solution();

    /**
     * @brief Resets the system matrices and vectors.
     */
    void reset_system();
};

