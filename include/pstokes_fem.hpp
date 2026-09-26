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
#include <logger.hpp>
#include <fem_function_2d.hpp>
#include <enums.hpp>

/**
 * @class pStokesProblem
 * @brief A class to solve the 2D Stokes problem using the finite element method.
 * 
 * This class provides functionality for assembling the system of equations, 
 * applying boundary conditions, solving linear and nonlinear systems, 
 * and storing the results for velocity and pressure fields.
 */
class pStokesProblem {

private:
    int nv_dofs;  ///< Number of velocity degrees of freedom
    int np_dofs;  ///< Number of pressure degrees of freedom
    int n_dofs;  ///< Total number of degrees of freedom
    Logger logger;  ///< Logger object for logging messages

public:
    FloatType rate_factor;  ///< Rate factor
    FloatType glen_exponent;  ///< Glen exponent
    FloatType eps_reg_2;  ///< Regularization term in Glen's flow law
    int fssa_version = FSSA_NONE;  ///< FSSA version to use by default
    FloatType fssa_param = 0.0;  ///< Free-surface stabilization parameter
    int gp_stress = 5;  ///< Gauss precision for stress block
    int gp_incomp = 5;  ///< Gauss precision for incompressibility block
    int gp_fssa_lhs = 5;  ///< Gauss precision for lhs FSSA
    int gp_fssa_rhs = 5;  ///< Gauss precision for rhs FSSA
    int gp_rhs = 5;  /// Gauss precision for rhs vector

    std::function<FloatType(FloatType, FloatType)> force_x;  ///< Body force in x
    std::function<FloatType(FloatType, FloatType)> force_z;  ///< Body force in z
    std::function<FloatType(FloatType, FloatType)> fssa_accum;  ///< Body force in z

    /**
     * @brief Bitmask of domain IDs for applying Dirichlet boundary conditions on horizontal velocity (ux).
     *
     * Combine multiple domains with bitwise OR (|). Use bitwise AND (&) and NOT (~) for filtering.
     */
    int ux_dirichlet_bc_mask = 0;

    /**
     * @brief Bitmask of domain IDs for applying Dirichlet boundary conditions on vertical velocity (uz).
     *
     * Combine multiple domains with bitwise OR (|). Use bitwise AND (&) and NOT (~) for filtering.
     */
    int uz_dirichlet_bc_mask = 0;

    std::vector<Eigen::Triplet<FloatType>> lhs_coeffs;  ///< Left-hand side coefficients (stiffness matrix)
    int nof_pushed_elements = 0;  ///< Counter for the number of elements pushed during assembly
    FloatType picard_stol = 1e-8;  ///< Step tolerance for Picard iterations
    FloatType picard_max_iter = 100;  ///< Maximum number of Picard iterations

    StructuredMesh &u_mesh;  ///< Mesh for velocity field (horizontal and vertical)
    StructuredMesh &p_mesh;  ///< Mesh for pressure field

    Eigen::VectorXi ux_v2d;  ///< Velocity vector (horizontal)
    Eigen::VectorXi uz_v2d;  ///< Velocity vector (vertical)
    Eigen::VectorXi u_v2d;  ///< Combined velocity vector
    Eigen::VectorXi p_v2d;  ///< Pressure vector

    Eigen::VectorXi ux_d2v;  ///< Velocity degrees of freedom mapping (horizontal)
    Eigen::VectorXi uz_d2v;  ///< Velocity degrees of freedom mapping (vertical)
    Eigen::VectorXi p_d2v;  ///< Pressure degrees of freedom mapping
    Eigen::VectorXi w_d2v;  ///< Mapping for the solution vector (unknowns)

    Eigen::SparseMatrix<FloatType> lhs_mat;  ///< Left-hand side system matrix
    Eigen::VectorX<FloatType> rhs_vec;  ///< Right-hand side force vector
    Eigen::VectorX<FloatType> w_vec;  ///< Solution vector

    /**
     * @brief Constructor for initializing the pStokesProblem with meshes for velocity and pressure.
     * 
     * @param[in] rate_factor The ice softness parameter.
     * @param[in] glen_exponent The flow law exponent.
     * @param[in] eps_reg_2 The regularization parameter (avoids infinite viscosity).
     * @param[in] force_x The force function in the x-direction (as a function of coordinates).
     * @param[in] force_z The force function in the z-direction (as a function of coordinates).
     * @param[in] u_mesh Reference to the mesh for velocity field (both horizontal and vertical).
     * @param[in] p_mesh Reference to the mesh for pressure field.
     */
    pStokesProblem(
        FloatType rate_factor,
        FloatType glen_exponent,
        FloatType eps_reg_2,
        std::function<FloatType(FloatType, FloatType)> force_x,
        std::function<FloatType(FloatType, FloatType)> force_z,
        StructuredMesh &u_mesh,
        StructuredMesh &p_mesh
    );

    /**
     * @brief Assembles the stress block of the left-hand side matrix.
     */
    void assemble_stress_block();

    /**
     * @brief Assembles the incompressibility block of the left-hand side matrix.
     */
    void assemble_incomp_block();

    /**
     * @brief Assembles the FSSA vertical block of the left-hand side matrix.
     */
    void assemble_fssa_vertical_block();

    /**
     * @brief Assembles the FSSA normal block of the left-hand side matrix.
     */
    void assemble_fssa_normal_block();

    /**
     * @brief Commits the assembled left-hand side matrix to the system.
     */
    void commit_lhs_mat(); 

    /**
     * @brief Assembles the right-hand side vector based on external forces.
     */
    void assemble_rhs_vec();

    /**
     * @brief Assembles the FSSA vertical right-hand side vector with external forces.
     */
    void assemble_fssa_vertical_rhs();

    /**
     * @brief Applies zero Dirichlet boundary conditions on the velocity components.
     */
    void apply_zero_dirichlet_bc();

    /**
     * @brief Applies Dirichlet boundary conditions for a specific velocity component.
     * 
     * @param[in] boundary_part The ID for the boundary part.
     * @param[in] velocity_component The velocity component to which the boundary condition is applied.
     * @param[in] u_func The function representing the Dirichlet boundary condition.
     *
     * This function is scheduled for removal, but is kept around for compatibility with unit tests.
     */
    void apply_dirichlet_bc(
        const int boundary_part,
        const int velocity_component,
        std::function<FloatType(FloatType, FloatType)> u_func
    );

    /**
     * @brief Prunes small coefficients from the left-hand side matrix to improve sparsity.
     * 
     * @param[in] threshold The coefficient magnitude below which entries are pruned.
     */
    void prune_lhs(FloatType threshold);

    /**
     * @brief Solves the linear system Ax=b.
     */
    void solve_linear_system();

    /**
     * @brief Solves the nonlinear system A(x)x=b, by iteratively solving
     * the linear system A(x0)x=b until ||x - x0|| < stol ||x||,
     * where stol is the user defined step tolerance picard_max_stol.
     */
    void solve_nonlinear_system();

    /**
     * @brief Retrieves the horizontal velocity as a FEM function.
     * 
     * @return A FEMFunction2D object representing the horizontal velocity field.
     */
    FEMFunction2D velocity_x();

    /**
     * @brief Retrieves the veloctiy velocity as a FEM function.
     * @return FEMFunction2D representing the solution.
     */
    FEMFunction2D velocity_z();

    /**
     * @brief Retrieves the pressure as a FEM function.
     * @return FEMFunction2D representing the solution.
     */
    FEMFunction2D pressure();

    /**
     * @brief Retrieves the current log level of the logger.
     * 
     * @return The current log level.
     */
    int log_level();

    /**
     * @brief Sets the log level for the logger.
     * 
     * @param[in] value The log level to set.
     */
    void set_log_level(int value);

    /**
     * @brief Resizes the left-hand side matrix to specified size
     *
     * @param[in] size The size to resize to
     */
    void resize_lhs(int size);

    /**
     * @brief Resets the left-hand side matrix to its initial state
     */
    void reset_lhs();

    /**
     * @brief Resets the right-hand side vector to its initial state.
     */
    void reset_rhs();

    /**
     * @brief Resets the system matrices and vectors.
     */
    void reset_system();
};
