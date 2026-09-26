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

#include <fem_function_2d.hpp>
#include <structured_mesh.hpp>
#include <float_type.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>

/**
 * @namespace FEM2D
 * @brief Namespace for 2D Finite Element Method (FEM) functions and operations.
 */
namespace FEM2D {

    /**
     * @brief Performs Gauss-Legendre quadrature for integration over a given cell.
     *
     * This function computes the Gauss-Legendre quadrature points and weights for a given precision and cell type.
     *
     * @param[in] precision The number of quadrature points (degree of precision).
     * @param[in] cell_type The type of the cell (e.g., triangle, quadrilateral).
     * @param[out] points_ret The quadrature points in reference space.
     * @param[out] weights_ret The quadrature weights.
     */
    void gauss_legendre_quadrature(
        const int precision,
        const int cell_type,
        Eigen::MatrixX<FloatType> &points_ret,
        Eigen::VectorX<FloatType> &weights_ret
    );

    /**
     * @brief Maps quadrature points from reference space (r,s) to physical space (x,z).
     *
     * This function transforms quadrature points from the reference element in (r,s) coordinates to physical 
     * (x,z) coordinates.
     *
     * @param[in] node_coords The coordinates of the nodes in physical space.
     * @param[in] qpoints_rs The quadrature points in reference space.
     * @param[out] qpoints_xz_ret The quadrature points in physical space.
     */
    void map_rs_to_xz(
        Eigen::MatrixX<FloatType> &node_coords,
        Eigen::MatrixX<FloatType> &qpoints_rs,
        Eigen::MatrixX<FloatType> &qpoints_xz_ret
    );

    /**
     * @brief Maps quadrature points from physical space (x,z) to reference space (r,s).
     *
     * This function transforms quadrature points from physical (x,z) coordinates to the reference element 
     * in (r,s) coordinates.
     *
     * @param[in] node_coords The coordinates of the nodes in physical space.
     * @param[in] qpoints_xz The quadrature points in physical space.
     * @param[out] qpoints_rs_ret The quadrature points in reference space.
     */
    void map_xz_to_rs(
        Eigen::MatrixX<FloatType> &node_coords,
        Eigen::MatrixX<FloatType> &qpoints_xz,
        Eigen::MatrixX<FloatType> &qpoints_rs_ret
    );

    /**
     * @brief Computes the Lagrange basis functions and their gradients at quadrature points in reference space.
     *
     * This function calculates the Lagrange basis functions and their gradients at given quadrature points in 
     * the reference space.
     *
     * @param[in] degree The degree of the polynomial (for Lagrange basis functions).
     * @param[in] cell_type The type of the cell (e.g., triangle, quadrilateral).
     * @param[in] qpoints_rs The quadrature points in reference space.
     * @param[out] phi_rs_ret The Lagrange basis functions at the quadrature points.
     * @param[out] grad_phi_rs_ret The gradients of the Lagrange basis functions at the quadrature points.
     */
    void lagrange_basis(
        const int degree,
        const int cell_type,
        Eigen::MatrixX<FloatType> &qpoints_rs,
        Eigen::MatrixX<FloatType> &phi_rs_ret,
        Eigen::MatrixX<FloatType> &grad_phi_rs_ret
    );

    /**
     * @brief Maps quadrature points and shape functions from reference space to physical space.
     *
     * This function maps quadrature points and shape functions from the reference element to the physical space
     * using the given node coordinates.
     *
     * @param[in] degree The degree of the polynomial for the shape functions.
     * @param[in] cell_type The type of the cell (e.g., triangle, quadrilateral).
     * @param[in] node_coords The coordinates of the nodes in physical space.
     * @param[in] qpoints_rs The quadrature points in reference space.
     * @param[in] phi_rs The shape functions at the quadrature points in reference space.
     * @param[in] grad_phi_rs The gradients of the shape functions at the quadrature points in reference space.
     * @param[out] detJ_rs_ret The determinant of the Jacobian at each quadrature point.
     * @param[out] qpoints_xz_ret The quadrature points in physical space.
     * @param[out] grad_phi_xz_ret The gradients of the shape functions in physical space.
     */
    void map_to_reference_cell(
        const int degree, 
        const int cell_type,
        const Eigen::MatrixX<FloatType> &node_coords,
        const Eigen::MatrixX<FloatType> &qpoints_rs,
        const Eigen::MatrixX<FloatType> &phi_rs,
        const Eigen::MatrixX<FloatType> &grad_phi_rs,
        Eigen::VectorX<FloatType> &detJ_rs_ret,
        Eigen::MatrixX<FloatType> &qpoints_xz_ret,
        Eigen::MatrixX<FloatType> &grad_phi_xz_ret
    );

    /**
     * @brief Returns the reference element points in reference space (r,s).
     *
     * This function returns the quadrature points for a reference element based on the specified cell type and degree.
     *
     * @param[in] cell_type The type of the cell (e.g., triangle, quadrilateral).
     * @param[in] degree The degree of the polynomial (number of quadrature points).
     * @return Eigen::MatrixX<FloatType> The quadrature points in reference space.
     */
    Eigen::MatrixX<FloatType> reference_element_points_rs(int cell_type, int degree);

    /**
     * @brief Assembles the mass matrix for a 2D FEM problem.
     *
     * This function computes the global mass matrix for the finite element model by assembling local mass 
     * matrix contributions from each element using quadrature.
     *
     * @param[in] mesh The structured mesh containing the finite element data.
     * @param[in] gp The number of Gauss quadrature points to use.
     * @return Eigen::SparseMatrix<FloatType> The assembled sparse mass matrix.
     */
    Eigen::SparseMatrix<FloatType> assemble_mass_matrix(
        StructuredMesh &mesh, int gp
    );

    /**
     * @brief Assembles the stiffness matrix for the finite element method.
     *
     * This function constructs the stiffness matrix by integrating the gradient of basis functions
     * over the reference element and mapping them to the physical domain.
     *
     * @param[in] mesh The structured mesh representing the computational domain.
     * @param[in] gp The number of Gauss points used for numerical integration.
     * @return Eigen::SparseMatrix<FloatType> The assembled stiffness matrix.
     */
    Eigen::SparseMatrix<FloatType> assemble_stiffness_matrix(
        StructuredMesh &mesh, int gp
    );

    /**
     * @brief Assembles the xx-component of the stiffness matrix for the finite element method.
     *
     * This function constructs the stiffness matrix by integrating the gradient of basis functions
     * in the x-direction over the reference element and mapping them to the physical domain.
     *
     * @param[in] mesh The structured mesh representing the computational domain.
     * @param[in] gp The number of Gauss points used for numerical integration.
     * @return Eigen::SparseMatrix<FloatType> The assembled stiffness matrix in the xx-direction.
     */
    Eigen::SparseMatrix<FloatType> assemble_stiffness_xx_matrix(
        StructuredMesh &mesh, int gp
    );

    /**
     * @brief Assembles the zz-component of the stiffness matrix for the finite element method.
     *
     * This function constructs the stiffness matrix by integrating the gradient of basis functions
     * in the z-direction over the reference element and mapping them to the physical domain.
     *
     * @param[in] mesh The structured mesh representing the computational domain.
     * @param[in] gp The number of Gauss points used for numerical integration.
     * @return Eigen::SparseMatrix<FloatType> The assembled stiffness matrix in the zz-direction.
     */
    Eigen::SparseMatrix<FloatType> assemble_stiffness_zz_matrix(
        StructuredMesh &mesh, int gp
    );

    /**
     * @brief Assembles the x-component of the stiffness matrix for the finite element method.
     *
     * This function constructs the stiffness matrix in the x-direction by integrating the product 
     * of the basis functions and the x-derivatives of the basis functions over the reference element 
     * and mapping them to the physical domain. The result is then added to the global stiffness matrix.
     *
     * @param[in] mesh The structured mesh representing the computational domain.
     * @param[in] gp The number of Gauss points used for numerical integration.
     * @return Eigen::SparseMatrix<FloatType> The assembled stiffness matrix in the x-direction.
     */
    Eigen::SparseMatrix<FloatType> assemble_stiffness_x_matrix(
        StructuredMesh &mesh, int gp
    );

    /**
     * @brief Assembles the z-component of the stiffness matrix for the finite element method.
     *
     * This function constructs the stiffness matrix in the z-direction by integrating the product 
     * of the basis functions and the z-derivatives of the basis functions over the reference element 
     * and mapping them to the physical domain. The result is then added to the global stiffness matrix.
     *
     * @param[in] mesh The structured mesh representing the computational domain.
     * @param[in] gp The number of Gauss points used for numerical integration.
     * @return Eigen::SparseMatrix<FloatType> The assembled stiffness matrix in the z-direction.
     */
    Eigen::SparseMatrix<FloatType> assemble_stiffness_z_matrix(
        StructuredMesh &mesh, int gp
    );

    /**
     * @brief Assembles the expansion matrix for the 2D FEM system.
     * 
     * This function constructs the expansion matrix that relates the degrees of freedom (DOFs)
     * of the 2D mesh to the mesh heights. It performs numerical integration using Gauss-Legendre 
     * quadrature and computes the contributions of the basis functions at each quadrature point. 
     * The matrix is assembled by iterating over the mesh elements (both horizontal and vertical 
     * layers), and the force contributions are calculated based on the input force function.
     *
     * The expansion matrix is used to relate the vector of DOFs \( F \) to the vector of mesh heights \( h \) 
     * via the relation \( F = P h \), where \( P \) is the expansion matrix. This is essential for solving 
     * the finite element problem in the 2D case.
     * 
     * @param[in] mesh The structured mesh containing the information about nodes, cells, and degrees of freedom.
     * @param[in] force A function representing the force applied at each quadrature point in physical coordinates.
     * @param[in] gp The precision (number of quadrature points) used for Gauss-Legendre quadrature.
     * 
     * @return Eigen::MatrixX<FloatType> The assembled expansion matrix of size (nof_dofs x (nx+1)).
     *         The matrix relates the DOFs to the mesh heights for the finite element system.
     *
     * @note The number of quadrature points is defined by the precision parameter `gp`. 
     *       This function assumes that the mesh is structured with nx horizontal layers and nz vertical layers.
     *       The expansion matrix is assembled by evaluating the basis functions at each quadrature point 
     *       and accumulating the contributions of the force function.
     */
    Eigen::MatrixX<FloatType> assemble_expansion_matrix(
        StructuredMesh &mesh,
        std::function<FloatType(FloatType, FloatType)> force,
        int gp
    );

    /**
     * @brief Computes the inner product u^T * M * v.
     * 
     * @param[in] u_vec First vector.
     * @param[in] M Sparse mass matrix.
     * @param[in] v_vec Second vector.
     * @return FloatType Inner product result.
     */
    FloatType inner(
        const Eigen::VectorX<FloatType> &u_vec,
        const Eigen::SparseMatrix<FloatType> &M,
        const Eigen::VectorX<FloatType> &v_vec
    );
}
