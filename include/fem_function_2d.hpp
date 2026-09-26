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

#include <structured_mesh.hpp>
#include <Eigen/Sparse>

/**
 * @class FEMFunction2D
 * @brief A class for representing and manipulating 2D FEM functions.
 * 
 * This class encapsulates a 2D finite element function, providing functionality to
 * evaluate, assign, and manipulate function values. It also supports assembling
 * the mass matrix, computing L2 norms, and performing differentiation and projection
 * operations in the x and z directions.
 * 
 * The class allows for operations on FEM functions such as addition, subtraction,
 * multiplication, and evaluation at specific points or mesh cells.
 */
class FEMFunction2D {
private:
    Eigen::SparseMatrix<FloatType> M; /**< Sparse mass matrix */
    bool mass_mat_is_assembled = false; /**< Flag indicating whether mass matrix is assembled */

public:
    /**
     * @brief Constructor for FEMFunction2D.
     * 
     * Initializes the function based on the provided mesh. Optionally, it can
     * also assemble the mass matrix.
     * 
     * @param[in] mesh The structured mesh for the 2D function.
     * @param[in] assemble_mass Flag to determine if mass matrix is assembled on initialization.
     */
    explicit FEMFunction2D(StructuredMesh &mesh);

    StructuredMesh &mesh; /**< Reference to the structured mesh */
    Eigen::VectorX<FloatType> vals; /**< Vector of function values at each degree of freedom (DOF) */

    /**
     * @brief Assigns a function to the FEMFunction2D object.
     * 
     * @param[in] func The function to assign to the FEMFunction2D.
     */
    void assign(std::function<FloatType(FloatType, FloatType)> func);

    /**
     * @brief Assigns a vector of values to the FEMFunction2D object.
     * 
     * @param[in] vec The vector of values to assign.
     */
    void assign(const Eigen::VectorX<FloatType> &vec);

    /**
     * @brief Assigns another FEMFunction2D object to this one.
     * 
     * @param[in] f The FEMFunction2D object to assign.
     */
    void assign(const FEMFunction2D &f);

    /**
     * @brief Evaluates the function at the given x and z coordinates.
     * 
     * @param[in] x The x-coordinate.
     * @param[in] z The z-coordinate.
     * @return The function value at the (x, z) point.
     */
    FloatType eval(FloatType x, FloatType z);

    /**
     * @brief Evaluates the function over a specified cell.
     * 
     * @param[in] cell_index The index of the cell to evaluate.
     * @return Vector of function values for the cell.
     */
    Eigen::VectorX<FloatType> eval_cell(int cell_index);

    /**
     * @brief Evaluates the function over a specified edge.
     * 
     * @param[in] edge_index The index of the edge to evaluate.
     * @return Vector of function values for the edge.
     */
    Eigen::VectorX<FloatType> eval_edge(int edge_index);

    /**
     * @brief Extracts a subvector corresponding to a vertex.
     * 
     * @param[in] domain_id The domain ID for the vertex.
     * @return Matrix of extracted subvector values.
     */
    Eigen::MatrixX<FloatType> extract_vertex_subvec(int domain_id);

    /**
     * @brief Extracts a subvector corresponding to a degree of freedom (DOF).
     * 
     * @param[in] domain_id The domain ID for the DOF.
     * @return Matrix of extracted subvector values.
     */
    Eigen::MatrixX<FloatType> extract_dof_subvec(int domain_id);

    /**
     * @brief Assembles the mass matrix for the FEM function, used for L2 norm calculation
     */
    void assemble_mass_matrix();

    /**
     * @brief Set preassembled mass matrix for the FEM function, used for L2 norm calculation
     * @param[in] M The mass matrix in sparse format
     */
    void set_mass_matrix(Eigen::SparseMatrix<FloatType> &M);

    /**
     * @brief Computes the derivative of the function in the x-direction (interpolated).
     * 
     * @return The resulting FEM function after differentiation in the x-direction.
     */
    FEMFunction2D diff_x_interp();

    /**
     * @brief Computes the derivative of the function in the z-direction (interpolated).
     * 
     * @return The resulting FEM function after differentiation in the z-direction.
     */
    FEMFunction2D diff_z_interp();

    /**
     * @brief Computes the derivative of the function in the x-direction (projected).
     * 
     * @return The resulting FEM function after projection in the x-direction.
     */
    FEMFunction2D diff_x_proj();

    /**
     * @brief Computes the derivative of the function in the z-direction (projected).
     * 
     * @return The resulting FEM function after projection in the z-direction.
     */
    FEMFunction2D diff_z_proj();

    /**
     * @brief Returns the mass matrix of the FEM function.
     * 
     * @return The sparse mass matrix.
     */
    Eigen::SparseMatrix<FloatType> mass_matrix();

    /**
    * @brief Computes the L2 norm of the function.
    * 
    * This function calculates the L2 norm of the function represented by the 
    * `FEMFunction2D` object. It requires that the mass matrix has been assembled 
    * prior to calling this function. If the mass matrix has not been assembled, 
    * an exception will be thrown.
    * 
    * @throws std::runtime_error if the mass matrix has not been assembled.
    * 
    * @return The L2 norm of the function as a value of type `FloatType`.
    */
    FloatType L2_norm();

    /**
     * @brief Function call operator for evaluating the function at a given point.
     * 
     * @param[in] x The x-coordinate.
     * @param[in] z The z-coordinate.
     * @return The function value at the (x, z) point.
     */
    FloatType operator()(FloatType x, FloatType z);

    /**
     * @brief Adds another FEM function to the current FEM function and returns the result.
     * 
     * This operator overload performs element-wise addition between two FEM functions. It creates
     * a new FEM function where each value of the current function is added to the corresponding value
     * of the provided FEM function `f`.
     * 
     * @param[in] f The FEM function to add to the current FEM function.
     * 
     * @return A new FEM function representing the result of the element-wise addition of the current
     *         FEM function and the input FEM function `f`.
     */
    FEMFunction2D operator+(FloatType val);

    /**
     * @brief Subtracts a constant value from the FEM function and returns the result.
     * 
     * This operator overload performs element-wise subtraction, where the constant value `val` is
     * subtracted from each value of the FEM function, resulting in a new FEM function.
     * 
     * @param[in] val The constant value to subtract from the current FEM function.
     * 
     * @return A new FEM function representing the result of subtracting `val` from the current function.
     */
    FEMFunction2D operator-(FloatType val);

    /**
     * @brief Multiplies the FEM function by a constant value and returns the result.
     * 
     * This operator overload performs element-wise multiplication, where each value of the FEM function
     * is multiplied by the constant `val`, resulting in a new FEM function.
     * 
     * @param[in] val The constant value to multiply by.
     * 
     * @return A new FEM function representing the result of multiplying the current function by `val`.
     */
    FEMFunction2D operator*(FloatType val);

    /**
     * @brief Adds another FEM function to the current FEM function and returns the result.
     * 
     * This operator overload performs element-wise addition between two FEM functions. It creates
     * a new FEM function where each value of the current function is added to the corresponding value
     * of the provided FEM function `f`.
     * 
     * @param[in] f The FEM function to add to the current FEM function.
     * 
     * @return A new FEM function representing the result of the element-wise addition of the current
     *         FEM function and the input FEM function `f`.
     */
    FEMFunction2D operator+(const FEMFunction2D &f);

    /**
     * @brief Subtracts another FEM function from the current FEM function and returns the result.
     * 
     * This operator overload performs element-wise subtraction between two FEM functions. It creates
     * a new FEM function where each value of the current function is subtracted by the corresponding value
     * of the provided FEM function `f`.
     * 
     * @param[in] f The FEM function to subtract from the current FEM function.
     * 
     * @return A new FEM function representing the result of the element-wise subtraction of the input
     *         FEM function `f` from the current function.
     */
    FEMFunction2D operator-(const FEMFunction2D &f);

    /**
     * @brief Multiplies the current FEM function by another FEM function element-wise and returns the result.
     * 
     * This operator overload performs element-wise multiplication between two FEM functions. It creates
     * a new FEM function where each value of the current function is multiplied by the corresponding value
     * of the provided FEM function `f`.
     * 
     * @param[in] f The FEM function to multiply with the current FEM function.
     * 
     * @return A new FEM function representing the result of the element-wise multiplication of the current
     *         FEM function and the input FEM function `f`.
     */
    FEMFunction2D operator*(const FEMFunction2D &f);
};
