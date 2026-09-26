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

#include <vector>
#include <Eigen/Dense>
#include <float_type.hpp>

/**
 * @class IntervalMesh
 * @brief Represents a one-dimensional interval mesh with finite element degrees of freedom.
 *
 * This class provides methods for constructing and manipulating a 1D mesh, with operations
 * related to degrees of freedom (DOFs), cells, vertices, and element properties such as degree.
 */
class IntervalMesh {
private:
    /** 
     * @brief Polynomial degree of the finite element basis functions. 
     * 
     * Determines the order of interpolation used within each element.
     */
    int _degree;

    /** 
     * @brief Number of cells in the mesh. 
     * 
     * Represents the total count of elements (triangles or quadrilaterals) in the mesh
     */
    int _nof_cells;

    /** 
     * @brief Number of vertices in the mesh. 
     * 
     * Represents the total count of vertex points.
     */
    int _nof_verts;

    /** 
     * @brief Total number of degrees of freedom (DOFs) in the mesh. 
     * 
     * This includes vertex-based, edge-based, and cell-based DOFs, depending on the polynomial degree.
     */
    int _nof_dofs;

    /** 
     * @brief Number of horizontal layers.
     * 
     * Represents the total count of layers in the mesh
     */
    int _hl;

    /** 
     * @brief Indices of vertex degrees of freedom (DOFs). 
     * 
     * Stores the global indices of all vertex-based DOFs in the finite element discretization.
     */
    std::vector<int> vertex_dof_inds;

    /**
     * @brief Computes and stores the indices of vertex degrees of freedom.
     *
     * This method identifies which degrees of freedom are located at the vertices
     * and stores their indices for later use.
     */
    void compute_vertex_dof_inds();

public:
    /** 
     * @brief Matrix storing point coordinates for the mesh. 
     * 
     * Each row corresponds to a point in the mesh, with columns representing spatial coordinates 
     * (e.g., x and z in 2D). This matrix defines the physical layout of the mesh.
     */
    Eigen::MatrixX<FloatType> pmat;

    /** 
     * @brief Cell-to-DOF mapping matrix. 
     * 
     * Each row represents a cell, and the columns store the global indices of the degrees of freedom (DOFs) 
     * associated with that cell. This matrix is essential for assembling system matrices in finite element methods.
     */
    Eigen::MatrixXi cmat;

    /** 
     * @brief Vector storing DOF types (boundary or interior). 
     * 
     * Each entry corresponds to a DOF and contains an identifier that indicates whether 
     * it is a boundary DOF or an interior DOF.
     */
    Eigen::VectorXi dimat;

    /**
     * @brief Constructs an IntervalMesh using a specified interval and mesh parameters.
     *
     * @param[in] x0 Starting point of the interval.
     * @param[in] x1 Ending point of the interval.
     * @param[in] n_cells Number of cells within the interval.
     * @param[in] degree Polynomial degree of the finite element basis functions.
     */
    IntervalMesh(FloatType x0, FloatType x1, int n_cells, int degree);

    /**
     * @brief Constructs an IntervalMesh from an existing point matrix and degree.
     *
     * @param[in] pmat A matrix representing point coordinates.
     * @param[in] degree Polynomial degree of the finite element basis functions.
     */
    IntervalMesh(const Eigen::MatrixX<FloatType> &pmat, int degree);

    /**
     * @brief Extracts the indices of degrees of freedom based on a specified identifier.
     *
     * @param[in] id Identifier used to filter the DOF indices.
     * @return A vector containing the filtered degree of freedom indices.
     */
    std::vector<int> extract_dof_inds(int id);

    /**
     * @brief Extracts the indices of vertex degrees of freedom based on a specified identifier.
     *
     * @param[in] id Identifier used to filter vertex DOF indices.
     * @return A vector containing the filtered vertex DOF indices.
     */
    std::vector<int> extract_vertex_dof_inds(int id);

    /**
     * @brief Returns the number of cells in the mesh.
     *
     * @return The number of cells.
     */
    int nof_cells();

    /**
     * @brief Returns the number of vertices in the mesh.
     *
     * @return The number of vertices.
     */
    int nof_verts();

    /**
     * @brief Returns the total number of degrees of freedom in the mesh.
     *
     * @return The total number of degrees of freedom.
     */
    int nof_dofs();

    /**
     * @brief Returns the number of degrees of freedom per cell.
     *
     * @return The number of degrees of freedom per cell.
     */
    int dofs_per_cell();

    /**
     * @brief Returns the polynomial degree of the finite element basis functions.
     *
     * @return The polynomial degree.
     */
    int degree();
};

