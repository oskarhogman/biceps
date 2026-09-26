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
#include <interval_mesh.hpp>
#include <enums.hpp>

IntervalMesh::IntervalMesh(FloatType x0, FloatType x1, int n_cells, int degree) :
    _degree(degree), _nof_cells(n_cells)
{
    if (n_cells < 1)
        throw std::invalid_argument("n_cells must be greater than 0");
    if (degree < 1)
        throw std::invalid_argument("degree must be greater than 0");
    else if (degree > 2)
        throw std::invalid_argument("degree greater than 2 is currently not supported");

    // Compute the number of degrees of freedom in the mesh
    _hl = _degree * n_cells;

    // Initialize matrices for point locations, cell-to-DOF mapping, and degree-of-freedom IDs
    pmat = Eigen::MatrixX<FloatType>::Zero(nof_dofs(), 2);
    cmat = Eigen::MatrixXi::Zero(nof_cells(), dofs_per_cell());
    dimat = Eigen::VectorXi::Zero(nof_dofs());

    // Set up point matrix (pmat) and degree-of-freedom identification (dimat)
    for (int di = 0; di < nof_dofs(); di++) {
        // Compute the position of each point in the mesh interval
        pmat(di, 0) = x0 + (x1 - x0) * di / (nof_dofs() - 1);

        // Mark nodes based on the point positions
        if (ABS_FUNC(pmat(di, 0) - x0) < 1e-10)
            dimat(di) = MESH1D::WEST_ID;  // Left boundary
        else if (ABS_FUNC(pmat(di, 0) - x1) < 1e-10)
            dimat(di) = MESH1D::EAST_ID;  // Right boundary
        else
            dimat(di) = MESH1D::INTERIOR_ID;  // Interior point
    }

    // Initialize the cell-to-degree-of-freedom (cmat) mapping
    for (int ci = 0; ci < nof_cells(); ci++)
        for (int k = 0; k < dofs_per_cell(); k++)
            cmat(ci, k) = ci * degree + k;

    // Compute and store the indices of vertex degrees of freedom
    compute_vertex_dof_inds();
}

void IntervalMesh::compute_vertex_dof_inds() {
    vertex_dof_inds.reserve(nof_verts());  // Pre-allocate space for vertex DOF indices
    for (int vi = 0; vi < _hl + 1; vi += _degree)  // Every `_degree`-th point is a vertex
        vertex_dof_inds.push_back(vi);  // Store the vertex DOF index
}

IntervalMesh::IntervalMesh(const Eigen::MatrixX<FloatType> &pmat, int degree)
    : _degree(degree), pmat(pmat)
{
    _nof_cells = (pmat.rows() - 1) / degree;

    // Initialize matrices for cell-to-DOF mapping and degree-of-freedom identification
    cmat = Eigen::MatrixXi::Zero(nof_cells(), dofs_per_cell());
    dimat = Eigen::VectorXi::Zero(nof_dofs());

    // Extract the first and last point coordinates from the point matrix
    FloatType x0 = pmat(0, 0);
    FloatType x1 = pmat(Eigen::last, 0);

    // Mark nodes based on the point positions
    for (int di = 0; di < nof_dofs(); di++) {
        if (ABS_FUNC(pmat(di, 0) - x0) < 1e-10)
            dimat(di) = MESH1D::WEST_ID;  // Left boundary
        else if (ABS_FUNC(pmat(di, 0) - x1) < 1e-10)
            dimat(di) = MESH1D::EAST_ID;  // Right boundary
        else
            dimat(di) = MESH1D::INTERIOR_ID;  // Interior point
    }

    // Initialize the cell-to-degree-of-freedom (cmat) mapping
    for (int ci = 0; ci < nof_cells(); ci++)
        for (int k = 0; k < dofs_per_cell(); k++)
            cmat(ci, k) = ci * degree + k;
}

std::vector<int> IntervalMesh::extract_vertex_dof_inds(int id)
{
    std::vector<int> vinds;  // Vector to store filtered vertex DOF indices
    for (int vi : vertex_dof_inds)
        if (dimat(vi) & id)  // Check if the DOF corresponds to the specified identifier
            vinds.push_back(vi);  // Add it to the result
    return vinds;
}

std::vector<int> IntervalMesh::extract_dof_inds(int id)
{
    std::vector<int> dinds;  // Vector to store filtered DOF indices
    for (int di = 0; di < dimat.rows(); di++)
        if (dimat(di) & id)  // Check if the DOF corresponds to the specified identifier
            dinds.push_back(di);
    return dinds;
}

int IntervalMesh::nof_cells()
{
    return _nof_cells;
}

int IntervalMesh::nof_verts()
{
    return _nof_cells + 1;
}

int IntervalMesh::nof_dofs()
{
    return degree() * nof_cells() + 1;
}

int IntervalMesh::dofs_per_cell()
{
    return _degree + 1;
}

int IntervalMesh::degree()
{
    return _degree;
}
