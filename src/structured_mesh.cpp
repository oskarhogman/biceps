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
#include <structured_mesh.hpp>
#include <functional>
#include <tuple>
#include <set>
#include <boost/format.hpp>
#include <fem_2d.hpp>
#include <enums.hpp>

StructuredMesh::StructuredMesh(int nx, int nz, int degree, int cell_type): 
    _nx(nx), _nz(nz), _degree(degree), _cell_type(cell_type)
{
    if (nx < 1)
        throw std::invalid_argument("nx must be greater than 0");
    if (nz < 1)
        throw std::invalid_argument("nz must be greater than 0");

    if (degree < 1)
        throw std::invalid_argument("degree must be greater than 0");
    else if (degree > 2)
        throw std::invalid_argument("degree greater than 2 is currently not supported");

    if (cell_type == MESH2D::TRIANGLE_LEFT || cell_type == MESH2D::TRIANGLE_RIGHT) {
        _dofs_per_cell = (_degree+1)*(_degree+2) >> 1;
        _nof_cells = 2*_nx*_nz;
        _nof_edges = 3*_nx*_nz + _nx + _nz;
        _dofs_per_edge = degree+1;
        _edges_per_cell = 3;
    } else if (cell_type == MESH2D::QUADRILATERAL) {
        _dofs_per_cell = (_degree+1)*(_degree+1);
        _nof_cells = _nx*_nz;
        _dofs_per_edge = degree+1;
        _nof_edges = 2*_nx*_nz + _nx + _nz;
        _edges_per_cell = 4;
    } else {
        throw std::invalid_argument(
            (boost::format{"Unrecognized cell_type with id '%d'"} %cell_type).str()
        );
    }

    _hl = _degree*_nx;
    _vl = _degree*_nz;
    _nof_verts = (_nx+1)*(_nz+1);
    _nof_dofs = (_hl+1)*(_vl+1);

    // Initialize matrices and vectors
    pmat = Eigen::MatrixX<FloatType>(_nof_dofs, 2);
    pmat_unit_box = Eigen::MatrixX<FloatType>(_nof_dofs, 2);
    cmat = Eigen::MatrixXi(_nof_cells, _dofs_per_cell);
    dimat = Eigen::VectorXi(_nof_dofs);
    cimat = Eigen::VectorXi(_nof_cells);

    // Assemble various matrices and compute necessary data structures
    assemble_pmat();
    assemble_cmat();
    assemble_emat();
    assemble_cimat();
    assemble_dimat();
    assemble_eimat();
    compute_vertex_dof_inds();
    compute_normals_and_tangents();

    // Initialize mappings
    v2s_map = Eigen::VectorXi::Zero(_nof_dofs);
    v2b_map = Eigen::VectorXi::Zero(_nof_dofs);
    for (int vi = 0; vi < _nof_dofs; vi++) {
        v2b_map[vi] = vi % (_hl+1);
        v2s_map[vi] = v2b_map[vi] + _vl*(_hl+1);
    }
}

void StructuredMesh::assemble_pmat()
{
    // Generate linearly spaced values for the x and z ranges
    Eigen::VectorX<FloatType> xrange = Eigen::VectorX<FloatType>::LinSpaced(_hl + 1, 0.0, 1.0);
    Eigen::VectorX<FloatType> zrange = Eigen::VectorX<FloatType>::LinSpaced(_vl + 1, 0.0, 1.0);

    int dof = 0;
    // Iterate over each combination of z and x coordinates
    for (FloatType z : zrange) {
        for (FloatType x : xrange) {
            // Assign the calculated coordinates to the global position matrix
            pmat(dof, 0) = x;   // x-coordinate
            pmat(dof, 1) = z;   // z-coordinate
            pmat_unit_box(dof, 0) = x;  // x-coordinate in unit box
            pmat_unit_box(dof, 1) = z;  // z-coordinate in unit box
            dof++;
        }
    }
}

void StructuredMesh::assemble_cmat()
{
    int ci = 0;  // Initialize cell index
    for (int i = 0; i < _vl; i += _degree) {
        for (int j = 0; j < _hl; j += _degree) {
            if (_cell_type == MESH2D::TRIANGLE_RIGHT) {
                triangle_right_connectivity(ci, i, j);
                ci += 2;
            } else if (_cell_type == MESH2D::TRIANGLE_LEFT) {
                triangle_left_connectivity(ci, i, j);
                ci += 2;
            } else if (_cell_type == MESH2D::QUADRILATERAL) {
                quadrilateral_connectivity(ci, i, j);
                ci += 1;
            }
        }
    }
}

void StructuredMesh::assemble_emat()
{
    // Initialize edge matrix with size based on number of edges and DOFs per edge
    emat = Eigen::MatrixXi(_nof_edges, _dofs_per_edge);

    // Set to track visited edges to ensure each edge is added once
    std::set<std::tuple<int, int>> visited_edges;
    std::tuple<int, int> edge;  // Edge represented as a tuple of (start, end)

    int first_ind, last_ind, ei = 0;  // Edge indices and edge counter
    for (int ci = 0; ci < cmat.rows(); ++ci) {  // Loop over all cells
        Eigen::VectorXi cell = cmat.row(ci);  // Get the current cell's DOFs
        for (int i = 0; i < _edges_per_cell; ++i) {  // Loop over edges per cell
            first_ind = i * _degree;
            if (i != _edges_per_cell - 1)  // If not the last edge in the cell
                last_ind = (i + 1) * _degree;
            else
                last_ind = 0;  // For the last edge, wrap around to the first DOF

            // Create an edge by ordering the DOFs in increasing order
            if (cell(first_ind) < cell(last_ind))
                edge = std::tuple<int, int>(cell(first_ind), cell(last_ind));
            else
                edge = std::tuple<int, int>(cell(last_ind), cell(first_ind));

            // If the edge hasn't been visited before, add it to emat
            if (visited_edges.find(edge) == visited_edges.end()) {
                visited_edges.insert(edge);
                for (int j = 0; j < _dofs_per_edge - 1; ++j) {
                    emat(ei, j) = cell(first_ind + j);  // Assign DOFs for the edge
                }
                emat(ei, _dofs_per_edge - 1) = cell(last_ind);  // Assign the last DOF
                ei++;
            }
        }
    }
}

inline void StructuredMesh::triangle_right_connectivity(int ci, int i, int j)
{
    if (_degree == 1) {
        // Linear case: Right triangle connectivity (degree 1)
        cmat(ci, 0) = (_hl + 1) * i + j;
        cmat(ci, 1) = (_hl + 1) * i + j + 1;
        cmat(ci, 2) = (_hl + 1) * (i + 1) + j;

        cmat(ci + 1, 0) = (_hl + 1) * (i + 1) + j + 1;
        cmat(ci + 1, 1) = (_hl + 1) * (i + 1) + j;
        cmat(ci + 1, 2) = (_hl + 1) * i + (j + 1);
    } else if (_degree == 2) {
        // Quadratic case: Right triangle connectivity (degree 2)
        cmat(ci, 0) = (_hl + 1) * i + j;
        cmat(ci, 1) = (_hl + 1) * i + j + 1;
        cmat(ci, 2) = (_hl + 1) * i + j + 2;
        cmat(ci, 3) = (_hl + 1) * (i + 1) + j + 1;
        cmat(ci, 4) = (_hl + 1) * (i + 2) + j;
        cmat(ci, 5) = (_hl + 1) * (i + 1) + j;

        cmat(ci + 1, 0) = (_hl + 1) * (i + 2) + j + 2;
        cmat(ci + 1, 1) = (_hl + 1) * (i + 2) + j + 1;
        cmat(ci + 1, 2) = (_hl + 1) * (i + 2) + j;
        cmat(ci + 1, 3) = (_hl + 1) * (i + 1) + j + 1;
        cmat(ci + 1, 4) = (_hl + 1) * i + j + 2;
        cmat(ci + 1, 5) = (_hl + 1) * (i + 1) + j + 2;
    }
}

inline void StructuredMesh::triangle_left_connectivity(int ci, int i, int j)
{
    if (_degree == 1) {
        // Linear case: Left triangle connectivity (degree 1)
        cmat(ci, 0) = (_hl + 1) * (i + 1) + j;
        cmat(ci, 1) = (_hl + 1) * i + j;
        cmat(ci, 2) = (_hl + 1) * (i + 1) + j + 1;

        cmat(ci + 1, 0) = (_hl + 1) * i + j + 1;
        cmat(ci + 1, 1) = (_hl + 1) * (i + 1) + j + 1;
        cmat(ci + 1, 2) = (_hl + 1) * i + j;
    } else if (_degree == 2) {
        // Quadratic case: Left triangle connectivity (degree 2)
        cmat(ci, 0) = (_hl + 1) * (i + 2) + j;
        cmat(ci, 1) = (_hl + 1) * (i + 1) + j;
        cmat(ci, 2) = (_hl + 1) * i + j;
        cmat(ci, 3) = (_hl + 1) * (i + 1) + j + 1;
        cmat(ci, 4) = (_hl + 1) * (i + 2) + j + 2;
        cmat(ci, 5) = (_hl + 1) * (i + 2) + j + 1;

        cmat(ci + 1, 0) = (_hl + 1) * i + j + 2;
        cmat(ci + 1, 1) = (_hl + 1) * (i + 1) + j + 2;
        cmat(ci + 1, 2) = (_hl + 1) * (i + 2) + j + 2;
        cmat(ci + 1, 3) = (_hl + 1) * (i + 1) + j + 1;
        cmat(ci + 1, 4) = (_hl + 1) * i + j;
        cmat(ci + 1, 5) = (_hl + 1) * i + j + 1;
    }
}

inline void StructuredMesh::quadrilateral_connectivity(int ci, int i, int j)
{
    if (_degree == 1) {
        // Linear case: Quadrilateral connectivity (degree 1)
        cmat(ci, 0) = (_hl + 1) * i + j;
        cmat(ci, 1) = (_hl + 1) * i + j + 1;
        cmat(ci, 2) = (_hl + 1) * (i + 1) + j + 1;
        cmat(ci, 3) = (_hl + 1) * (i + 1) + j;
    } else if (_degree == 2) {
        // Quadratic case: Quadrilateral connectivity (degree 2)
        cmat(ci, 0) = (_hl + 1) * i + j;
        cmat(ci, 1) = (_hl + 1) * i + j + 1;
        cmat(ci, 2) = (_hl + 1) * i + j + 2;
        cmat(ci, 3) = (_hl + 1) * (i + 1) + j + 2;
        cmat(ci, 4) = (_hl + 1) * (i + 2) + j + 2;
        cmat(ci, 5) = (_hl + 1) * (i + 2) + j + 1;
        cmat(ci, 6) = (_hl + 1) * (i + 2) + j;
        cmat(ci, 7) = (_hl + 1) * (i + 1) + j;
        cmat(ci, 8) = (_hl + 1) * (i + 1) + j + 1;
    }
}

void StructuredMesh::assemble_cimat() 
{
    if (_cell_type == MESH2D::QUADRILATERAL)
        assemble_cimat_quadrilateral();
    else if (_cell_type == MESH2D::TRIANGLE_LEFT)
        assemble_cimat_triangle_left();
    else if (_cell_type == MESH2D::TRIANGLE_RIGHT)
        assemble_cimat_triangle_right();
}

void StructuredMesh::assemble_cimat_triangle_left() 
{
    // Placeholder (not implemented yet)
}

void StructuredMesh::assemble_cimat_triangle_right() 
{
    // Placeholder (not implemented yet())
}

void StructuredMesh::assemble_cimat_quadrilateral()
{
    int i, j, ci;

    // Mark corner cells with predefined IDs
    cimat(0) = MESH2D::SOUTH_WEST_ID;
    cimat(_nx - 1) = MESH2D::SOUTH_EAST_ID;
    cimat(_nof_cells - _nx) = MESH2D::NORTH_WEST_ID;
    cimat(_nof_cells - 1) = MESH2D::NORTH_EAST_ID;

    // Mark North boundary cells (excluding corners)
    i = _nz - 1;  // Last row
    for (j = 1; j < _nx - 1; ++j) {
        ci = i * _nx + j;
        cimat(ci) = MESH2D::NORTH_ID;
    }

    // Mark West boundary cells (excluding corners)
    j = 0;  // First column
    for (i = 1; i < _nz - 1; ++i) { 
        ci = i * _nx + j;
        cimat(ci) = MESH2D::WEST_ID;
    }

    // Mark East boundary cells (excluding corners)
    j = _nx - 1;  // Last column
    for (i = 1; i < _nz - 1; ++i) {
        ci = i * _nx + j;
        cimat(ci) = MESH2D::EAST_ID;
    }

    // Mark South boundary cells (excluding corners)
    i = 0;  // First row
    for (j = 1; j < _nx - 1; ++j) {
        ci = i * (_nx - 1) + j;
        cimat(ci) = MESH2D::SOUTH_ID;
    }

    // Mark interior cells (non-boundary nodes)
    for (i = 1; i < _nz - 1; ++i) {
        for (j = 1; j < _nx - 1; ++j) {
            ci = i * _nx + j;
            cimat(ci) = MESH2D::INTERIOR_ID;
        }
    }
}

void StructuredMesh::assemble_dimat()
{
    // Compute the width of the domain in the x-direction
    FloatType domain_width = pmat(Eigen::last, 0) - pmat(0, 0);

    // Tolerances for boundary identification based on mesh resolution
    FloatType xtol = domain_width * 1e-3 / _hl;
    FloatType ztol = 1e-3 / _vl;

    // Iterate over all degrees of freedom (DOFs)
    for (int i = 0; i < _nof_dofs; ++i) {

        // Check if the current DOF lies in the North-West corner of the domain
        if (pmat(i, 1) > 1.0 - ztol && pmat(i, 0) < xtol)
            dimat(i) = MESH2D::NORTH_WEST_ID;

        // Check if the current DOF lies in the South-West corner of the domain
        else if (pmat(i, 1) < ztol && pmat(i, 0) < xtol)
            dimat(i) = MESH2D::SOUTH_WEST_ID;

        // Check if the current DOF lies in the South-East corner of the domain
        else if (pmat(i, 1) < ztol && pmat(i, 0) > 1.0 - xtol)
            dimat(i) = MESH2D::SOUTH_EAST_ID;

        // Check if the current DOF lies in the North-East corner of the domain
        else if (pmat(i, 1) > 1.0 - ztol && pmat(i, 0) > 1.0 - xtol)
            dimat(i) = MESH2D::NORTH_EAST_ID;

        // Check if the current DOF lies on the North edge (excluding corners)
        else if (pmat(i, 1) > 1.0 - ztol)
            dimat(i) = MESH2D::NORTH_ID;

        // Check if the current DOF lies on the South edge (excluding corners)
        else if (pmat(i, 1) < ztol)
            dimat(i) = MESH2D::SOUTH_ID;

        // Check if the current DOF lies on the West edge (excluding corners)
        else if (pmat(i, 0) < xtol)
            dimat(i) = MESH2D::WEST_ID;

        // Check if the current DOF lies on the East edge (excluding corners)
        else if (pmat(i, 0) > 1.0 - xtol)
            dimat(i) = MESH2D::EAST_ID;

        // Otherwise, the current DOF lies in the interior of the domain
        else
            dimat(i) = MESH2D::INTERIOR_ID;
    }
}

void StructuredMesh::assemble_eimat()
{
    int dpe = _dofs_per_edge;

    // Initialize the edge matrix with zeros, where each row corresponds to an edge and each column
    // stores boundary information and node indices for the edge
    eimat = Eigen::MatrixXi::Zero(_nof_edges, dpe + 1);

    // Compute domain width in the x-direction
    FloatType domain_width = pmat(Eigen::last, 0) - pmat(0, 0);

    // Define tolerances for boundary identification
    FloatType xtol = domain_width * 1e-3 / _hl; 
    FloatType ztol = 1e-3 / _vl;

    // Classify edges based on the positions of the nodes
    for (int ei = 0; ei < emat.rows(); ++ei) {
        // Get the horizontal and vertical positions of the two nodes defining the edge
        FloatType x1 = pmat(emat(ei, 0), 0);
        FloatType z1 = pmat(emat(ei, 0), 1);
        FloatType x2 = pmat(emat(ei, 1), 0);
        FloatType z2 = pmat(emat(ei, 1), 1);

        // Check the boundary conditions for the edge and assign the appropriate ID
        if (z1 > 1.0 - ztol && z2 > 1.0 - ztol)
            eimat(ei, 0) = MESH2D::NORTH_ID;  // Edge is on the North boundary
        else if (z1 < ztol && z2 < ztol)
            eimat(ei, 0) = MESH2D::SOUTH_ID;  // Edge is on the South boundary
        else if (x1 < xtol && x2 < xtol)
            eimat(ei, 0) = MESH2D::WEST_ID;   // Edge is on the West boundary
        else if (x1 > 1.0 - xtol && x2 > 1.0 - xtol)
            eimat(ei, 0) = MESH2D::EAST_ID;   // Edge is on the East boundary
        else
            eimat(ei, 0) = MESH2D::INTERIOR_ID;  // Edge is in the interior
    }

    // Set of visited edges to avoid duplicating edges in the matrix
    std::set<std::tuple<int, int>> visited_edges;
    std::tuple<int, int> edge;

    int v1, v2;
    int ei = 0;  // Edge index
    // Iterate over all cells in the connectivity matrix (cmat) to assign DOFs to edges
    for (int ci = 0; ci < cmat.rows(); ++ci) {
        Eigen::VectorXi cell = cmat.row(ci);  // Get the nodes of the current cell

        // Iterate over all edges in the cell
        for (int i = 0; i < _edges_per_cell; ++i) {
            // Get the two nodes defining the edge
            v1 = cell(i * _degree);
            if (i != _edges_per_cell - 1)
                v2 = cell((i + 1) * _degree);
            else
                v2 = cell(0);  // Last edge wraps around to the first node

            // Sort the edge nodes to maintain consistent ordering (smaller node index first)
            if (v1 < v2)
                edge = std::tuple<int, int>(v1, v2);
            else
                edge = std::tuple<int, int>(v2, v1);

            // Process edge only if it has not been visited yet
            if (visited_edges.find(edge) == visited_edges.end()) {
                eimat(ei, 1) = v1;  // Assign the first node to the edge
                eimat(ei, dpe) = v2;  // Assign the second node to the edge
                if (_degree == 2)
                    eimat(ei, 2) = cell(i * _degree + 1);  // For degree 2 elements, assign the middle node
                visited_edges.insert(edge);  // Mark the edge as visited
                ei++;  // Move to the next edge
            }
        }
    }
}

const int inline StructuredMesh::corner_dof_SW()
{
    return 0;
}

const int inline StructuredMesh::corner_dof_SE()
{
    return _hl;
}

const int inline StructuredMesh::corner_dof_NE()
{
    return (_hl + 1)*(_vl + 1) - 1;
}

const int inline StructuredMesh::corner_dof_NW()
{
    return (_hl + 1)*_vl;
}

bool inline StructuredMesh::is_corner_dof(int vi)
{
    return (
        vi == corner_dof_SW() || vi == corner_dof_SE() ||
        vi == corner_dof_NE() || vi == corner_dof_NW()
    );
}

void StructuredMesh::compute_edge_normals_and_tangents()
{
    // Initialize the edge_tangents and edge_normals matrices with zeros
    edge_tangents = Eigen::MatrixX<FloatType>::Zero(_nof_edges, 2);
    edge_normals = Eigen::MatrixX<FloatType>::Zero(_nof_edges, 2);

    // Extract the indices of the boundary edges
    std::vector<int> boundary_einds = extract_edge_inds(MESH2D::BOUNDARY_ID);

    // Loop over each boundary edge and compute its tangent and normal
    for (int ei : boundary_einds) {
        // Get the vertex indices defining the boundary edge
        int v1 = eimat(ei, 1);
        int v2 = eimat(ei, _dofs_per_edge);

        // Compute the tangent vector (vector from v1 to v2) and normalize it
        edge_tangents(ei, Eigen::all) = pmat.row(v2) - pmat.row(v1);
        edge_tangents(ei, Eigen::all) /= edge_tangents.row(ei).norm();

        // Compute the normal as a perpendicular vector to the tangent
        edge_normals(ei, 0) = edge_tangents(ei, 1);
        edge_normals(ei, 1) = -edge_tangents(ei, 0);
    }
}

void StructuredMesh::compute_dof_normals_and_tangents()
{
    // Initialize the dof_tangents and dof_normals matrices with zeros
    dof_tangents = Eigen::MatrixX<FloatType>::Zero(_nof_dofs, 2);
    dof_normals = Eigen::MatrixX<FloatType>::Zero(_nof_dofs, 2);

    // Compute normals/tangents for vertex dofs
    std::vector<int> boundary_einds = extract_edge_inds(MESH2D::BOUNDARY_ID);
    for (int i = 0; i < boundary_einds.size(); ++i) {
        int ei = boundary_einds[i];
        int v1 = eimat(ei, 1);
        int v2 = eimat(ei, _dofs_per_edge);
        // Exclude corners
        dof_tangents(v1, Eigen::all) += edge_tangents.row(ei);
        dof_normals(v1, Eigen::all) += edge_normals.row(ei);
        dof_tangents(v2, Eigen::all) += edge_tangents.row(ei);
        dof_normals(v2, Eigen::all) += edge_normals.row(ei);
    }

    // Normalize vertex dofs normals/tangents
    std::vector<int> boundary_vertex_dofs = extract_vertex_dof_inds(MESH2D::BOUNDARY_ID);
    for (int di: boundary_vertex_dofs) {
        dof_tangents(di, Eigen::all) /= dof_tangents.row(di).norm();
        dof_normals(di, Eigen::all) /= dof_normals.row(di).norm();
    }

    // Compute normals/tangents for edge dofs
    if (_degree == 2)
        for (int ei: boundary_einds) {
            int ve = eimat(ei, 2);
            dof_tangents(ve, Eigen::all) = edge_tangents.row(ei);
            dof_normals(ve, Eigen::all) = edge_normals.row(ei);
        }

    // Set normal/tangent at SW corner to edge normal/tangent
    dof_tangents(
        corner_dof_SW(), Eigen::all
    ) = dof_tangents(corner_dof_SW()+1, Eigen::all);
    dof_normals(
        corner_dof_SW(), Eigen::all
    ) = dof_normals(corner_dof_SW()+1, Eigen::all);

    // Set normal/tangent at SE corner to edge normal/tangent
    dof_tangents(
        corner_dof_SE(), Eigen::all
    ) = dof_tangents(corner_dof_SE()-1, Eigen::all);
    dof_normals(
        corner_dof_SE(), Eigen::all
    ) = dof_normals(corner_dof_SE()-1, Eigen::all);

    // Set normal/tangent at NE corner to edge normal/tangent
    dof_tangents(
        corner_dof_NE(), Eigen::all
    ) = dof_tangents(corner_dof_NE()-1, Eigen::all);
    dof_normals(
        corner_dof_NE(), Eigen::all
    ) = dof_normals(corner_dof_NE()-1, Eigen::all);

    // Set normal/tangent at NW corner to edge normal/tangent
    dof_tangents(
        corner_dof_NW(), Eigen::all
    ) = dof_tangents(corner_dof_NW()+1, Eigen::all);
    dof_normals(
        corner_dof_NW(), Eigen::all
    ) = dof_normals(corner_dof_NW()+1, Eigen::all);
}

void StructuredMesh::compute_vertex_normals_and_tangents()
{
    // Compute the tangents for the vertices based on the DOF indices
    vertex_tangents = dof_tangents(vertex_dof_inds, Eigen::all);

    // Compute the normals for the vertices based on the DOF indices
    vertex_normals = dof_normals(vertex_dof_inds, Eigen::all);
}

void StructuredMesh::compute_normals_and_tangents()
{
    compute_edge_normals_and_tangents();
    compute_dof_normals_and_tangents();
    compute_vertex_normals_and_tangents();
}

std::vector<int> StructuredMesh::extract_edge_inds(int id)
{
    std::vector<int> einds;
    for (int ei = 0; ei < eimat.rows(); ++ei)
        if (eimat(ei, 0) & id)
            einds.push_back(ei);
    return einds;
}

std::vector<int> StructuredMesh::extract_cell_inds(int id)
{
    std::vector<int> cell_inds;
    for (int ci = 0; ci < cimat.rows(); ++ci)
        if (cimat(ci) & id)
            cell_inds.push_back(ci);
    return cell_inds;
}

std::vector<int> StructuredMesh::extract_vertex_dof_inds(int id)
{
    std::vector<int> dinds;
    for (int vi : vertex_dof_inds)
        if (dimat(vi) & id)
            dinds.push_back(vi);

    return dinds;
}

std::vector<int> StructuredMesh::extract_dof_inds(int id)
{
    std::vector<int> dinds;
    for (int di = 0; di < dimat.rows(); ++di)
        if (dimat(di) & id)
            dinds.push_back(di);
    return dinds;
}

void StructuredMesh::extrude_x(FloatType x0, FloatType x1)
{
    // Create a vector of ones with the same size as the number of DOFs
    Eigen::VectorX<FloatType> ones = Eigen::VectorX<FloatType>::Ones(pmat.rows());

    // Perform linear interpolation of x-coordinates between x0 and x1
    pmat(Eigen::all, 0) = (ones - pmat(Eigen::all, 0)) * x0 + pmat(Eigen::all, 0) * x1;

    // Recompute boundary normals and tangents to maintain mesh consistency
    compute_normals_and_tangents();
}

void StructuredMesh::extrude_x(const Eigen::VectorX<FloatType> &xvec_p1)
{
    // Initialize the x_vec to store the extruded values
    Eigen::VectorX<FloatType> x_vec;

    // For degree 1 elements, use the input vector directly
    if (_degree == 1)
        x_vec = xvec_p1;
    // For degree 2 elements, expand the vector and interpolate internal nodes
    else if (_degree == 2) {
        x_vec = Eigen::VectorX<FloatType>::Zero(_hl + 1);

        // Copy the input vector values at every second position in x_vec
        x_vec(Eigen::seq(0, Eigen::last, 2)) = xvec_p1;

        // Set x coordinate of internal nodes
        for (int i = 1; i < x_vec.size() - 1; i += 2)
            x_vec(i) = 0.5 * (x_vec(i - 1) + x_vec(i + 1));
    }

    // Fill in the mesh coordinates along the x-direction (pmat) using the extruded values
    int dof = 0;
    for (int i = 0; i < _vl + 1; ++i) {
        for (int j = 0; j < _hl + 1; ++j) {
            // Assign the extruded x-coordinate to the pmat matrix
            pmat(dof, 0) = x_vec(j);
            dof++;
        }
    }

    // Recompute boundary normals and tangents after extrusion
    compute_normals_and_tangents();
}

void StructuredMesh::extrude_z(
    std::function<FloatType (FloatType)> zb_expr,
    std::function<FloatType (FloatType)> zs_expr
)
{
    // Extract indices of surface vertices
    std::vector<int> surf_vert_inds = extract_vertex_dof_inds(MESH2D::SURFACE_ID);

    // Get the x-coordinates of surface vertices
    Eigen::VectorX<FloatType> xvec_p1 = pmat(surf_vert_inds, 0);

    // Compute bottom and top surface elevations using the provided functions
    Eigen::VectorX<FloatType> zb_vec_p1 = xvec_p1.unaryExpr(zb_expr);
    Eigen::VectorX<FloatType> zs_vec_p1 = xvec_p1.unaryExpr(zs_expr);

    // Perform the extrusion using the computed surface elevations
    extrude_z(zb_vec_p1, zs_vec_p1);
}

void StructuredMesh::extrude_z(const Eigen::VectorX<FloatType> &zs_vec_p1)
{
    // Extract the z-coordinates of the bedrock surface from the current mesh
    std::vector<int> bed_vert_inds = extract_vertex_dof_inds(MESH2D::BED_ID);
    Eigen::VectorX<FloatType> zb_vec_p1 = pmat(bed_vert_inds, 1);

    // Call the main extrude_z function with the existing bottom surface and new top surface
    extrude_z(zb_vec_p1, zs_vec_p1);
}

void StructuredMesh::extrude_z(
    const Eigen::VectorX<FloatType> &zb_vec_p1, const Eigen::VectorX<FloatType> &zs_vec_p1
)
{
    Eigen::VectorX<FloatType> zb_vec_p2 = Eigen::VectorX<FloatType>::Zero(_hl+1);
    Eigen::VectorX<FloatType> zs_vec_p2 = Eigen::VectorX<FloatType>::Zero(_hl+1);

    // Precalculate displacement of surface and bedrock
    int k = 0;
    for (int i = 0; i < _hl+1; i+=_degree) {
        zb_vec_p2[i] = zb_vec_p1[k];
        zs_vec_p2[i] = zs_vec_p1[k];
        k++;
    }
    // Set the correct z position of all cell corner nodes
    for (int i = 0; i < _vl+1; i+=_degree) {
        for (int j = 0; j < _hl+1; j+=_degree) {
            int vi = i*(_hl+1) + j;
            FloatType z = pmat_unit_box(vi, 1);
            pmat(vi, 1) = (1.0-z)*zb_vec_p2(v2b_map(vi)) + z*zs_vec_p2(v2b_map(vi));
        }
    }

    if (_degree == 2) {
        // set the correct z position of all cell interior nodes
        for (int i = 1; i < _vl; i+=_degree) {
            for (int j = 1; j < _hl; j+=_degree) {
                int vi_NW = (i+1)*(_hl+1) + j-1;
                int vi_NE = (i+1)*(_hl+1) + j+1;
                int vi_SW = (i-1)*(_hl+1) + j-1;
                int vi_SE = (i-1)*(_hl+1) + j+1;
                int vi = i*(_hl+1) + j;
                if (_cell_type == MESH2D::TRIANGLE_LEFT)
                    pmat(vi, 1) = 0.5*(pmat(vi_NE, 1) + pmat(vi_SW, 1));
                else if (_cell_type == MESH2D::TRIANGLE_RIGHT)
                    pmat(vi, 1) = 0.5*(pmat(vi_NW, 1) + pmat(vi_SE, 1));
                else if (_cell_type == MESH2D::QUADRILATERAL)
                    pmat(vi, 1) = 0.25*(pmat(vi_NW, 1) + pmat(vi_NE, 1) + pmat(vi_SW, 1) + pmat(vi_SE, 1));
            }
        }

        // set the correct z position of all horizontal edge nodes
        for (int i = 0; i < _vl+1; i+=_degree) {
            for (int j = 1; j < _hl; j+=_degree) {
                int vi_W = i*(_hl+1) + j-1;
                int vi_E = i*(_hl+1) + j+1;
                int vi = i*(_hl+1) + j;	
                pmat(vi, 1) = 0.5*(pmat(vi_W, 1) + pmat(vi_E, 1));
            }
        }

        // Set the correct z position of all vertical edge nodes
        for (int i = 1; i < _vl; i+=_degree) {
            for (int j = 0; j < _hl+1; j+=_degree) {
                int vi_N = (i+1)*(_hl+1) + j;
                int vi_S = (i-1)*(_hl+1) + j;
                int vi = i*(_hl+1) + j;
                pmat(vi, 1) = 0.5*(pmat(vi_N, 1) + pmat(vi_S, 1));
            }
        }
    }

    // Recompute boundary normals and tangents
    compute_normals_and_tangents();
}

void StructuredMesh::compute_vertex_dof_inds() {
    int n_verts = (_nx+1)*(_nz+1);
    vertex_dof_inds.reserve(n_verts);

    for (int i = 0; i < _vl+1; i+=_degree) {
        for (int j = 0; j < _hl+1; j+=_degree) {
            vertex_dof_inds.push_back(i*(_hl+1) + j);
        }
    }
}

int StructuredMesh::nx()
{
    return _nx;
}

int StructuredMesh::nz()
{
    return _nz;
}

int StructuredMesh::hl()
{
    return _hl;
}

int StructuredMesh::vl()
{
    return _vl;
}

int StructuredMesh::nof_cells()
{
    return _nof_cells;
}

int StructuredMesh::dofs_per_cell()
{
    return _dofs_per_cell;
}

int StructuredMesh::nof_edges()
{
    return _nof_edges;
}

int StructuredMesh::dofs_per_edge()
{
    return _dofs_per_edge;
}

int StructuredMesh::cell_type()
{
    return _cell_type;
}

int StructuredMesh::nof_verts()
{
    return _nof_verts;
}

int StructuredMesh::nof_dofs()
{
    return _nof_dofs;
}

int StructuredMesh::degree()
{
    return _degree;
}

FloatType StructuredMesh::area()
{
    // Declare matrices and vectors for storing node coordinates, quadrature points,
    // Lagrange basis functions, derivatives, and other intermediate results.
    Eigen::MatrixX<FloatType> node_coords, qpoints_rs, qpoints_xz, phi_rs, dphi_rs, dphi_xz;
    Eigen::VectorX<FloatType> qweights, detJ_rs;
    Eigen::VectorXi element;

    // Perform Gauss-Legendre quadrature to obtain the quadrature points and weights
    // in reference coordinates for integration.
    FEM2D::gauss_legendre_quadrature(
        1, cell_type(), qpoints_rs, qweights
    );

    // Compute Lagrange basis functions and their derivatives at the quadrature points
    FEM2D::lagrange_basis(
        degree(), cell_type(), qpoints_rs, phi_rs, dphi_rs
    );

    // Initialize the Jacobian determinant vector and quadrature points in physical coordinates.
    detJ_rs = Eigen::VectorX<FloatType>::Zero(qpoints_rs.rows());
    qpoints_xz = Eigen::MatrixX<FloatType>::Zero(qpoints_rs.rows(), 2);
    dphi_xz = Eigen::MatrixX<FloatType>::Zero(2 * qpoints_rs.rows(), dofs_per_cell());

    // Initialize the total area to zero.
    FloatType _area = 0.0;

    // Iterate over all cells in the mesh (cmat contains the connectivity matrix of the mesh).
    for (int k = 0; k < cmat.rows(); ++k) {
        element = cmat.row(k); // Get the node indices for the current element.
        node_coords = pmat(element, Eigen::all); // Get the coordinates of the nodes.

        // Map the quadrature points from the reference cell to the physical cell using
        // the Lagrange basis functions and compute the Jacobian determinant.
        FEM2D::map_to_reference_cell(
            degree(), cell_type(), node_coords, qpoints_rs,
            phi_rs, dphi_rs, detJ_rs, qpoints_xz, dphi_xz
        );

        // For each quadrature point, accumulate the contribution to the area.
        for (int q = 0; q < qpoints_rs.rows(); ++q) {
            _area += ABS_FUNC(detJ_rs(q)) * qweights(q); // Add the area contribution from this quadrature point.
        }
    }

    // Return the computed total area of the mesh.
    return _area;
}
