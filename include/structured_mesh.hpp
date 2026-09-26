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
#include <float_type.hpp>
#include <vector>
#include <Eigen/Dense>

/**
 * @class StructuredMesh
 * @brief Represents a structured mesh used in finite element analysis.
 * 
 * This class defines and initializes a structured mesh, with the ability to create
 * different cell types (triangles or quadrilaterals) and handle various operations
 * like connectivity, assembly of matrices, boundary conditions, and other mesh-related computations.
 */
class StructuredMesh {
private:
    /**
     * @brief Polynomial degree used for finite element method (FEM).
     * 
     * This variable determines the order of the basis functions in the FEM analysis.
     */
    int _degree;

    /**
     * @brief Total number of degrees of freedom (DOFs) in the mesh.
     * 
     * Represents the total number of unknowns in the system, calculated based on the number of vertices, edges, and elements.
     */
    int _nof_dofs;

    /**
     * @brief Total number of vertices in the mesh.
     * 
     * Typically calculated as the product of the number of nodes in the x and z directions: (nx+1) * (nz+1).
     */
    int _nof_verts;

    /**
     * @brief Total number of cells (elements) in the mesh.
     * 
     * This represents the number of discrete elements making up the computational domain.
     */
    int _nof_cells;

    /**
     * @brief Total number of edges in the mesh.
     * 
     * Represents the boundaries between the cells. The total number of edges is calculated based on the number of elements.
     */
    int _nof_edges;

    /**
     * @brief Number of degrees of freedom (DOFs) per cell.
     * 
     * This value depends on the element type and the polynomial degree, determining how many DOFs are associated with each cell.
     */
    int _dofs_per_cell;

    /**
     * @brief Number of edges per cell.
     * 
     * Typically, this value is 4 for quadrilateral elements or 3 for triangular elements.
     */
    int _edges_per_cell;

    /**
     * @brief Number of degrees of freedom (DOFs) associated with each edge.
     * 
     * This depends on the finite element formulation and the polynomial degree.
     */
    int _dofs_per_edge;

    /**
     * @brief Type of cell used in the mesh.
     * 
     * This integer value represents the cell type (e.g., triangular, quadrilateral) used in the finite element analysis.
     */
    int _cell_type;

    /**
     * @brief Number of divisions in the x-direction of the mesh.
     * 
     * Defines the number of cells (or nodes) in the x-direction of the computational domain.
     */
    int _nx;

    /**
     * @brief Number of divisions in the z-direction of the mesh.
     * 
     * Defines the number of cells (or nodes) in the z-direction of the computational domain.
     */
    int _nz;

    /**
     * @brief Number of horizontal divisions (cells or nodes) in the mesh.
     * 
     * Represents the number of divisions in the horizontal direction, determining the mesh grid's layout in the x-direction.
     */
    int _hl;

    /**
     * @brief Number of vertical divisions (cells or nodes) in the mesh.
     * 
     * Represents the number of divisions in the vertical direction, determining the mesh grid's layout in the z-direction.
     */
    int _vl;

    /**
     * @brief Indices of the degrees of freedom (DOFs) associated with the vertices.
     * 
     * This vector stores the indices of the DOFs for each vertex in the mesh. It is used to reference the DOFs associated with each vertex in the finite element analysis.
     */
    std::vector<int> vertex_dof_inds;

    /**
     * @brief Assembles the position matrix (pmat) for the mesh.
     * 
     * This method calculates the position matrix (pmat), which stores the (x, z) 
     * coordinates for each degree of freedom in the mesh. It generates two linearly 
     * spaced ranges for the x and z coordinates (based on the number of horizontal 
     * and vertical levels), and fills the matrix with corresponding values for each 
     * degree of freedom.
     * 
     * @details The x and z coordinates are computed using `Eigen::VectorX<FloatType>::LinSpaced()`. 
     * The method iterates over all the points in the grid and assigns the corresponding 
     * coordinates (x, z) to the matrix `pmat` and `pmat_unit_box`. The matrix `pmat` 
     * stores the positions in the global mesh space, while `pmat_unit_box` stores 
     * the positions in the unit box (normalized coordinates between 0 and 1).
     */
    void assemble_pmat();

    /**
     * @brief Assembles the connectivity matrix (cmat) for the mesh.
     * 
     * This method assembles the connectivity matrix `cmat` by iterating over all 
     * cells in the mesh and determining the connectivity for each cell. The function 
     * considers the cell type (`TRIANGLE_RIGHT`, `TRIANGLE_LEFT`, or `QUADRILATERAL`) 
     * and assigns the appropriate connectivity based on the element type. 
     * The matrix `cmat` stores the degrees of freedom (DOFs) for each cell in the mesh.
     * 
     * @details The method iterates over the mesh cells, stepping by the degree of 
     * the elements (`_degree`). Based on the chosen cell type, the corresponding 
     * connectivity function (`triangle_right_connectivity`, `triangle_left_connectivity`, 
     * or `quadrilateral_connectivity`) is called to update the `cmat`.
     */
    void assemble_cmat();

    /**
     * @brief Assembles the edge matrix (emat) for the mesh.
     * 
     * This method assembles the edge matrix `emat`, which stores the connectivity 
     * between the degrees of freedom (DOFs) for each edge in the mesh. It loops 
     * over the connectivity matrix `cmat`, processes each cell, and ensures that 
     * each edge is only recorded once, even if it is shared between two cells.
     * 
     * @details The method iterates over all the cells in the mesh and identifies 
     * edges based on the degrees of freedom of each cell. It ensures that each 
     * edge is only stored once in the edge matrix `emat` by using a set (`visited_edges`) 
     * to track the edges that have been processed. The `emat` matrix stores the DOFs 
     * of each edge for the finite element analysis.
     */
    void assemble_emat();

    /**
     * @brief Assembles the connectivity matrix (cmat) based on the cell type.
     * 
     * This method assembles the connectivity matrix `cmat` by calling different 
     * functions depending on the cell type: `QUADRILATERAL`, `TRIANGLE_LEFT`, or 
     * `TRIANGLE_RIGHT`. It ensures that the correct connectivity is calculated 
     * based on the element type for the mesh.
     * 
     * @details Depending on the value of `_cell_type`, this method delegates the 
     * assembly of the connectivity matrix to the appropriate method: 
     * `assemble_cimat_quadrilateral()`, `assemble_cimat_triangle_left()`, or 
     * `assemble_cimat_triangle_right()`.
     */
    void assemble_cimat();

    /**
     * @brief Assembles the edge matrix `eimat`, which contains information about the boundary conditions
     *        and the nodes that form each edge in the mesh.
     * 
     * This function creates a matrix `eimat` where each row corresponds to an edge, and each column 
     * stores information about the boundary conditions or the nodes that form the edge. The boundary 
     * conditions are determined based on the positions of the edge nodes in the `pmat` matrix, and the 
     * edges are classified as either boundary edges (e.g., NORTH, SOUTH, EAST, WEST) or interior edges 
     * depending on their locations within the domain.
     * 
     * @details The function first classifies the edges by their boundary conditions based on the nodes' 
     * positions. The edges are classified as:
     * - Boundary edges: Identified by their proximity to the boundaries of the domain (i.e., NORTH, SOUTH, 
     *   EAST, WEST).
     * - Interior edges: These edges do not touch the boundary and are classified as `INTERIOR_ID`.
     * The function uses `xtol` and `ztol` to handle the tolerance for boundary proximity.
     * 
     * After boundary classification, the function processes the `cmat` (connectivity matrix) to assign 
     * DOFs (degrees of freedom) to the edges, while ensuring that each edge is only processed once using 
     * the `visited_edges` set.
     * 
     * @note This method assumes that the degree of the elements (`_degree`) is either 1 or 2, and handles 
     * both cases appropriately when filling in the `eimat`.
     */
    void assemble_eimat();

    /**
     * @brief Assembles the boundary and interior node matrix `dimat`.
     * 
     * This method assigns boundary condition IDs (e.g., NORTH, SOUTH, EAST, WEST) 
     * or interior node IDs to the degrees of freedom (DOFs) based on their 
     * positions within the domain. The `pmat` matrix holds the positions of 
     * each node (degree of freedom) in the mesh, and the function uses these 
     * positions to determine whether a node lies on a boundary or within the 
     * interior of the mesh.
     * 
     * Boundary nodes are classified based on their proximity to the edges 
     * of the domain, with tolerance values applied to ensure correct classification.
     * 
     * @details The method iterates over all DOFs and checks the `pmat` matrix, 
     * which contains the (x, z) coordinates of each DOF. Based on these coordinates,
     * it assigns the appropriate boundary or interior condition:
     * - Corners: `NORTH_WEST_ID`, `SOUTH_WEST_ID`, `SOUTH_EAST_ID`, `NORTH_EAST_ID`
     * - Edges: `NORTH_ID`, `SOUTH_ID`, `WEST_ID`, `EAST_ID`
     * - Interior: `INTERIOR_ID`
     * 
     * The boundaries are defined using tolerances (`xtol` and `ztol`) to ensure 
     * that DOFs close to the domain's edges are correctly assigned to boundary conditions.
     */
    void assemble_dimat();

    /**
     * @brief Assembles the connectivity matrix for quadrilateral cells.
     * 
     * This method populates the connectivity matrix `cmat` for quadrilateral cells 
     * in the mesh. It assigns the appropriate connectivity IDs to the boundary 
     * elements (north, south, east, west) as well as interior elements. The 
     * method handles boundary conditions and assigns cell IDs for all the degrees 
     * of freedom in the mesh.
     * 
     * @details The method iterates through the cells and sets the corresponding 
     * cell types for boundary nodes (e.g., `MESH2D::SOUTH_WEST_ID`, `MESH2D::NORTH_EAST_ID`) 
     * and interior nodes (e.g., `MESH2D::INTERIOR_ID`). It first sets the corner 
     * elements (like the southwest and southeast corners), then fills in the 
     * boundary edges (north, south, east, west), and finally assigns interior 
     * node IDs to all the interior degrees of freedom.
     */
    void assemble_cimat_quadrilateral();

    /**
     * @brief Assembles the connectivity matrix for left-cut triangle cells.
     * 
     * This method is designed to assemble the connectivity matrix `cmat` for 
     * left-cut triangle cells. However, the implementation for this method 
     * is currently not provided. Typically, it would populate the `cmat` matrix 
     * by determining the connectivity of the degrees of freedom for each triangle 
     * based on the cell configuration and the mesh structure.
     * 
     * @details The method is a placeholder for future implementation where 
     * the specific logic for left-cut triangles should be written. It will 
     * interact with the mesh's properties such as `_degree` and `_cell_type` 
     * to determine the appropriate connectivity for each element.
     */
    void assemble_cimat_triangle_left();

    /**
     * @brief Assembles the connectivity matrix for right-oriented triangle cells.
     * 
     * This method is designed to assemble the connectivity matrix `cmat` for 
     * right-oriented triangle cells. Similar to `assemble_cimat_triangle_left()`, 
     * it is currently not implemented. In a completed implementation, this method 
     * would calculate the connectivity for the mesh's degrees of freedom and 
     * update the `cmat` matrix accordingly.
     * 
     * @details The method serves as a placeholder for future implementation of 
     * right-oriented triangle connectivity. Once implemented, it would process 
     * the mesh's cell structure and update the `cmat` matrix with the appropriate 
     * connectivity for the given mesh configuration.
     */
    void assemble_cimat_triangle_right();

    /**
     * @brief Computes the indices of the vertex degrees of freedom (DOFs).
     *
     * This function identifies and stores the indices of the DOFs that correspond 
     * to the vertices of the structured mesh. The number of vertex DOFs depends on 
     * the polynomial degree of the finite element basis functions.
     *
     * - For linear elements (degree = 1), every grid point is a vertex DOF.
     * - For quadratic elements (degree = 2), only every other grid point is a vertex DOF.
     *
     * The function reserves memory for the expected number of vertex DOFs to optimize performance.
     * The computed indices are stored in the `vertex_dof_inds` vector.
     */
    void compute_vertex_dof_inds();

    /**
     * @brief Computes the tangents and normals for each boundary edge in the mesh.
     * 
     * This function calculates the tangents and normals for each boundary edge of the mesh.
     * The tangents are calculated as the vector difference between the two vertices defining 
     * an edge, and then normalized. The normals are computed as the perpendicular vectors to 
     * the tangents, with one component flipped to ensure the correct orientation.
     * 
     * The tangents and normals are stored in the `edge_tangents` and `edge_normals` matrices, respectively.
     * 
     * Boundary edges are identified using their edge ID, which corresponds to the `MESH2D::BOUNDARY_ID`. 
     * For each boundary edge, the tangent and normal vectors are computed and stored.
     * 
     * @note This function only processes boundary edges, as they are the only ones for which tangents 
     *       and normals are typically computed in mesh processing.
     */
    void compute_edge_normals_and_tangents();

    /**
     * @brief Computes the tangents and normals for each vertex in the mesh.
     * 
     * This function calculates the tangents and normals for each vertex in the mesh. The tangents and 
     * normals are derived from the degrees of freedom (DOF) associated with the vertices. The function 
     * uses `vertex_dof_inds` (the indices of the DOFs associated with the vertices) to extract and compute 
     * the tangents and normals for each vertex.
     * 
     * The tangents and normals are computed using helper functions `dof_tangents` and `dof_normals`, 
     * which process the DOF values to derive these vectors.
     * 
     * @note This function operates on the vertices of the mesh and uses their corresponding DOF indices.
     */
    void compute_vertex_normals_and_tangents();

    /**
     * @brief Computes the normals and tangents for degrees of freedom (DOFs) associated with vertices and edges.
     * 
     * This function calculates the tangent and normal vectors for each degree of freedom (DOF) in the mesh. 
     * It starts by initializing matrices for the DOF tangents and normals. Then, it computes the normals and 
     * tangents for the vertex DOFs by extracting boundary edge information. The tangents and normals are then 
     * normalized for boundary vertices. The function also computes the tangents and normals for edge DOFs (if degree 
     * is 2) and handles corner DOFs by setting their normal and tangent values based on adjacent edge information.
     * 
     * @note Corner DOFs are specifically handled by copying the tangents and normals from adjacent edges.
     */
    void compute_dof_normals_and_tangents();

    /**
     * @brief Computes normals and tangents for edges, degrees of freedom (DOFs), and vertices.
     *
     * This function sequentially computes:
     *  - Edge normals and tangents using `compute_edge_normals_and_tangents()`
     *  - DOF normals and tangents using `compute_dof_normals_and_tangents()`
     *  - Vertex normals and tangents using `compute_vertex_normals_and_tangents()`
     *
     * These computations are essential for enforcing boundary conditions, 
     * computing fluxes, and solving PDEs on the structured mesh.
     */
    void compute_normals_and_tangents();

    /**
     * @brief Computes the connectivity for a left-cut triangle cell.
     * 
     * This method defines the connectivity of a triangular element with left 
     * orientation. It calculates the degrees of freedom (DOF) for each vertex 
     * and stores them in the connectivity matrix `cmat`. It handles both 
     * first-degree (linear) and second-degree (quadratic) elements.
     * 
     * @param[in] ci The cell index in the mesh.
     * @param[in] i The row index of the cell in the mesh.
     * @param[in] j The column index of the cell in the mesh.
     */
    void triangle_left_connectivity(int ci, int i, int j);

    /**
     * @brief Computes the connectivity for a right-oriented triangle cell.
     * 
     * This method defines the connectivity of a triangular element with right 
     * orientation. It calculates the degrees of freedom (DOF) for each vertex 
     * and stores them in the connectivity matrix `cmat`. The function handles 
     * both first-degree (linear) and second-degree (quadratic) elements.
     * 
     * @param[in] ci The cell index in the mesh.
     * @param[in] i The row index of the cell in the mesh.
     * @param[in] j The column index of the cell in the mesh.
     */
    void triangle_right_connectivity(int ci, int i, int j);

    /**
     * @brief Computes the connectivity for a quadrilateral cell.
     * 
     * This method defines the connectivity for quadrilateral elements. It calculates 
     * the degrees of freedom (DOF) for each vertex and stores them in the connectivity 
     * matrix `cmat`. The function supports both first-degree (linear) and 
     * second-degree (quadratic) elements.
     * 
     * @param[in] ci The cell index in the mesh.
     * @param[in] i The row index of the cell in the mesh.
     * @param[in] j The column index of the cell in the mesh.
     */
    void quadrilateral_connectivity(int ci, int i, int j);

    /**
     * @brief Returns the degree of freedom (DOF) index for the southwest corner of the mesh.
     * 
     * This function returns the index of the degree of freedom (DOF) corresponding to the southwest 
     * corner of the mesh.
     * 
     * @return The index of the southwest corner DOF (always 0).
     */
    const int corner_dof_SW();

    /**
     * @brief Returns the degree of freedom (DOF) index for the southeast corner of the mesh.
     * 
     * This function returns the index of the degree of freedom (DOF) corresponding to the southeast 
     * corner of the mesh.
     * 
     * @return The index of the southeast corner DOF.
     */
    const int corner_dof_SE();

    /**
     * @brief Returns the degree of freedom (DOF) index for the northeast corner of the mesh.
     * 
     * This function returns the index of the degree of freedom (DOF) corresponding to the northeast 
     * corner of the mesh.
     * 
     * @return The index of the northeast corner DOF.
     */
    const int corner_dof_NE();

    /**
     * @brief Returns the degree of freedom (DOF) index for the northwest corner of the mesh.
     * 
     * This function returns the index of the degree of freedom (DOF) corresponding to the northwest 
     * corner of the mesh.
     * 
     * @return The index of the northwest corner DOF.
     */
    const int corner_dof_NW();

    /**
     * @brief Checks if a given vertex index corresponds to a corner degree of freedom (DOF).
     * 
     * This function checks whether the given vertex index `vi` corresponds to one of the four corner 
     * DOFs (southwest, southeast, northwest, or northeast).
     * 
     * @param[in] vi The vertex index to check.
     * @return `true` if `vi` is a corner DOF, otherwise `false`.
     */
    bool is_corner_dof(int vi);

public:
    /**
     * @brief Node coordinates of the unit box mesh.
     * 
     * This matrix stores the reference (unit-box) coordinates of the mesh nodes before transformation.
     */
    Eigen::MatrixX<FloatType> pmat_unit_box;

    /**
     * @brief Node coordinates of the physical mesh.
     * 
     * Stores the actual coordinates of the mesh nodes in physical space after any transformations or extrusions.
     */
    Eigen::MatrixX<FloatType> pmat;

    /**
     * @brief Connectivity matrix defining element-to-node relationships.
     * 
     * Each row represents a mesh element and contains indices of the nodes that form that element.
     */
    Eigen::MatrixXi cmat;

    /**
     * @brief Edge connectivity matrix.
     * 
     * Each row represents an edge in the mesh and contains indices of the two nodes that define that edge.
     */
    Eigen::MatrixXi emat;

    /**
     * @brief Edge-to-element incidence matrix.
     * 
     * Each row represents an edge and stores the elements (cells) that share that edge.
     */
    Eigen::MatrixXi eimat;

    /**
     * @brief Vertex-to-domain classification vector.
     * 
     * Stores classification flags for each vertex, indicating whether it belongs to a specific boundary or domain region.
     */
    Eigen::VectorXi dimat;

    /**
     * @brief Cell-to-domain classification vector.
     * 
     * Stores classification flags for each cell, used to differentiate between various regions of the mesh.
     */
    Eigen::VectorXi cimat;

    /**
     * @brief List of all edges in the mesh.
     * 
     * Each row contains the indices of the two vertices that form an edge.
     */
    Eigen::MatrixXi edge_list;

    /**
     * @brief Tangent vectors of mesh edges.
     * 
     * Stores the unit tangent vectors for each edge in the mesh.
     */
    Eigen::MatrixX<FloatType> edge_tangents;

    /**
     * @brief Normal vectors of mesh edges.
     * 
     * Stores the unit normal vectors for each edge in the mesh, typically used for boundary conditions and flux computations.
     */
    Eigen::MatrixX<FloatType> edge_normals;

    /**
     * @brief Tangent vectors at mesh vertices.
     * 
     * Stores the average tangent vectors at each vertex, computed based on surrounding edges.
     */
    Eigen::MatrixX<FloatType> vertex_tangents;

    /**
     * @brief Normal vectors at mesh vertices.
     * 
     * Stores the average normal vectors at each vertex, computed based on surrounding edges.
     */
    Eigen::MatrixX<FloatType> vertex_normals;

    /**
     * @brief Tangent vectors at degrees of freedom (DOFs).
     * 
     * Stores the tangent vectors associated with each DOF, typically computed from surrounding mesh elements.
     */
    Eigen::MatrixX<FloatType> dof_tangents;

    /**
     * @brief Normal vectors at degrees of freedom (DOFs).
     * 
     * Stores the normal vectors associated with each DOF, typically computed from surrounding mesh elements.
     */
    Eigen::MatrixX<FloatType> dof_normals;

    /**
     * @brief Mapping from vertex indices to surface vertex indices.
     * 
     * Maps each vertex in the mesh to its corresponding surface vertex index, if applicable.
     */
    Eigen::VectorXi v2s_map;

    /**
     * @brief Mapping from vertex indices to bedrock vertex indices.
     * 
     * Maps each vertex in the mesh to its corresponding bedrock vertex index, if applicable.
     */
    Eigen::VectorXi v2b_map;

    /**
     * @class StructuredMesh
     * @brief Represents a structured mesh used in finite element analysis.
     * 
     * This class defines and initializes a structured mesh, with the ability to create
     * different cell types (triangles or quadrilaterals) and handle various operations
     * like connectivity, assembly of matrices, boundary conditions, and other mesh-related computations.
     */
    StructuredMesh(int nx, int nz, int degree, int cell_type);

    /**
     * @brief Extrudes the input vector `xvec_p1` along the x-axis to fill in the `pmat` matrix, which
     *        represents the mesh's coordinates along the x-direction.
     * 
     * This function takes an input vector `xvec_p1` and extrudes it along the x-axis to populate the 
     * `pmat` matrix. The degree of the mesh elements (1 or 2) determines how the input vector is 
     * processed. For linear elements (`_degree == 1`), the input vector is directly used. For quadratic 
     * elements (`_degree == 2`), the vector is expanded to accommodate intermediate nodes, and the 
     * values for the internal nodes are interpolated.
     * 
     * The function iterates over the mesh's vertices in the x-direction and assigns corresponding values 
     * to the `pmat` matrix. After populating the `pmat` matrix, it calls the `compute_normals_and_tangents` 
     * function to recompute the boundary normals and tangents for the mesh.
     * 
     * @param[in] xvec_p1 The input vector containing the x-coordinates of the mesh vertices at the next level 
     *                of extrusion.
     * 
     * @note The input vector `xvec_p1` contains the coordinates for the extruded mesh in the x-direction.
     *       The method handles linear and quadratic degrees of freedom, adjusting the mesh based on the element 
     *       degree.
     */
    void extrude_x(const Eigen::VectorX<FloatType> &xvec_p1);

    /**
     * @brief Linearly extrudes the mesh in the x-direction between given bounds.
     *
     * This function updates the x-coordinates of all degrees of freedom (DOFs)
     * to lie within the range `[x0, x1]`, while maintaining their relative positions.
     *
     * @param[in] x0 The starting x-coordinate (left boundary).
     * @param[in] x1 The ending x-coordinate (right boundary).
     *
     * After modifying the x-coordinates, the function recomputes boundary 
     * normals and tangents to ensure consistency.
     */
    void extrude_x(FloatType x0, FloatType x1);

    /**
     * @brief Extrudes the mesh in the z-direction using provided bottom and top surface expressions.
     *
     * This function modifies the z-coordinates of the mesh by applying two user-defined 
     * functions (`zb_expr` for the bottom surface and `zs_expr` for the top surface) 
     * to the x-coordinates of surface vertices. The mesh is then extruded accordingly.
     *
     * @param[in] zb_expr A function that computes the bottom surface elevation as a function of x.
     * @param[in] zs_expr A function that computes the top surface elevation as a function of x.
     *
     * After computing the new bottom and top elevations, the function calls `extrude_z()`
     * to update the mesh structure.
     */
    void extrude_z(
        std::function<FloatType (FloatType)> zb_expr,
        std::function<FloatType (FloatType)> zs_expr
    );

    /**
     * @brief Extrudes the mesh in the z-direction using specified bottom and top surface elevations.
     *
     * This function updates the z-coordinates of the mesh based on the provided vectors `zb_vec_p1` 
     * and `zs_vec_p1`, which represent the bottom and top surface elevations at vertices points.
     * 
     * The method ensures that the correct z-positions are assigned to all cell corner nodes 
     * and interpolates values for interior and edge nodes when using quadratic elements.
     * After updating the mesh geometry, it recomputes boundary normals and tangents.
     *
     * @param[in] zb_vec_p1 A vector containing the z-coordinates of the bottom surface at mesh vertices.
     * @param[in] zs_vec_p1 A vector containing the z-coordinates of the top surface at mesh vertices.
     */
    void extrude_z(
        const Eigen::VectorX<FloatType> &zb_vec_p1,
        const Eigen::VectorX<FloatType> &zs_vec_p1
    );
    /**
     * @brief Extrudes the mesh in the z-direction using a specified top surface elevation.
     *
     * This function updates the z-coordinates of the mesh while keeping the current 
     * bottom surface elevation unchanged. It extracts the existing bedrock elevations 
     * from the mesh and calls the overloaded `extrude_z` function to apply the new 
     * top surface elevation.
     *
     * @param[in] zs_vec_p1 A vector containing the z-coordinates of the top surface at mesh vertices.
     */
    void extrude_z(const Eigen::VectorX<FloatType> &zs_vec_p1);

    /**
     * @brief Extracts the indices of mesh cells that match a given identifier.
     *
     * This function scans through the mesh connectivity matrix (`cimat`) and collects
     * the indices of cells that have the specified `id` flag set.
     *
     * @param[in] id The identifier used to filter cells.
     * @return A vector containing the indices of the matching cells.
     */
    std::vector<int> extract_cell_inds(int id);

    /**
     * @brief Extracts indices of edges that match a given identifier.
     *
     * This function scans through the edge matrix (`eimat`) and collects
     * the indices of edges whose first entry contains the specified `id`.
     *
     * @param[in] id The identifier used to filter edges.
     * @return A vector containing the indices of the matching edges.
     */
    std::vector<int> extract_edge_inds(int id);
    std::vector<int> extract_dof_inds(int id);

    /**
     * @brief Extracts indices of vertex degrees of freedom (DOFs) that match a given identifier.
     *
     * This function filters the vertex DOFs based on the provided `id`, checking if the
     * identifier is set in the `dimat` matrix.
     *
     * @param[in] id The identifier used to filter vertex DOFs.
     * @return A vector containing the indices of the matching vertex DOFs.
     */
    std::vector<int> extract_vertex_dof_inds(int id);

    /**
     * @brief Returns the number of cells in the x-direction (horizontal grid size).
     * @return The value of _nx.
     */
    int nx();

    /**
     * @brief Returns the number of cells in the z-direction (vertical grid size).
     * @return The value of _nz.
     */
    int nz();

    /**
     * @brief Returns the number of layers in the x-direction of the mesh.
     * @return The value of _hl.
     */
    int hl();

    /**
     * @brief Returns the number of layers in the z-direction of the mesh.
     * @return The value of _vl.
     */
    int vl();

    /**
     * @brief Returns the degree of the mesh (e.g., linear or quadratic).
     * @return The value of _degree.
     */
    int degree();

    /**
     * @brief Returns the total number of vertices in the mesh.
     * @return The value of _nof_verts.
     */
    int nof_verts();

    /**
     * @brief Returns the total number of degrees of freedom in the mesh.
     * @return The value of _nof_dofs.
     */
    int nof_dofs();

    /**
     * @brief Returns the total number of cells in the mesh.
     * @return The value of _nof_cells.
     */
    int nof_cells();

    /**
     * @brief Returns the total number of edges in the mesh.
     * @return The value of _nof_edges.
     */
    int nof_edges();

    /**
     * @brief Returns the number of degrees of freedom per cell.
     * @return The value of _dofs_per_cell.
     */
    int dofs_per_cell();

    /**
     * @brief Returns the number of degrees of freedom per edge.
     * @return The value of _dofs_per_edge.
     */
    int dofs_per_edge();

    /**
     * @brief Returns the type of the mesh cell.
     * @return The value of _cell_type.
     */
    int cell_type();

    /**
     * @brief Computes the total area of the mesh by integrating the determinant of the Jacobian 
     * over all cells using Gaussian quadrature.
     * 
     * This function computes the area of the mesh by performing numerical integration on the 
     * reference cell using Gaussian quadrature. It applies the necessary transformations from 
     * reference coordinates to physical coordinates for each element in the mesh. The area is 
     * calculated as the sum of the contributions from all cells in the mesh. It uses basis functions 
     * and their derivatives to map the quadrature points and compute the Jacobian determinant.
     * 
     * The function performs the following steps:
     * 1. Uses Gaussian quadrature to get quadrature points (`qpoints_rs`) and weights (`qweights`) 
     *    in reference coordinates for integration.
     * 2. Computes the Lagrange basis functions and their derivatives at the quadrature points.
     * 3. Iterates over each cell, mapping the quadrature points from the reference element to 
     *    the physical element using the Jacobian matrix (`detJ_rs`).
     * 4. Computes the area contribution from each quadrature point using the absolute value of the 
     *    Jacobian determinant and its corresponding quadrature weight.
     * 5. Returns the total area by summing the contributions from all quadrature points in all cells.
     * 
     * @return The computed total area of the mesh as a `FloatType` value.
     */
    FloatType area();
};
