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
#include <fem_1d.hpp>
#include <fem_2d.hpp>
#include <enums.hpp>
#include <boost/format.hpp>

// Nonlinear viscosity function
static inline FloatType glen_flow_viscosity(
    FloatType A, FloatType n_i, FloatType eps_reg_2, FloatType eff_strain_rate_2
) {
    return 0.5*POW_FUNC(A, -1.0/n_i)*POW_FUNC(
        eff_strain_rate_2 + eps_reg_2, (1.0-n_i)/(2.0*n_i)
    );
}

pStokesProblem::pStokesProblem(
    FloatType rate_factor, FloatType glen_exponent, FloatType eps_reg_2,
    std::function<FloatType(FloatType, FloatType)> force_x,
    std::function<FloatType(FloatType, FloatType)> force_z,
    StructuredMesh &u_mesh, StructuredMesh &p_mesh
) :
    rate_factor(rate_factor), glen_exponent(glen_exponent), eps_reg_2(eps_reg_2),
    force_x(force_x), force_z(force_z), u_mesh(u_mesh), p_mesh(p_mesh),
    logger(INFO, std::cout)
{
    // Set the number of dofs for velocity and pressure fields
    nv_dofs = u_mesh.nof_dofs();  // Number of velocity dofs (both horizontal and vertical)
    np_dofs = p_mesh.nof_dofs();  // Number of pressure dofs
    n_dofs = 2*nv_dofs + np_dofs;  // Total number of dofs (velocity + pressure)

    // Initialize velocity and pressure vectors for the FEM system (2D)
    ux_v2d = Eigen::VectorXi::Zero(nv_dofs);  // Horizontal velocity dofs vector
    uz_v2d = Eigen::VectorXi::Zero(nv_dofs);  // Vertical velocity dofs vector
    u_v2d = Eigen::VectorXi::Zero(2*nv_dofs);  // Combined velocity dofs vector
    p_v2d = Eigen::VectorXi::Zero(np_dofs);  // Pressure dofs vector

    // Initialize dof to variable index mappings
    ux_d2v = Eigen::VectorXi::Constant(n_dofs, -1);  // Mapping for horizontal velocity to dof
    uz_d2v = Eigen::VectorXi::Constant(n_dofs, -1);  // Mapping for vertical velocity to dof

    // Fill the ux_v2d and ux_d2v vectors for horizontal velocity components
    int dof = 0;
    for (int vi = 0; vi < nv_dofs; ++vi) {
        ux_v2d(vi) = dof;  // Set the index for horizontal velocity component
        ux_d2v(dof) = vi;  // Set the mapping from dof to velocity index
        dof += 2;
    }

    // Fill the uz_v2d and uz_d2v vectors for vertical velocity components
    dof = 1;
    for (int vi = 0; vi < nv_dofs; ++vi) {
        uz_v2d(vi) = dof;  // Set the index for vertical velocity component
        uz_d2v(dof) = vi;  // Set the mapping from dof to velocity index
        dof += 2;
    }

    // Fill the p_v2d vector for pressure components
    dof = 2*nv_dofs;
    for (int vi = 0; vi < np_dofs; ++vi) {
        p_v2d(vi) = dof;  // Set the index for pressure component
        dof += 1;  // Increment dof for the next pressure component
    }

    // Combine horizontal and vertical velocity mappings into u_v2d
    u_v2d(Eigen::seqN(0, nv_dofs, 2)) = ux_v2d;  // Horizontal velocity vector
    u_v2d(Eigen::seq(1, 2*nv_dofs, 2)) = uz_v2d;  // Vertical velocity vector

    // Initialize sparse matrices for the system
    lhs_mat = Eigen::SparseMatrix<FloatType>(n_dofs, n_dofs);  // Left-hand side matrix
    rhs_vec = Eigen::VectorX<FloatType>::Zero(n_dofs);  // Right-hand side vector
    w_vec = Eigen::VectorX<FloatType>::Zero(n_dofs);  // Solution vector

    // Estimate the maximum number of non-zero entries for the left-hand side matrix
    int u_nnz_per_dof = 2*(2*u_mesh.degree() + 1)*(2*u_mesh.degree() + 1);  // Non-zeros per velocity dof
    int p_nnz_per_dof = (2*p_mesh.degree() + 1)*(2*p_mesh.degree() + 1);  // Non-zeros per pressure dof
    int nnz_per_dof = u_nnz_per_dof + p_nnz_per_dof;  // Total non-zeros per dof
    lhs_coeffs.reserve(nnz_per_dof * n_dofs);  // Reserve memory for the left-hand side matrix coefficients
}

void pStokesProblem::assemble_stress_block()
{
    // Initialize matrices and vectors for computation
    Eigen::MatrixX<FloatType> node_coords_u, qpoints_rs, qpoints_xz,
        phi_rs, dphi_rs, dphi_xz;
    Eigen::VectorX<FloatType> qweights, detJ_rs;
    Eigen::VectorXi element_u;

    // Perform Gauss-Legendre quadrature to get quadrature points and weights
    FEM2D::gauss_legendre_quadrature(
        gp_stress, u_mesh.cell_type(), qpoints_rs, qweights
    );

    // Generate Lagrange basis functions for the quadrature points
    FEM2D::lagrange_basis(
        u_mesh.degree(),
        u_mesh.cell_type(),
        qpoints_rs,
        phi_rs, dphi_rs
    );

    // Initialize determinant of the Jacobian and other variables
    detJ_rs = Eigen::VectorX<FloatType>::Zero(qpoints_rs.rows());
    qpoints_xz = Eigen::MatrixX<FloatType>::Zero(qpoints_rs.rows(), 2);
    dphi_xz = Eigen::MatrixX<FloatType>::Zero(2*qpoints_rs.rows(), u_mesh.dofs_per_cell());

    // Initialize matrices for stress block components (A_xx, A_xz, A_zz)
    Eigen::MatrixX<FloatType> A_xx = Eigen::MatrixX<FloatType>::Zero(
        u_mesh.dofs_per_cell(), u_mesh.dofs_per_cell()
    );
    Eigen::MatrixX<FloatType> A_xz = Eigen::MatrixX<FloatType>::Zero(
        u_mesh.dofs_per_cell(), u_mesh.dofs_per_cell()
    );
    Eigen::MatrixX<FloatType> A_zz = Eigen::MatrixX<FloatType>::Zero(
        u_mesh.dofs_per_cell(), u_mesh.dofs_per_cell()
    );

    // Initialize the full stress block matrix
    Eigen::MatrixX<FloatType> A_block = Eigen::MatrixX<FloatType>::Zero(
        2*u_mesh.dofs_per_cell(), 2*u_mesh.dofs_per_cell()
    );

    // Loop over all mesh cells
    for (int k = 0; k < u_mesh.cmat.rows(); ++k) {
        // Retrieve the element indices and corresponding node coordinates
        element_u = u_mesh.cmat.row(k);
        node_coords_u = u_mesh.pmat(element_u, Eigen::all);

        // Map the element to the reference cell
        FEM2D::map_to_reference_cell(
            u_mesh.degree(), u_mesh.cell_type(), node_coords_u, qpoints_rs,
            phi_rs, dphi_rs, detJ_rs, qpoints_xz, dphi_xz
        );

        // Extract velocity components (ux and uz) on the current element
        Eigen::VectorX<FloatType> ux_vec = w_vec(ux_v2d(element_u));
        Eigen::VectorX<FloatType> uz_vec = w_vec(uz_v2d(element_u));

        // Split the derivative matrices for the horizontal and vertical velocity components
        Eigen::MatrixX<FloatType> Sx = dphi_xz(Eigen::seq(0, Eigen::last, 2), Eigen::all);
        Eigen::MatrixX<FloatType> Sz = dphi_xz(Eigen::seq(1, Eigen::last, 2), Eigen::all);

        // Compute the gradients of the velocity components
        Eigen::VectorX<FloatType> ddx_ux = Sx*ux_vec;
        Eigen::VectorX<FloatType> ddz_ux = Sz*ux_vec;
        Eigen::VectorX<FloatType> ddx_uz = Sx*uz_vec;
        Eigen::VectorX<FloatType> ddz_uz = Sz*uz_vec;

        // Loop over quadrature points
        for (int q = 0; q < qpoints_rs.rows(); ++q) {
            int wx = 2*q;  // Index of x-derivative
            int wz = wx + 1;  // Index of z-derivative

            // Compute the effective strain rate based on the velocity gradients
            FloatType eff_strain_rate_2 = 0.5*(
                POW_FUNC(ddx_ux(q), 2)
                + 0.5*POW_FUNC(ddz_ux(q) + ddx_uz(q), 2)
                + POW_FUNC(ddz_uz(q), 2)
            );

            // Glen's flow law rheology to compute viscosity
            FloatType eta = glen_flow_viscosity(
                rate_factor, glen_exponent, eps_reg_2, eff_strain_rate_2
            );

            // Multiply by the determinant of the Jacobian and quadrature weight
            FloatType detJxWxEta = ABS_FUNC(detJ_rs(q)) * qweights(q) * eta;

            // Assemble the stress block matrix (A_xx, A_xz, A_zz)
            for (int i = 0; i < u_mesh.dofs_per_cell(); i++) {
                for (int j = 0; j < u_mesh.dofs_per_cell(); j++) {
                    // Add contributions to the stress matrix components
                    A_xx(i, j) += (2.0*dphi_xz(wx, i)*dphi_xz(wx, j) + dphi_xz(wz, i)*dphi_xz(wz, j))*detJxWxEta;
                    A_xz(i, j) += dphi_xz(wz, i)*dphi_xz(wx, j) * detJxWxEta;
                    A_zz(i, j) += (dphi_xz(wx, i)*dphi_xz(wx, j) + 2.0*dphi_xz(wz, i)*dphi_xz(wz, j))* detJxWxEta;
                }
            }
        }

        // Push the computed values into the coefficient list for the left-hand side matrix
        for (int i = 0; i < u_mesh.dofs_per_cell(); ++i) {
            for (int j = 0; j < u_mesh.dofs_per_cell(); ++j) {
                // Add coefficients for horizontal velocity (A_xx)
                lhs_coeffs.push_back(Eigen::Triplet<FloatType>(
                    ux_v2d(element_u(i)),
                    ux_v2d(element_u(j)),
                    A_xx(i, j)
                ));
                // Add coefficients for horizontal-vertical velocity (A_xz)
                lhs_coeffs.push_back(Eigen::Triplet<FloatType>(
                    ux_v2d(element_u(i)),
                    uz_v2d(element_u(j)),
                    A_xz(i, j)
                ));
                // Add coefficients for vertical-horizontal velocity (A_xz)
                lhs_coeffs.push_back(Eigen::Triplet<FloatType>(
                    uz_v2d(element_u(i)),
                    ux_v2d(element_u(j)),
                    A_xz(j, i)
                ));
                // Add coefficients for vertical velocity (A_zz)
                lhs_coeffs.push_back(Eigen::Triplet<FloatType>(
                    uz_v2d(element_u(i)),
                    uz_v2d(element_u(j)),
                    A_zz(i, j)
                ));
            }
        }

        // Reset the matrices for the next element
        A_xx.setZero();
        A_xz.setZero();
        A_zz.setZero();
    }

    // Update the number of pushed elements in the lhs_coeffs list
    nof_pushed_elements = lhs_coeffs.size();
}

void pStokesProblem::assemble_incomp_block()
{
    // Declare matrices and vectors for node coordinates, quadrature points, and derivatives
    Eigen::MatrixX<FloatType> node_coords_u, node_coords_p, qpoints_rs, qpoints_xz,
        phi_rs, dphi_rs, dphi_xz, psi_rs, dpsi_rs, dpsi_xz;
    Eigen::VectorX<FloatType> qweights, detJ_rs;
    Eigen::VectorXi element_u, element_p;

    // Perform Gauss-Legendre quadrature to calculate quadrature points and weights
    FEM2D::gauss_legendre_quadrature(
        gp_incomp, u_mesh.cell_type(), qpoints_rs, qweights
    );

    // Generate Lagrange basis functions and their derivatives for pressure (p) element
    FEM2D::lagrange_basis(
        p_mesh.degree(), p_mesh.cell_type(), qpoints_rs, psi_rs, dpsi_rs
    );

    // Generate Lagrange basis functions and their derivatives for velocity (u) element
    FEM2D::lagrange_basis(
        u_mesh.degree(), u_mesh.cell_type(), qpoints_rs, phi_rs, dphi_rs
    );

    // Initialize the determinant of Jacobian and other matrices required for integration
    detJ_rs = Eigen::VectorX<FloatType>::Zero(qpoints_rs.rows());
    qpoints_xz = Eigen::MatrixX<FloatType>::Zero(qpoints_rs.rows(), 2);
    dphi_xz = Eigen::MatrixX<FloatType>::Zero(2*qpoints_rs.rows(), u_mesh.dofs_per_cell());
    dpsi_xz = Eigen::MatrixX<FloatType>::Zero(2*qpoints_rs.rows(), p_mesh.dofs_per_cell());

    // Declare matrices to store contributions to the incompressibility block (B_px, B_pz, B_xp, B_zp)
    Eigen::MatrixX<FloatType> B_px = Eigen::MatrixX<FloatType>::Zero(
        p_mesh.dofs_per_cell(), u_mesh.dofs_per_cell()
    );
    Eigen::MatrixX<FloatType> B_pz = Eigen::MatrixX<FloatType>::Zero(
        p_mesh.dofs_per_cell(), u_mesh.dofs_per_cell()
    );
    Eigen::MatrixX<FloatType> B_xp = Eigen::MatrixX<FloatType>::Zero(
        u_mesh.dofs_per_cell(), p_mesh.dofs_per_cell()
    );
    Eigen::MatrixX<FloatType> B_zp = Eigen::MatrixX<FloatType>::Zero(
        u_mesh.dofs_per_cell(), p_mesh.dofs_per_cell()
    );

    // Loop over all elements in the mesh to compute the contributions to the incompressibility block
    for (int k = 0; k < u_mesh.cmat.rows(); ++k) {
        // Get the node indices for velocity (u) and pressure (p) elements
        element_u = u_mesh.cmat.row(k);
        element_p = p_mesh.cmat.row(k);

        // Get the coordinates of the nodes for velocity and pressure elements
        node_coords_u = u_mesh.pmat(element_u, Eigen::all);
        node_coords_p = p_mesh.pmat(element_p, Eigen::all);

        // Map the pressure element node coordinates to the reference cell
        FEM2D::map_to_reference_cell(
            p_mesh.degree(), p_mesh.cell_type(), node_coords_p, qpoints_rs,
            psi_rs, dpsi_rs, detJ_rs, qpoints_xz, dpsi_xz
        );

        // Map the velocity element node coordinates to the reference cell
        FEM2D::map_to_reference_cell(
            u_mesh.degree(), u_mesh.cell_type(), node_coords_u, qpoints_rs,
            phi_rs, dphi_rs, detJ_rs, qpoints_xz, dphi_xz
        );

        // Loop over all quadrature points to compute the matrix entries for this element
        for (int q = 0; q < qpoints_rs.rows(); ++q) {
            int w = 2*q;  // Index for the quadrature points corresponding to horizontal velocity
            // Calculate the weighted Jacobian determinant for this quadrature point
            FloatType detJxW = ABS_FUNC(detJ_rs(q)) * qweights(q);

            // Loop over the dofs for pressure and velocity to compute the B matrices
            for (int i = 0; i < p_mesh.dofs_per_cell(); ++i) {
                for (int j = 0; j < u_mesh.dofs_per_cell(); ++j) {
                    B_px(i, j) += -psi_rs(q, i) * dphi_xz(w, j) * detJxW;
                    B_pz(i, j) += -psi_rs(q, i) * dphi_xz(w + 1, j) * detJxW;
                }
            }
            // Loop over the dofs for velocity and pressure to compute the B matrices
            for (int i = 0; i < u_mesh.dofs_per_cell(); ++i) {
                for (int j = 0; j < p_mesh.dofs_per_cell(); ++j) {
                    B_xp(i, j) += -dphi_xz(w, i) * psi_rs(q, j) * detJxW;
                    B_zp(i, j) += -dphi_xz(w + 1, i) * psi_rs(q, j) * detJxW;
                }
            }
        }

        // Insert the computed values into the sparse matrix (lhs_coeffs) for pressure-velocity terms
        for (int i = 0; i < p_mesh.dofs_per_cell(); ++i) {
            for (int j = 0; j < u_mesh.dofs_per_cell(); ++j) {
                lhs_coeffs.push_back(Eigen::Triplet<FloatType>(
                    p_v2d(element_p(i)), ux_v2d(element_u(j)), B_px(i, j)
                ));
                lhs_coeffs.push_back(Eigen::Triplet<FloatType>(
                    p_v2d(element_p(i)), uz_v2d(element_u(j)), B_pz(i, j)
                ));
            }
        }

        // Insert the computed values into the sparse matrix (lhs_coeffs) for velocity-pressure terms
        for (int i = 0; i < u_mesh.dofs_per_cell(); ++i) {
            for (int j = 0; j < p_mesh.dofs_per_cell(); ++j) {
                lhs_coeffs.push_back(Eigen::Triplet<FloatType>(
                    ux_v2d(element_u(i)), p_v2d(element_p(j)), B_xp(i, j)
                ));
                lhs_coeffs.push_back(Eigen::Triplet<FloatType>(
                    uz_v2d(element_u(i)), p_v2d(element_p(j)), B_zp(i, j)
                ));
            }
        }

        // Reset blocks
        B_px.setZero();
        B_pz.setZero();
        B_xp.setZero();
        B_zp.setZero();
    }

    // Track the number of elements inserted into the lhs_coeffs matrix
    nof_pushed_elements = lhs_coeffs.size();
}

void pStokesProblem::assemble_fssa_vertical_block() {
    // Declare matrices and vectors for quadrature points, basis functions, and other required values
    Eigen::MatrixX<FloatType> node_coords, qpoints_x, phi_r, dphi_r, dphi_x;
    Eigen::VectorX<FloatType> qweights, qpoints_r, detJ_r;
    Eigen::VectorXi edge_vi;

    // Retrieve quadrature points and weights
    FEM1D::gauss_legendre_quadrature(gp_fssa_lhs, qpoints_r, qweights);

    // Calculate Lagrange basis functions and their derivatives for the velocity element (u_mesh)
    FEM1D::lagrange_basis(
        u_mesh.degree(), qpoints_r, phi_r, dphi_r
    );

    // Initialize matrices for storing quadrature points in the x-direction, determinant of the Jacobian, and derivative of basis functions
    qpoints_x = Eigen::MatrixX<FloatType>::Zero(qpoints_r.rows(), 2);
    detJ_r = Eigen::VectorX<FloatType>::Zero(qpoints_r.rows());
    dphi_x = Eigen::MatrixX<FloatType>::Zero(
        2*qpoints_r.rows(), u_mesh.dofs_per_edge()
    );

    // Initialize matrices for the vertical and horizontal force contributions
    Eigen::MatrixX<FloatType> A_xz = Eigen::MatrixX<FloatType>::Zero(
        u_mesh.dofs_per_edge(), u_mesh.dofs_per_edge()
    );
    Eigen::MatrixX<FloatType> A_zz = Eigen::MatrixX<FloatType>::Zero(
        u_mesh.dofs_per_edge(), u_mesh.dofs_per_edge()
    );

    // Extract the indices of the surface edges from the mesh
    std::vector<int> surf_edge_inds = u_mesh.extract_edge_inds(MESH2D::SURFACE_ID);

    // Loop over the surface edges to assemble the contributions to the stiffness matrices
    for (int si : surf_edge_inds) {
        // Get the node indices for the current edge and extract the node coordinates
        edge_vi = u_mesh.emat(si, Eigen::all); // take out coordinates for the edge-borders
        node_coords = u_mesh.pmat(edge_vi, Eigen::all);

        // Map the edge coordinates to the reference cell (local coordinates)
        FEM1D::map_to_reference_cell(
            u_mesh.degree(), node_coords, qpoints_r,
            phi_r, dphi_r, detJ_r, qpoints_x, dphi_x
        );

        // Retrieve the surface normal vector components (nx, nz) for the current edge
        FloatType nx = u_mesh.edge_normals(si, 0);
        FloatType nz = u_mesh.edge_normals(si, 1);

        // Loop over quadrature points in edge
        for (int q = 0; q < qpoints_r.rows(); ++q) {
            FloatType fz = force_z(qpoints_x(q, 0), qpoints_x(q, 1));

            // Compute the weighted Jacobian determinant (detJ * weight)
            FloatType dFxW = ABS_FUNC(detJ_r(q)) * qweights(q);

            // Loop over edge dofs and compute the matrix contributions
            for (int i = 0; i < u_mesh.dofs_per_edge(); i++) {
                for (int j = 0; j < u_mesh.dofs_per_edge(); j++) {
                    A_xz(i, j) -= fssa_param * phi_r(q, i) * phi_r(q, j) * fz * dFxW * nx;
                    A_zz(i, j) -= fssa_param * phi_r(q, i) * phi_r(q, j) * fz * dFxW * nz;
                }
            }
        }

        // Insert the contributions to the sparse matrix (lhs_coeffs) for the vertical block
        for (int i = 0; i < u_mesh.dofs_per_edge(); ++i) {
            for (int j = 0; j < u_mesh.dofs_per_edge(); ++j) {
                lhs_coeffs.push_back(Eigen::Triplet<FloatType>(
                    uz_v2d(edge_vi(i)),
                    ux_v2d(edge_vi(j)),
                    A_xz(i, j)
                ));
                lhs_coeffs.push_back(Eigen::Triplet<FloatType>(
                    uz_v2d(edge_vi(i)),
                    uz_v2d(edge_vi(j)),
                    A_zz(i, j)
                ));
            }
        }

        // Reset the blocks
        A_xz.setZero();
        A_zz.setZero();
    }

    // Track the number of elements inserted into the lhs_coeffs matrix
    nof_pushed_elements = lhs_coeffs.size();
}

void pStokesProblem::assemble_fssa_normal_block() {
    // Declare matrices and vectors for quadrature points, basis functions, and other required values
    Eigen::MatrixX<FloatType> node_coords_u, qpoints_x, phi_r, dphi_r, dphi_x;
    Eigen::VectorX<FloatType> qweights, qpoints_r, detJ_r;
    Eigen::VectorXi edge_vi;

    // Retrieve quadrature points and weights using Gauss-Legendre quadrature
    FEM1D::gauss_legendre_quadrature(gp_fssa_lhs, qpoints_r, qweights);

    // Calculate Lagrange basis functions and their derivatives for the velocity element (u_mesh)
    FEM1D::lagrange_basis(
        u_mesh.degree(), qpoints_r, phi_r, dphi_r
    );

    // Initialize matrices for storing quadrature points in the x-direction, determinant of the Jacobian, and derivative of basis functions
    qpoints_x = Eigen::MatrixX<FloatType>::Zero(qpoints_r.rows(), 2);
    detJ_r = Eigen::VectorX<FloatType>::Zero(qpoints_r.rows());
    dphi_x = Eigen::MatrixX<FloatType>::Zero(
        2*qpoints_r.rows(), u_mesh.dofs_per_edge()
    );

    // Initialize matrices for stiffness matrix contributions
    Eigen::MatrixX<FloatType> A_xx = Eigen::MatrixX<FloatType>::Zero(
        u_mesh.dofs_per_edge(), u_mesh.dofs_per_edge()
    );
    Eigen::MatrixX<FloatType> A_xz = Eigen::MatrixX<FloatType>::Zero(
        u_mesh.dofs_per_edge(), u_mesh.dofs_per_edge()
    );
    Eigen::MatrixX<FloatType> A_zx = Eigen::MatrixX<FloatType>::Zero(
        u_mesh.dofs_per_edge(), u_mesh.dofs_per_edge()
    );
    Eigen::MatrixX<FloatType> A_zz = Eigen::MatrixX<FloatType>::Zero(
        u_mesh.dofs_per_edge(), u_mesh.dofs_per_edge()
    );

    // Extract the indices of the surface edges from the mesh
    std::vector<int> surf_edge_inds = u_mesh.extract_edge_inds(MESH2D::SURFACE_ID);

    // Loop over the surface edges to assemble the contributions to the stiffness matrices
    for (int si : surf_edge_inds) {
        // Get the node indices for the current edge and extract the node coordinates
        edge_vi = u_mesh.emat(si, Eigen::all);
        node_coords_u = u_mesh.pmat(edge_vi, Eigen::all);

        // Map the edge coordinates to the reference cell (local coordinates)
        FEM1D::map_to_reference_cell(
            u_mesh.degree(), node_coords_u, qpoints_r,
            phi_r, dphi_r, detJ_r, qpoints_x, dphi_x
        );

        // Retrieve the surface normal vector components (nx, nz) for the current edge
        FloatType nz = sqrt(u_mesh.edge_normals(si, 1));
        FloatType nx = u_mesh.edge_normals(si, 0) / nz;

        // Loop over quadrature points in edge and assemble matrix contributions
        for (int q = 0; q < qpoints_r.rows(); ++q) {
            FloatType fz = force_z(qpoints_x(q, 0), qpoints_x(q, 1));

            // Compute the weighted Jacobian determinant (detJ * weight)
            FloatType dFxW = ABS_FUNC(detJ_r(q)) * qweights(q);

            // Loop over edge dofs and compute the matrix contributions
            for (int i = 0; i < u_mesh.dofs_per_edge(); ++i) {
                for (int j = 0; j < u_mesh.dofs_per_edge(); ++j) {
                    A_xx(i, j) -= fssa_param * phi_r(q, i) * phi_r(q, j) * fz * dFxW * nx * nx;
                    A_xz(i, j) -= fssa_param * phi_r(q, i) * phi_r(q, j) * fz * dFxW * nx * nz;
                    A_zx(i, j) -= fssa_param * phi_r(q, i) * phi_r(q, j) * fz * dFxW * nz * nx;
                    A_zz(i, j) -= fssa_param * phi_r(q, i) * phi_r(q, j) * fz * dFxW * nz * nz;
                }
            }
        }

        // Insert the contributions to the sparse matrix (lhs_coeffs) for the normal block
        for (int i = 0; i < u_mesh.dofs_per_edge(); ++i) {
            for (int j = 0; j < u_mesh.dofs_per_edge(); ++j) {
                lhs_coeffs.push_back(Eigen::Triplet<FloatType>(
                    ux_v2d(edge_vi(i)),
                    ux_v2d(edge_vi(j)),
                    A_xx(i, j)
                ));
                lhs_coeffs.push_back(Eigen::Triplet<FloatType>(
                    ux_v2d(edge_vi(i)),
                    uz_v2d(edge_vi(j)),
                    A_xz(i, j)
                ));
                lhs_coeffs.push_back(Eigen::Triplet<FloatType>(
                    uz_v2d(edge_vi(i)),
                    ux_v2d(edge_vi(j)),
                    A_zx(i, j)
                ));
                lhs_coeffs.push_back(Eigen::Triplet<FloatType>(
                    uz_v2d(edge_vi(i)),
                    uz_v2d(edge_vi(j)),
                    A_zz(i, j)
                ));
            }
        }

        // Reset the matrices for the next iteration
        A_xx.setZero();
        A_xz.setZero();
        A_zx.setZero();
        A_zz.setZero();
    }

    // Track the number of elements inserted into the lhs_coeffs matrix
    nof_pushed_elements = lhs_coeffs.size();
}

void pStokesProblem::assemble_fssa_vertical_rhs() {
    // Declare matrices and vectors for quadrature points, basis functions, and other required values
    Eigen::MatrixX<FloatType> node_coords_u, qpoints_x, phi_r, dphi_r, dphi_x;
    Eigen::VectorX<FloatType> qweights, qpoints_r, detJ_r;
    Eigen::VectorXi edge_vi;

    // Retrieve quadrature points and weights using Gauss-Legendre quadrature
    FEM1D::gauss_legendre_quadrature(gp_fssa_rhs, qpoints_r, qweights);

    // Calculate Lagrange basis functions and their derivatives for the velocity element (u_mesh)
    FEM1D::lagrange_basis(u_mesh.degree(), qpoints_r, phi_r, dphi_r);

    // Initialize matrices for storing quadrature points in the x-direction, determinant of the Jacobian, and derivative of basis functions
    qpoints_x = Eigen::MatrixX<FloatType>::Zero(qpoints_r.rows(), 2);
    detJ_r = Eigen::VectorX<FloatType>::Zero(qpoints_r.rows());
    dphi_x = Eigen::MatrixX<FloatType>::Zero(
        2 * qpoints_r.rows(), u_mesh.dofs_per_edge()
    );

    // Extract the indices of the surface edges from the mesh
    std::vector<int> surf_edge_inds = u_mesh.extract_edge_inds(MESH2D::SURFACE_ID);

    // Loop over the surface edges to assemble contributions to the right-hand side vector (rhs_vec)
    for (int si : surf_edge_inds) {
        // Get the node indices for the current edge and extract the node coordinates
        edge_vi = u_mesh.emat(si, Eigen::all);
        node_coords_u = u_mesh.pmat(edge_vi, Eigen::all);

        // Map the edge coordinates to the reference cell (local coordinates)
        FEM1D::map_to_reference_cell(
            u_mesh.degree(), node_coords_u, qpoints_r,
            phi_r, dphi_r, detJ_r, qpoints_x, dphi_x
        );

        // Retrieve the surface normal vector component (nz) for the current edge
        FloatType nz = u_mesh.edge_normals(si, 1);

        // Loop over quadrature points in the edge
        for (int q = 0; q < qpoints_r.rows(); ++q) {
            // Compute the force and accumulation values at the quadrature point
            FloatType fx = force_x(qpoints_x(q, 0), qpoints_x(q, 1));
            FloatType fz = force_z(qpoints_x(q, 0), qpoints_x(q, 1));
            FloatType ac = fssa_accum(qpoints_x(q, 0), qpoints_x(q, 1));

            // Compute the weighted Jacobian determinant (detJ * weight)
            FloatType dFxW = ABS_FUNC(detJ_r(q)) * qweights(q);

            // Loop over edge dofs and update the right-hand side vector (rhs_vec) with the contributions
            for (int i = 0; i < u_mesh.dofs_per_edge(); ++i) {
                rhs_vec(ux_v2d(edge_vi(i))) += fssa_param * phi_r(q, i) * ac * fx * dFxW * nz;
                rhs_vec(uz_v2d(edge_vi(i))) += fssa_param * phi_r(q, i) * ac * fz * dFxW * nz;
            }
        }
    }
}


void pStokesProblem::assemble_rhs_vec()
{
    // Declare matrices and vectors for quadrature points, basis functions, and other required values
    Eigen::MatrixX<FloatType> node_coords_u, qpoints_rs, qpoints_xz,
        phi_rs, dphi_rs, dphi_xz;
    Eigen::VectorX<FloatType> qweights, detJ_rs;
    Eigen::VectorXi element_u;

    // Retrieve quadrature points and weights using Gauss-Legendre quadrature for the mesh cells
    FEM2D::gauss_legendre_quadrature(
        gp_rhs, u_mesh.cell_type(), qpoints_rs, qweights
    );

    // Calculate Lagrange basis functions and their derivatives for the mesh cells
    FEM2D::lagrange_basis(
        u_mesh.degree(), u_mesh.cell_type(), qpoints_rs, phi_rs, dphi_rs
    );

    // Initialize vectors for the determinant of the Jacobian and quadrature points in the local coordinates
    detJ_rs = Eigen::VectorX<FloatType>::Zero(qpoints_rs.rows());
    qpoints_xz = Eigen::MatrixX<FloatType>::Zero(qpoints_rs.rows(), 2);
    dphi_xz = Eigen::MatrixX<FloatType>::Zero(2 * qpoints_rs.rows(), u_mesh.dofs_per_cell());

    // Loop over the mesh elements to assemble the contributions to the right-hand side vector (rhs_vec)
    for (int k = 0; k < u_mesh.cmat.rows(); k++) {
        // Get the node indices for the current element and extract the node coordinates
        element_u = u_mesh.cmat.row(k);
        node_coords_u = u_mesh.pmat(element_u, Eigen::all);

        // Map the element coordinates to the reference cell (local coordinates)
        FEM2D::map_to_reference_cell(
            u_mesh.degree(), u_mesh.cell_type(), node_coords_u, qpoints_rs,
            phi_rs, dphi_rs, detJ_rs, qpoints_xz, dphi_xz
        );

        // Loop over the quadrature points to compute the contributions to rhs_vec
        for (int q = 0; q < qpoints_rs.rows(); ++q) {
            // Compute the weighted Jacobian determinant (detJ * weight)
            FloatType detJxW = ABS_FUNC(detJ_rs(q)) * qweights(q);

            // Loop over the degrees of freedom (DOFs) for the current element and update the right-hand side vector (rhs_vec)
            for (int i = 0; i < u_mesh.dofs_per_cell(); ++i) {
                // Add the contributions for the x-direction force
                rhs_vec(ux_v2d(element_u(i))) += (
                    force_x(qpoints_xz(q, 0), qpoints_xz(q, 1)) * phi_rs(q, i)
                ) * detJxW;

                // Add the contributions for the z-direction force
                rhs_vec(uz_v2d(element_u(i))) += (
                    force_z(qpoints_xz(q, 0), qpoints_xz(q, 1)) * phi_rs(q, i)
                ) * detJxW;
            }
        }
    }
}

void pStokesProblem::commit_lhs_mat()
{
    lhs_mat.setFromTriplets(lhs_coeffs.begin(), lhs_coeffs.end());
}

void pStokesProblem::apply_zero_dirichlet_bc()
{
    // Loop over all nnz elements and set off-diagonals corresponding to boundary nodes to zero
    for (int col = 0; col < lhs_mat.outerSize(); ++col)
        // Check if the column corresponds to a boundary node for either ux or uz 
        if (
            ux_d2v(col) != -1 && u_mesh.dimat(ux_d2v(col)) & ux_dirichlet_bc_mask ||
            uz_d2v(col) != -1 && u_mesh.dimat(uz_d2v(col)) & uz_dirichlet_bc_mask 
        ) {
            // Eliminate column
            for (Eigen::SparseMatrix<FloatType>::InnerIterator it(lhs_mat, col); it; ++it) {
                it.valueRef() = 0.0;
            }
            lhs_mat.coeffRef(col, col) = 1.0;
            rhs_vec(col) = 0.0; // Zero Dirichlet bc
        } else {
            for (Eigen::SparseMatrix<FloatType>::InnerIterator it(lhs_mat, col); it; ++it) {
                int row = it.row();
                // Check if the row corresponds to a boundary node for either ux or uz
                if (
                    ux_d2v(row) != -1 && u_mesh.dimat(ux_d2v(row)) & ux_dirichlet_bc_mask ||
                    uz_d2v(row) != -1 && u_mesh.dimat(uz_d2v(row)) & uz_dirichlet_bc_mask
                ) {
                    // Eliminate row
                    it.valueRef() = 0.0;
                }
            }
        }
    prune_lhs(1e-12);
}

void pStokesProblem::apply_dirichlet_bc(
    const int boundary_part,
    const int velocity_component,
    std::function<FloatType(FloatType, FloatType)> u_func
) {
    // Loop over all nnz elements and set off-diagonals corresponding to boundary nodes to zero
    int *outer_end = lhs_mat.outerIndexPtr() + lhs_mat.rows();
    int col = 0;
    for (
        int *outer_p = lhs_mat.outerIndexPtr();
        outer_p != outer_end;
        outer_p++
    ) {
        int nnz = *(outer_p + 1) - *outer_p;
        for (int i = 0; i < nnz; i++) {
            int row = *(lhs_mat.innerIndexPtr() + *outer_p + i);
            if (velocity_component == HORIZONTAL && ux_d2v(row) != -1) {
                if (u_mesh.dimat(ux_d2v(row)) & boundary_part) {
                    FloatType *val_ptr = lhs_mat.valuePtr() + *outer_p + i;
                    *val_ptr = 0.0;
                }
            }
            else if (velocity_component == VERTICAL && uz_d2v(row) != -1) {
                if (u_mesh.dimat(uz_d2v(row)) & boundary_part) {
                    FloatType *val_ptr = lhs_mat.valuePtr() + *outer_p + i;
                    *val_ptr = 0.0;
                }
            }
        }
        col++;
    }

    // Loop over all nnz elements and set diagonals corresponding to boundary nodes to one
    col = 0;
    for (
        int *outer_p = lhs_mat.outerIndexPtr();
        outer_p != outer_end;
        outer_p++
    ) {
        if (velocity_component == HORIZONTAL && ux_d2v(col) != -1) {
            if (u_mesh.dimat(ux_d2v(col)) & boundary_part) {
                FloatType x = u_mesh.pmat(ux_d2v(col), 0);
                FloatType z = u_mesh.pmat(ux_d2v(col), 1);
                lhs_mat.coeffRef(col, col) = 1.0;
                rhs_vec(col) = u_func(x, z);  // Dirichlet bc
            }
        }
        else if (velocity_component == VERTICAL && uz_d2v(col) != -1) {
            if (u_mesh.dimat(uz_d2v[col]) & boundary_part) {
                FloatType x = u_mesh.pmat(uz_d2v(col), 0);
                FloatType z = u_mesh.pmat(uz_d2v(col), 1);
                lhs_mat.coeffRef(col, col) = 1.0;
                rhs_vec(col) = u_func(x, z);  // Dirichlet bc
            }
        }
        col++;
    }
}

void pStokesProblem::prune_lhs(FloatType threshold)
{
    // NOTE: Pruning zeros is a costly operation and should only be done once.
    lhs_mat.prune(threshold, 1);
}

void pStokesProblem::solve_linear_system()
{
    Eigen::SparseLU<Eigen::SparseMatrix<FloatType>> solver;
    solver.analyzePattern(lhs_mat);
    solver.factorize(lhs_mat);
    w_vec = solver.solve(rhs_vec);
}

void pStokesProblem::solve_nonlinear_system()
{
    // Assemble incompressibility block
    assemble_incomp_block();

    // Assemble FSSA block according to fssa version
    if (fssa_version == FSSA_VERTICAL)
        assemble_fssa_vertical_block();
    else if (fssa_version == FSSA_NORMAL)
        assemble_fssa_normal_block();

    // Assemble the right-hand side vector
    assemble_rhs_vec();
    int nof_pushed_elements_before_stress_assembly = nof_pushed_elements;

    // Initialize error and old/new velocity functions for Picard iteration
    FloatType error_step = 0.0;
    FEMFunction2D ux_old_func = velocity_x();
    FEMFunction2D uz_old_func = velocity_z();
    FEMFunction2D ux_new_func(u_mesh);
    FEMFunction2D uz_new_func(u_mesh);
    FEMFunction2D ex_func(u_mesh);
    FEMFunction2D ez_func(u_mesh);

    // Assemble the mass matrix for calculating the L2 error
    Eigen::SparseMatrix<FloatType> M = FEM2D::assemble_mass_matrix(u_mesh, u_mesh.degree() + 1);
    ux_new_func.set_mass_matrix(M);
    uz_new_func.set_mass_matrix(M);
    ex_func.set_mass_matrix(M);
    ez_func.set_mass_matrix(M);

    // Picard iteration loop
    int ip = 0;
    do {
        // Only need to reassemble stress block between iterations
        lhs_coeffs.resize(nof_pushed_elements_before_stress_assembly);
        nof_pushed_elements = nof_pushed_elements_before_stress_assembly;
        // Assemble stress block and commit
        assemble_stress_block();
        commit_lhs_mat();

        // Apply homogenous Dirichlet BC and solve linear system
        apply_zero_dirichlet_bc();
        solve_linear_system();

        // Compute the step error and check convergence
        ux_new_func.vals = w_vec(ux_v2d);
        uz_new_func.vals = w_vec(uz_v2d);
        ex_func.vals = ux_new_func.vals - ux_old_func.vals;
        ez_func.vals = uz_new_func.vals - uz_old_func.vals;
        FloatType ux = ux_new_func.L2_norm();
        FloatType uz = uz_new_func.L2_norm();
        FloatType ex = ex_func.L2_norm();
        FloatType ez = ez_func.L2_norm();
        error_step = sqrt((ex*ex + ez*ez)/(ux*ux + uz*uz));

        // Log iteration information
        logger.log_msg((boost::format{"<pStokes> Picard iteration %d: r(step) = %g"} %(ip+1) %error_step).str(), TRACE);

        // Update old functions for the next iteration
        ux_old_func.assign(ux_new_func);
        uz_old_func.assign(uz_new_func);
        ++ip;
    } while (error_step > picard_stol && ip < picard_max_iter);

    // Log summary after solving
    logger.log_msg("-------------------------------------", INFO);
    logger.log_msg("********** pStokes Summary **********", INFO);
    logger.log_msg("-------------------------------------", INFO);
    logger.log_msg((boost::format{"Picard solver finished in %d iterations"} %ip).str(), INFO);
    logger.log_msg((boost::format{"r (step): %.4g"} % (error_step)).str(), INFO);
    logger.log_msg((boost::format{"min ux, max ux: %.4g, %.4g m/yr"} % (1e3*w_vec(ux_v2d).minCoeff()) % (1e3*w_vec(ux_v2d).maxCoeff())).str(), INFO);
    logger.log_msg((boost::format{"min uz, max uz: %.4g, %.4g m/yr"} % (1e3*w_vec(uz_v2d).minCoeff()) % (1e3*w_vec(uz_v2d).maxCoeff())).str(), INFO);
    logger.log_msg((boost::format{"min p, max p: %.4g, %.4g MPa"} % (w_vec(p_v2d).minCoeff()) % (w_vec(p_v2d).maxCoeff())).str(), INFO);
    logger.log_msg("-------------------------------------", INFO);
}

FEMFunction2D pStokesProblem::velocity_x()
{
    FEMFunction2D ux_fem_func(u_mesh);
    ux_fem_func.vals = w_vec(ux_v2d);
    return ux_fem_func;
}

FEMFunction2D pStokesProblem::velocity_z()
{
    FEMFunction2D uz_fem_func(u_mesh);
    uz_fem_func.vals = w_vec(uz_v2d);
    return uz_fem_func;
}

FEMFunction2D pStokesProblem::pressure()
{
    FEMFunction2D p_fem_func(p_mesh);
    p_fem_func.vals = w_vec(p_v2d);
    return p_fem_func;
}

void pStokesProblem::resize_lhs(int size)
{
    if (size < 0)
        throw std::invalid_argument("size can not be negative");

    lhs_coeffs.resize(size);
}

void pStokesProblem::reset_lhs()
{
    lhs_coeffs.clear();
    nof_pushed_elements = 0;
}

void pStokesProblem::reset_rhs()
{
    rhs_vec.setZero();
}

void pStokesProblem::reset_system()
{
    reset_lhs();
    reset_rhs();
}

int pStokesProblem::log_level()
{
    return logger.log_level();
}

void pStokesProblem::set_log_level(int value)
{
    logger.set_log_level(value);
}
