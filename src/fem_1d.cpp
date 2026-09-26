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
#include <fem_1d.hpp>

void FEM1D::map_to_reference_cell(
    int degree, 
    Eigen::MatrixX<FloatType> &node_coords,
    Eigen::VectorX<FloatType> &qpoints_r,
    Eigen::MatrixX<FloatType> &phi_r,
    Eigen::MatrixX<FloatType> &grad_phi_r,
    Eigen::VectorX<FloatType> &detJ_r_ret,
    Eigen::MatrixX<FloatType> &qpoints_x_ret,
    Eigen::MatrixX<FloatType> &grad_phi_x_ret
) {
    Eigen::VectorX<FloatType> dF;

    for (int i = 0; i < qpoints_r.rows(); i++) {
        // Compute the determinant of the Jacobian (dF.norm() gives the norm of the gradient vector)
        dF = grad_phi_r(i, Eigen::all) * node_coords;
        detJ_r_ret(i) = dF.norm();  // Store the determinant of Jacobian at the i-th quadrature point

        // Compute the physical quadrature points by multiplying shape functions with node coordinates
        qpoints_x_ret(i, Eigen::all) = phi_r(i, Eigen::all) * node_coords;

        // Compute the gradient of shape functions in physical space using the Jacobian determinant
        grad_phi_x_ret(i, Eigen::all) = 1.0 / detJ_r_ret(i) * grad_phi_r(i, Eigen::all);
    }
}

void FEM1D::lagrange_basis(
    int degree,
    Eigen::VectorX<FloatType> &qpoints_r,
    Eigen::MatrixX<FloatType> &phi_r_ret,
    Eigen::MatrixX<FloatType> &grad_phi_r_ret
) {
    phi_r_ret = Eigen::MatrixX<FloatType>::Zero(qpoints_r.rows(), degree+1);
    grad_phi_r_ret = Eigen::MatrixX<FloatType>::Zero(qpoints_r.rows(), degree+1);

    for (int i = 0; i < qpoints_r.rows(); i++) {
        FloatType r = qpoints_r(i);
        if (degree == 1) {
            phi_r_ret(i, 0) = (1-r);
            phi_r_ret(i, 1) = r;

            grad_phi_r_ret(i, 0) = -1;
            grad_phi_r_ret(i, 1) = 1;
        } else if (degree == 2) {
            phi_r_ret(i, 0) = 1 - 3*r + 2*r*r;
            phi_r_ret(i, 1) = 4*r - 4*r*r;
            phi_r_ret(i, 2) = -r + 2*r*r;

            grad_phi_r_ret(i, 0) = -3 + 4*r;
            grad_phi_r_ret(i, 1) = 4 - 8*r;
            grad_phi_r_ret(i, 2) = -1 + 4*r;
        }
    }
}

Eigen::MatrixX<FloatType> FEM1D::reference_element_points_rs(
    const int degree
) {
    IntervalMesh mesh(0.0, 1.0, 1, degree);	
    Eigen::MatrixX<FloatType> ref_coords = mesh.pmat(mesh.cmat.row(0), Eigen::all);
    return ref_coords;
}

void FEM1D::gauss_legendre_quadrature(
    const int precision,
    Eigen::VectorX<FloatType> &points,
    Eigen::VectorX<FloatType> &weights
) {
    points = Eigen::VectorX<FloatType>::Zero(precision);
    weights = Eigen::VectorX<FloatType>::Zero(precision);

    if (precision == 1) {
        points <<
            0.500000000000000000000000L;
        weights <<
            1.000000000000000000000000L;
    } else if (precision == 2) {
        points <<
            0.211324865405187117745426L,
            0.788675134594812882254574L;
        weights <<
            0.500000000000000000000000L,
            0.500000000000000000000000L;
    } else if (precision == 3) {
        points <<
            0.112701665379258311482073L,
            0.500000000000000000000000L,
            0.887298334620741688517927L;
        weights <<
            0.277777777777777777777778L,
            0.444444444444444444444444L,
            0.277777777777777777777778L;
    } else if (precision == 4) {
        points <<
            0.069431844202973712388027L,
            0.330009478207571867598667L,
            0.669990521792428132401333L,
            0.930568155797026287611973L;
        weights <<
            0.173927422568726928686532L,
            0.326072577431273071313468L,
            0.326072577431273071313468L,
            0.173927422568726928686532L;
    } else if (precision == 5) {
        points <<
            0.046910077030668003601187L,
            0.230765344947158454481843L,
            0.500000000000000000000000L,
            0.769234655052841545518157L,
            0.953089922969331996398813L;
        weights <<
            0.118463442528094543757132L,
            0.239314335249683234020646L,
            0.284444444444444444444444L,
            0.239314335249683234020646L,
            0.118463442528094543757132L;
    } else {
        throw std::invalid_argument("FEM1D::reference_element_points_rs: precision > 5 is too damn high");
    }
}

Eigen::SparseMatrix<FloatType> FEM1D::assemble_mass_matrix(IntervalMesh &mesh, int gp) {
    // Local variables for node coordinates, quadrature points, shape functions, and Jacobian determinants
    Eigen::MatrixX<FloatType> node_coords, qpoints_x, phi_r, dphi_r, dphi_x;
    Eigen::VectorX<FloatType> qweights, qpoints_r, detJ_r;
    Eigen::VectorXi element;
    std::vector<Eigen::Triplet<FloatType>> mat_coeffs;

    // Initialize the sparse matrix to store the mass matrix
    Eigen::SparseMatrix<FloatType> M_sp_ret = Eigen::SparseMatrix<FloatType>(
        mesh.nof_dofs(), mesh.nof_dofs()
    );

    // Compute Gauss-Legendre quadrature points and weights
    FEM1D::gauss_legendre_quadrature(gp, qpoints_r, qweights);
    // Compute Lagrange basis functions and their derivatives at quadrature points
    FEM1D::lagrange_basis(mesh.degree(), qpoints_r, phi_r, dphi_r);

    // Initialize arrays to store quadrature points in physical space and derivatives
    qpoints_x = Eigen::MatrixX<FloatType>::Zero(qpoints_r.rows(), 2);
    detJ_r = Eigen::VectorX<FloatType>::Zero(qpoints_r.rows());
    dphi_x = Eigen::MatrixX<FloatType>::Zero(
        qpoints_r.rows(), mesh.dofs_per_cell()
    );

    // Block matrix for local mass matrix contributions
    Eigen::MatrixX<FloatType> M_block = Eigen::MatrixX<FloatType>::Zero(
        mesh.dofs_per_cell(), mesh.dofs_per_cell()
    );

    // Loop over each element of the mesh
    for (int ci = 0; ci < mesh.nof_cells(); ci++) {
        element = mesh.cmat(ci, Eigen::all);  // Element connectivity
        node_coords = mesh.pmat(element, Eigen::all);  // Node coordinates for the current element

        // Map quadrature points from reference cell to physical cell
        FEM1D::map_to_reference_cell(
            mesh.degree(), node_coords, qpoints_r,
            phi_r, dphi_r, detJ_r, qpoints_x, dphi_x
        );

        // Loop over the quadrature points to calculate the local mass matrix
        for (int q = 0; q < qpoints_r.rows(); q++) {
            FloatType detJxW = qweights(q) * detJ_r(q);  // Weighted Jacobian determinant
            for (int i = 0; i < mesh.dofs_per_cell(); i++) {
                for (int j = 0; j < mesh.dofs_per_cell(); j++) {
                    M_block(i, j) += phi_r(q, i) * phi_r(q, j) * detJxW;  // Assemble mass matrix block
                }
            }
        }

        // Add local contributions to the global sparse matrix
        for (int i = 0; i < mesh.dofs_per_cell(); i++) {
            for (int j = 0; j < mesh.dofs_per_cell(); j++) {
                mat_coeffs.push_back(Eigen::Triplet<FloatType>(
                    element(i),  ///< Global row index
                    element(j),  ///< Global column index
                    M_block(i, j)  ///< Matrix entry value
                ));
            }
        }

        // Reset the block matrix for the next element
        M_block.setZero();
    }

    // Create the sparse matrix from the list of triplets
    M_sp_ret.setFromTriplets(mat_coeffs.begin(), mat_coeffs.end());
    return M_sp_ret;
}

FloatType FEM1D::inner(
    const Eigen::VectorX<FloatType> &u_vec,
    const Eigen::SparseMatrix<FloatType> &M,
    const Eigen::VectorX<FloatType> &v_vec
) {
    return u_vec.dot(M*v_vec);
}
