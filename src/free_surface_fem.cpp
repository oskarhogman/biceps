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
#include <free_surface_fem.hpp>
#include <fem_1d.hpp>

FreeSurfaceProblem::FreeSurfaceProblem(
    IntervalMesh &h_mesh, IntervalMesh &u_mesh
) : h_mesh(h_mesh), u_mesh(u_mesh)
{
    // Initialize the left-hand side matrix and right-hand side vector.
    lhs_mat = Eigen::SparseMatrix<FloatType>(
        h_mesh.nof_dofs(), h_mesh.nof_dofs()
    );
    rhs_vec = Eigen::VectorX<FloatType>::Zero(h_mesh.nof_dofs());
    zs_vec = h_mesh.pmat(Eigen::all, 1);
    lhs_coeffs.reserve((2*h_mesh.degree()+1)*h_mesh.nof_dofs());
}

void FreeSurfaceProblem::assemble_lhs_explicit()
{
    // Local variables for storing matrix and vector data during assembly.
    Eigen::MatrixX<FloatType> node_coords, qpoints_x, phi_r, dphi_r, dphi_x;
    Eigen::VectorX<FloatType> qweights, qpoints_r, detJ_r;
    Eigen::VectorXi element;

    // Perform Gauss-Legendre quadrature for the given precision.
    FEM1D::gauss_legendre_quadrature(gp_lhs, qpoints_r, qweights);

    // Generate Lagrange basis functions for the finite element degree.
    FEM1D::lagrange_basis(
        h_mesh.degree(), qpoints_r, phi_r, dphi_r
    );

    // Initialize matrices and vectors for mapping and calculation.
    qpoints_x = Eigen::MatrixX<FloatType>::Zero(qpoints_r.rows(), 2);
    detJ_r = Eigen::VectorX<FloatType>::Zero(qpoints_r.rows());
    dphi_x = Eigen::MatrixX<FloatType>::Zero(
        qpoints_r.rows(), h_mesh.dofs_per_cell()
    );

    // Initialize the matrix for each element.
    Eigen::MatrixX<FloatType> A = Eigen::MatrixX<FloatType>::Zero(
        h_mesh.dofs_per_cell(), h_mesh.dofs_per_cell()
    );

    // Loop over cells and compute the contribution to the LHS matrix.
    for (int ci = 0; ci < h_mesh.nof_cells(); ++ci) {
        element = h_mesh.cmat(ci, Eigen::all);
        node_coords = h_mesh.pmat(element, Eigen::all);
        FEM1D::map_to_reference_cell(
            h_mesh.degree(), node_coords, qpoints_r,
            phi_r, dphi_r, detJ_r, qpoints_x, dphi_x
        );

        // Integrate over quadrature points to compute the matrix A.
        for (int q = 0; q < qpoints_r.rows(); ++q) {
            FloatType detJxW = qweights(q)*detJ_r(q);
            for (int i = 0; i < h_mesh.dofs_per_cell(); ++i) {
                for (int j = 0; j < h_mesh.dofs_per_cell(); ++j) {
                    A(i, j) += phi_r(q, i)*phi_r(q, j)*detJxW;
                }
            }
        }

        // Store the computed coefficients into the triplet list.
        for (int i = 0; i < h_mesh.dofs_per_cell(); ++i) {
            for (int j = 0; j < h_mesh.dofs_per_cell(); ++j) {
                lhs_coeffs.push_back(Eigen::Triplet<FloatType>(
                    element(i),
                    element(j),
                    A(i, j)
                ));
            }
        }
        A.setZero();
    }
    nof_pushed_elements = lhs_coeffs.size();
}

void FreeSurfaceProblem::assemble_rhs_CN(
    FEMFunction1D &h0_fem_func,
    FEMFunction1D &ux_fem_func,
    FEMFunction1D &uz_fem_func,
    FEMFunction1D &ac_fem_func,
    FEMFunction1D &ux_fem_funcOld,
    FEMFunction1D &uz_fem_funcOld,
    FEMFunction1D &ac_fem_funcOld,
    FloatType dt
)
{
    // Local variables for storing matrix and vector data during assembly.
    Eigen::MatrixX<FloatType> node_coords_h, node_coords_u, qpoints_x, h_phi_r, h_dphi_r,
        h_dphi_x, u_phi_r, u_dphi_r, u_dphi_x;
    Eigen::VectorX<FloatType> qweights, qpoints_r, detJ_r;
    Eigen::VectorXi element_h, element_u;

    // Perform Gauss-Legendre quadrature for the given precision.
    FEM1D::gauss_legendre_quadrature(gp_rhs, qpoints_r, qweights);

    // Generate Lagrange basis functions for the finite element degree.
    FEM1D::lagrange_basis(
        h_mesh.degree(), qpoints_r, h_phi_r, h_dphi_r
    );
    FEM1D::lagrange_basis(
        u_mesh.degree(), qpoints_r, u_phi_r, u_dphi_r
    );

    // Initialize matrices and vectors for mapping and calculation.
    qpoints_x = Eigen::MatrixX<FloatType>::Zero(qpoints_r.rows(), 2);
    detJ_r = Eigen::VectorX<FloatType>::Zero(qpoints_r.rows());
    h_dphi_x = Eigen::MatrixX<FloatType>::Zero(
        qpoints_r.rows(), h_mesh.dofs_per_cell()
    );
    u_dphi_x = Eigen::MatrixX<FloatType>::Zero(
        qpoints_r.rows(), u_mesh.dofs_per_cell()
    );

    // Loop over cells to compute the contribution to the RHS vector.
    for (int ci = 0; ci < h_mesh.nof_cells(); ++ci) {
        element_h = h_mesh.cmat(ci, Eigen::all);
        element_u = u_mesh.cmat(ci, Eigen::all);
        node_coords_h = h_mesh.pmat(element_h, Eigen::all);
        node_coords_u = u_mesh.pmat(element_u, Eigen::all);
        FEM1D::map_to_reference_cell(
            u_mesh.degree(), node_coords_u, qpoints_r,
            u_phi_r, u_dphi_r, detJ_r, qpoints_x, u_dphi_x
        );
        FEM1D::map_to_reference_cell(
            h_mesh.degree(), node_coords_h, qpoints_r,
            h_phi_r, h_dphi_r, detJ_r, qpoints_x, h_dphi_x
        );

        // Evaluate the functions at the quadrature points.
        Eigen::VectorX<FloatType> h0_vec = h_phi_r*h0_fem_func.eval_cell(ci);
        Eigen::VectorX<FloatType> dh_vec = h_dphi_x*h0_fem_func.eval_cell(ci);
        Eigen::VectorX<FloatType> ux_vec = u_phi_r*ux_fem_func.eval_cell(ci);
        Eigen::VectorX<FloatType> uz_vec = u_phi_r*uz_fem_func.eval_cell(ci);
        Eigen::VectorX<FloatType> ac_vec = h_phi_r*ac_fem_func.eval_cell(ci);
        Eigen::VectorX<FloatType> ux_vecOld = u_phi_r*ux_fem_funcOld.eval_cell(ci);
        Eigen::VectorX<FloatType> uz_vecOld = u_phi_r*uz_fem_funcOld.eval_cell(ci);
        Eigen::VectorX<FloatType> ac_vecOld = h_phi_r*ac_fem_funcOld.eval_cell(ci);


        // Integrate over quadrature points to compute the RHS contributions.
        for (int q = 0; q < qpoints_r.rows(); ++q) {
            FloatType detJxW = qweights(q)*detJ_r(q);
            for (int i = 0; i < h_mesh.dofs_per_cell(); ++i) {
                rhs_vec(element_h(i)) += h_phi_r(q, i)*(
                    h0_vec(q) + dt*(0.5*(uz_vec(q)+uz_vecOld(q)) - 0.5*ux_vecOld(q)*dh_vec(q) + 0.5*(ac_vec(q)+ac_vecOld(q)))
                )*detJxW;
            }
        }
    }
}


void FreeSurfaceProblem::assemble_rhs_crankish(
    FEMFunction1D &h0_fem_func,
    FEMFunction1D &ux_fem_func,
    FEMFunction1D &uz_fem_func,
    FEMFunction1D &ac_fem_func,
    FloatType dt
)
{
    // Local variables for storing matrix and vector data during assembly.
    Eigen::MatrixX<FloatType> node_coords_h, node_coords_u, qpoints_x, h_phi_r, h_dphi_r,
        h_dphi_x, u_phi_r, u_dphi_r, u_dphi_x;
    Eigen::VectorX<FloatType> qweights, qpoints_r, detJ_r;
    Eigen::VectorXi element_h, element_u;

    // Perform Gauss-Legendre quadrature for the given precision.
    FEM1D::gauss_legendre_quadrature(gp_rhs, qpoints_r, qweights);

    // Generate Lagrange basis functions for the finite element degree.
    FEM1D::lagrange_basis(
        h_mesh.degree(), qpoints_r, h_phi_r, h_dphi_r
    );
    FEM1D::lagrange_basis(
        u_mesh.degree(), qpoints_r, u_phi_r, u_dphi_r
    );

    // Initialize matrices and vectors for mapping and calculation.
    qpoints_x = Eigen::MatrixX<FloatType>::Zero(qpoints_r.rows(), 2);
    detJ_r = Eigen::VectorX<FloatType>::Zero(qpoints_r.rows());
    h_dphi_x = Eigen::MatrixX<FloatType>::Zero(
        qpoints_r.rows(), h_mesh.dofs_per_cell()
    );
    u_dphi_x = Eigen::MatrixX<FloatType>::Zero(
        qpoints_r.rows(), u_mesh.dofs_per_cell()
    );

    // Loop over cells to compute the contribution to the RHS vector.
    for (int ci = 0; ci < h_mesh.nof_cells(); ++ci) {
        element_h = h_mesh.cmat(ci, Eigen::all);
        element_u = u_mesh.cmat(ci, Eigen::all);
        node_coords_h = h_mesh.pmat(element_h, Eigen::all);
        node_coords_u = u_mesh.pmat(element_u, Eigen::all);
        FEM1D::map_to_reference_cell(
            u_mesh.degree(), node_coords_u, qpoints_r,
            u_phi_r, u_dphi_r, detJ_r, qpoints_x, u_dphi_x
        );
        FEM1D::map_to_reference_cell(
            h_mesh.degree(), node_coords_h, qpoints_r,
            h_phi_r, h_dphi_r, detJ_r, qpoints_x, h_dphi_x
        );

        // Evaluate the functions at the quadrature points.
        Eigen::VectorX<FloatType> h0_vec = h_phi_r*h0_fem_func.eval_cell(ci);
        Eigen::VectorX<FloatType> dh_vec = h_dphi_x*h0_fem_func.eval_cell(ci);
        Eigen::VectorX<FloatType> ux_vec = u_phi_r*ux_fem_func.eval_cell(ci);
        Eigen::VectorX<FloatType> uz_vec = u_phi_r*uz_fem_func.eval_cell(ci);
        Eigen::VectorX<FloatType> ac_vec = h_phi_r*ac_fem_func.eval_cell(ci);

        // Integrate over quadrature points to compute the RHS contributions.
        for (int q = 0; q < qpoints_r.rows(); ++q) {
            FloatType detJxW = qweights(q)*detJ_r(q);
            for (int i = 0; i < h_mesh.dofs_per_cell(); ++i) {
                rhs_vec(element_h(i)) += h_phi_r(q, i)*(
                    h0_vec(q) + dt*(uz_vec(q) - 0.5*ux_vec(q)*dh_vec(q) + ac_vec(q))
                )*detJxW;
            }
        }
    }
}

void FreeSurfaceProblem::assemble_lhs_crankish(
    FEMFunction1D &ux_fem_func, FloatType dt
) {
    // Local variables for storing matrix and vector data during assembly.
    Eigen::MatrixX<FloatType> node_coords_h, node_coords_u, qpoints_x, h_phi_r, h_dphi_r,
        h_dphi_x, u_phi_r, u_dphi_r, u_dphi_x;
    Eigen::VectorX<FloatType> qweights, qpoints_r, detJ_r;
    Eigen::VectorXi element_h, element_u;

    // Perform Gauss-Legendre quadrature for the given precision.
    FEM1D::gauss_legendre_quadrature(gp_lhs, qpoints_r, qweights);

    // Generate Lagrange basis functions for the finite element degree.
    FEM1D::lagrange_basis(
        h_mesh.degree(), qpoints_r, h_phi_r, h_dphi_r
    );
    FEM1D::lagrange_basis(
        u_mesh.degree(), qpoints_r, u_phi_r, u_dphi_r
    );

    // Initialize matrices and vectors for mapping and calculation.
    qpoints_x = Eigen::MatrixX<FloatType>::Zero(qpoints_r.rows(), 2);
    detJ_r = Eigen::VectorX<FloatType>::Zero(qpoints_r.rows());
    h_dphi_x = Eigen::MatrixX<FloatType>::Zero(
        qpoints_r.rows(), h_mesh.dofs_per_cell()
    );
    u_dphi_x = Eigen::MatrixX<FloatType>::Zero(
        qpoints_r.rows(), u_mesh.dofs_per_cell()
    );

    // Initialize the matrix for each element.
    Eigen::MatrixX<FloatType> A = Eigen::MatrixX<FloatType>::Zero(
        h_mesh.dofs_per_cell(), h_mesh.dofs_per_cell()
    );

    // Loop over cells to compute the implicit contribution to the LHS matrix.
    for (int ci = 0; ci < h_mesh.nof_cells(); ++ci) {
        element_h = h_mesh.cmat(ci, Eigen::all);
        element_u = u_mesh.cmat(ci, Eigen::all);
        node_coords_h = h_mesh.pmat(element_h, Eigen::all);
        node_coords_u = u_mesh.pmat(element_u, Eigen::all);
        FEM1D::map_to_reference_cell(
            u_mesh.degree(), node_coords_u, qpoints_r,
            u_phi_r, u_dphi_r, detJ_r, qpoints_x, u_dphi_x
        );
        FEM1D::map_to_reference_cell(
            h_mesh.degree(), node_coords_h, qpoints_r,
            h_phi_r, h_dphi_r, detJ_r, qpoints_x, h_dphi_x
        );

        // Evaluate the velocity function at the quadrature points.
        Eigen::VectorX<FloatType> ux_vec = u_phi_r*ux_fem_func.eval_cell(ci);

        // Integrate over quadrature points to compute the LHS matrix.
        for (int q = 0; q < qpoints_r.rows(); ++q) {
            FloatType detJxW = qweights(q)*detJ_r(q);
            for (int i = 0; i < h_mesh.dofs_per_cell(); ++i) {
                for (int j = 0; j < h_mesh.dofs_per_cell(); ++j) {
                    A(i, j) += (
                        h_phi_r(q, i)*h_phi_r(q, j) +
                        0.5*dt*ux_vec(q)*h_phi_r(q, i)*h_dphi_x(q, j)
                    )*detJxW;
                }
            }
        }

        // Store the computed coefficients into the triplet list.
        for (int i = 0; i < h_mesh.dofs_per_cell(); ++i) {
            for (int j = 0; j < h_mesh.dofs_per_cell(); ++j) {
                lhs_coeffs.push_back(Eigen::Triplet<FloatType>(
                    element_h(i),
                    element_h(j),
                    A(i, j)
                ));
            }
        }
        A.setZero();
    }
    nof_pushed_elements = lhs_coeffs.size();
}


void FreeSurfaceProblem::assemble_rhs_explicit(
    FEMFunction1D &h0_fem_func,
    FEMFunction1D &ux_fem_func,
    FEMFunction1D &uz_fem_func,
    FEMFunction1D &ac_fem_func,
    FloatType dt
)
{
    // Local variables for storing matrix and vector data during assembly.
    Eigen::MatrixX<FloatType> node_coords_h, node_coords_u, qpoints_x, h_phi_r, h_dphi_r,
        h_dphi_x, u_phi_r, u_dphi_r, u_dphi_x;
    Eigen::VectorX<FloatType> qweights, qpoints_r, detJ_r;
    Eigen::VectorXi element_h, element_u;

    // Perform Gauss-Legendre quadrature for the given precision.
    FEM1D::gauss_legendre_quadrature(gp_rhs, qpoints_r, qweights);

    // Generate Lagrange basis functions for the finite element degree.
    FEM1D::lagrange_basis(
        h_mesh.degree(), qpoints_r, h_phi_r, h_dphi_r
    );
    FEM1D::lagrange_basis(
        u_mesh.degree(), qpoints_r, u_phi_r, u_dphi_r
    );

    // Initialize matrices and vectors for mapping and calculation.
    qpoints_x = Eigen::MatrixX<FloatType>::Zero(qpoints_r.rows(), 2);
    detJ_r = Eigen::VectorX<FloatType>::Zero(qpoints_r.rows());
    h_dphi_x = Eigen::MatrixX<FloatType>::Zero(
        qpoints_r.rows(), h_mesh.dofs_per_cell()
    );
    u_dphi_x = Eigen::MatrixX<FloatType>::Zero(
        qpoints_r.rows(), u_mesh.dofs_per_cell()
    );

    // Loop over cells to compute the contribution to the RHS vector.
    for (int ci = 0; ci < h_mesh.nof_cells(); ++ci) {
        element_h = h_mesh.cmat(ci, Eigen::all);
        element_u = u_mesh.cmat(ci, Eigen::all);
        node_coords_h = h_mesh.pmat(element_h, Eigen::all);
        node_coords_u = u_mesh.pmat(element_u, Eigen::all);
        FEM1D::map_to_reference_cell(
            u_mesh.degree(), node_coords_u, qpoints_r,
            u_phi_r, u_dphi_r, detJ_r, qpoints_x, u_dphi_x
        );
        FEM1D::map_to_reference_cell(
            h_mesh.degree(), node_coords_h, qpoints_r,
            h_phi_r, h_dphi_r, detJ_r, qpoints_x, h_dphi_x
        );

        // Evaluate the functions at the quadrature points.
        Eigen::VectorX<FloatType> h0_vec = h_phi_r*h0_fem_func.eval_cell(ci);
        Eigen::VectorX<FloatType> dh_vec = h_dphi_x*h0_fem_func.eval_cell(ci);
        Eigen::VectorX<FloatType> ux_vec = u_phi_r*ux_fem_func.eval_cell(ci);
        Eigen::VectorX<FloatType> uz_vec = u_phi_r*uz_fem_func.eval_cell(ci);
        Eigen::VectorX<FloatType> ac_vec = h_phi_r*ac_fem_func.eval_cell(ci);

        // Integrate over quadrature points to compute the RHS contributions.
        for (int q = 0; q < qpoints_r.rows(); ++q) {
            FloatType detJxW = qweights(q)*detJ_r(q);
            for (int i = 0; i < h_mesh.dofs_per_cell(); ++i) {
                rhs_vec(element_h(i)) += h_phi_r(q, i)*(
                    h0_vec(q) + dt*(uz_vec(q) - ux_vec(q)*dh_vec(q) + ac_vec(q))
                )*detJxW;
            }
        }
    }
}

void FreeSurfaceProblem::assemble_lhs_BDF2(
    FEMFunction1D &ux_fem_func, FloatType dt
) {
    // Local variables for storing matrix and vector data during assembly.
    Eigen::MatrixX<FloatType> node_coords_h, node_coords_u, qpoints_x, h_phi_r, h_dphi_r,
        h_dphi_x, u_phi_r, u_dphi_r, u_dphi_x;
    Eigen::VectorX<FloatType> qweights, qpoints_r, detJ_r;
    Eigen::VectorXi element_h, element_u;

    // Perform Gauss-Legendre quadrature for the given precision.
    FEM1D::gauss_legendre_quadrature(gp_lhs, qpoints_r, qweights);

    // Generate Lagrange basis functions for the finite element degree.
    FEM1D::lagrange_basis(
        h_mesh.degree(), qpoints_r, h_phi_r, h_dphi_r
    );
    FEM1D::lagrange_basis(
        u_mesh.degree(), qpoints_r, u_phi_r, u_dphi_r
    );

    // Initialize matrices and vectors for mapping and calculation.
    qpoints_x = Eigen::MatrixX<FloatType>::Zero(qpoints_r.rows(), 2);
    detJ_r = Eigen::VectorX<FloatType>::Zero(qpoints_r.rows());
    h_dphi_x = Eigen::MatrixX<FloatType>::Zero(
        qpoints_r.rows(), h_mesh.dofs_per_cell()
    );
    u_dphi_x = Eigen::MatrixX<FloatType>::Zero(
        qpoints_r.rows(), u_mesh.dofs_per_cell()
    );

    // Initialize the matrix for each element.
    Eigen::MatrixX<FloatType> A = Eigen::MatrixX<FloatType>::Zero(
        h_mesh.dofs_per_cell(), h_mesh.dofs_per_cell()
    );

    // Loop over cells to compute the implicit contribution to the LHS matrix.
    for (int ci = 0; ci < h_mesh.nof_cells(); ++ci) {
        element_h = h_mesh.cmat(ci, Eigen::all);
        element_u = u_mesh.cmat(ci, Eigen::all);
        node_coords_h = h_mesh.pmat(element_h, Eigen::all);
        node_coords_u = u_mesh.pmat(element_u, Eigen::all);
        FEM1D::map_to_reference_cell(
            u_mesh.degree(), node_coords_u, qpoints_r,
            u_phi_r, u_dphi_r, detJ_r, qpoints_x, u_dphi_x
        );
        FEM1D::map_to_reference_cell(
            h_mesh.degree(), node_coords_h, qpoints_r,
            h_phi_r, h_dphi_r, detJ_r, qpoints_x, h_dphi_x
        );

        // Evaluate the velocity function at the quadrature points.
        Eigen::VectorX<FloatType> ux_vec = u_phi_r*ux_fem_func.eval_cell(ci);

        // Integrate over quadrature points to compute the LHS matrix.
        for (int q = 0; q < qpoints_r.rows(); ++q) {
            FloatType detJxW = qweights(q)*detJ_r(q);
            for (int i = 0; i < h_mesh.dofs_per_cell(); ++i) {
                for (int j = 0; j < h_mesh.dofs_per_cell(); ++j) {
                    A(i, j) += (
                        h_phi_r(q, i)*h_phi_r(q, j) +
                        (2.0/3.0)*dt*ux_vec(q)*h_phi_r(q, i)*h_dphi_x(q, j)
                    )*detJxW;
                }
            }
        }

        // Store the computed coefficients into the triplet list.
        for (int i = 0; i < h_mesh.dofs_per_cell(); ++i) {
            for (int j = 0; j < h_mesh.dofs_per_cell(); ++j) {
                lhs_coeffs.push_back(Eigen::Triplet<FloatType>(
                    element_h(i),
                    element_h(j),
                    A(i, j)
                ));
            }
        }
        A.setZero();
    }
    nof_pushed_elements = lhs_coeffs.size();
}

void FreeSurfaceProblem::assemble_rhs_BDF2(
    FEMFunction1D &h0_fem_func,
    FEMFunction1D &ux_fem_func,
    FEMFunction1D &uz_fem_func,
    FEMFunction1D &ac_fem_func,
    FEMFunction1D &h0_fem_funcOld,
    FloatType dt
)
{
    // Local variables for storing matrix and vector data during assembly.
    Eigen::MatrixX<FloatType> node_coords_h, node_coords_u, qpoints_x, h_phi_r, h_dphi_r,
        h_dphi_x, u_phi_r, u_dphi_r, u_dphi_x;
    Eigen::VectorX<FloatType> qweights, qpoints_r, detJ_r;
    Eigen::VectorXi element_h, element_u;

    // Perform Gauss-Legendre quadrature for the given precision.
    FEM1D::gauss_legendre_quadrature(gp_rhs, qpoints_r, qweights);

    // Generate Lagrange basis functions for the finite element degree.
    FEM1D::lagrange_basis(
        h_mesh.degree(), qpoints_r, h_phi_r, h_dphi_r
    );
    FEM1D::lagrange_basis(
        u_mesh.degree(), qpoints_r, u_phi_r, u_dphi_r
    );

    // Initialize matrices and vectors for mapping and calculation.
    qpoints_x = Eigen::MatrixX<FloatType>::Zero(qpoints_r.rows(), 2);
    detJ_r = Eigen::VectorX<FloatType>::Zero(qpoints_r.rows());
    h_dphi_x = Eigen::MatrixX<FloatType>::Zero(
        qpoints_r.rows(), h_mesh.dofs_per_cell()
    );
    u_dphi_x = Eigen::MatrixX<FloatType>::Zero(
        qpoints_r.rows(), u_mesh.dofs_per_cell()
    );

    // Loop over cells to compute the contribution to the RHS vector.
    for (int ci = 0; ci < h_mesh.nof_cells(); ++ci) {
        element_h = h_mesh.cmat(ci, Eigen::all);
        element_u = u_mesh.cmat(ci, Eigen::all);
        node_coords_h = h_mesh.pmat(element_h, Eigen::all);
        node_coords_u = u_mesh.pmat(element_u, Eigen::all);
        FEM1D::map_to_reference_cell(
            u_mesh.degree(), node_coords_u, qpoints_r,
            u_phi_r, u_dphi_r, detJ_r, qpoints_x, u_dphi_x
        );
        FEM1D::map_to_reference_cell(
            h_mesh.degree(), node_coords_h, qpoints_r,
            h_phi_r, h_dphi_r, detJ_r, qpoints_x, h_dphi_x
        );

        // Evaluate the functions at the quadrature points.
        Eigen::VectorX<FloatType> h0_vec = h_phi_r*h0_fem_func.eval_cell(ci);
        Eigen::VectorX<FloatType> dh_vec = h_dphi_x*h0_fem_func.eval_cell(ci);
        Eigen::VectorX<FloatType> ux_vec = u_phi_r*ux_fem_func.eval_cell(ci);
        Eigen::VectorX<FloatType> uz_vec = u_phi_r*uz_fem_func.eval_cell(ci);
        Eigen::VectorX<FloatType> ac_vec = h_phi_r*ac_fem_func.eval_cell(ci);
        Eigen::VectorX<FloatType> h0_vecOld = h_phi_r*h0_fem_funcOld.eval_cell(ci);


        // Integrate over quadrature points to compute the RHS contributions.
        for (int q = 0; q < qpoints_r.rows(); ++q) {
            FloatType detJxW = qweights(q)*detJ_r(q);
            for (int i = 0; i < h_mesh.dofs_per_cell(); ++i) {
                rhs_vec(element_h(i)) += h_phi_r(q, i)*(
                    (4.0/3.0)*h0_vec(q)-(1.0/3.0)*h0_vecOld(q) + (2.0/3.0)*dt*(uz_vec(q) + ac_vec(q))
                )*detJxW;

                
            }
        }
    }
}

void FreeSurfaceProblem::assemble_rhs_Multistep(
    FEMFunction1D &h0_fem_func,
    FEMFunction1D &ux_fem_func,
    FEMFunction1D &uz_fem_func,
    FEMFunction1D &ac_fem_func,
    FEMFunction1D &h0_fem_funcOld,
    FEMFunction1D &ux_fem_funcOld,
    FEMFunction1D &uz_fem_funcOld,
    FEMFunction1D &ac_fem_funcOld,
    FloatType weightk, 
    FloatType weightkm1,
    FloatType dt
)
{
    // Local variables for storing matrix and vector data during assembly.
    Eigen::MatrixX<FloatType> node_coords_h, node_coords_u, qpoints_x, h_phi_r, h_dphi_r,
        h_dphi_x, u_phi_r, u_dphi_r, u_dphi_x;
    Eigen::VectorX<FloatType> qweights, qpoints_r, detJ_r;
    Eigen::VectorXi element_h, element_u;

    // Perform Gauss-Legendre quadrature for the given precision.
    FEM1D::gauss_legendre_quadrature(gp_rhs, qpoints_r, qweights);

    // Generate Lagrange basis functions for the finite element degree.
    FEM1D::lagrange_basis(
        h_mesh.degree(), qpoints_r, h_phi_r, h_dphi_r
    );
    FEM1D::lagrange_basis(
        u_mesh.degree(), qpoints_r, u_phi_r, u_dphi_r
    );

    // Initialize matrices and vectors for mapping and calculation.
    qpoints_x = Eigen::MatrixX<FloatType>::Zero(qpoints_r.rows(), 2);
    detJ_r = Eigen::VectorX<FloatType>::Zero(qpoints_r.rows());
    h_dphi_x = Eigen::MatrixX<FloatType>::Zero(
        qpoints_r.rows(), h_mesh.dofs_per_cell()
    );
    u_dphi_x = Eigen::MatrixX<FloatType>::Zero(
        qpoints_r.rows(), u_mesh.dofs_per_cell()
    );

    // Loop over cells to compute the contribution to the RHS vector.
    for (int ci = 0; ci < h_mesh.nof_cells(); ++ci) {
        element_h = h_mesh.cmat(ci, Eigen::all);
        element_u = u_mesh.cmat(ci, Eigen::all);
        node_coords_h = h_mesh.pmat(element_h, Eigen::all);
        node_coords_u = u_mesh.pmat(element_u, Eigen::all);
        FEM1D::map_to_reference_cell(
            u_mesh.degree(), node_coords_u, qpoints_r,
            u_phi_r, u_dphi_r, detJ_r, qpoints_x, u_dphi_x
        );
        FEM1D::map_to_reference_cell(
            h_mesh.degree(), node_coords_h, qpoints_r,
            h_phi_r, h_dphi_r, detJ_r, qpoints_x, h_dphi_x
        );

        // Evaluate the functions at the quadrature points.
        Eigen::VectorX<FloatType> h0_vec = h_phi_r*h0_fem_func.eval_cell(ci);
        Eigen::VectorX<FloatType> dh_vec = h_dphi_x*h0_fem_func.eval_cell(ci);
        Eigen::VectorX<FloatType> ux_vec = u_phi_r*ux_fem_func.eval_cell(ci);
        Eigen::VectorX<FloatType> uz_vec = u_phi_r*uz_fem_func.eval_cell(ci);
        Eigen::VectorX<FloatType> ac_vec = h_phi_r*ac_fem_func.eval_cell(ci);

        Eigen::VectorX<FloatType> h0_vecOld = h_phi_r*h0_fem_funcOld.eval_cell(ci);
        Eigen::VectorX<FloatType> dh_vecOld = h_dphi_x*h0_fem_funcOld.eval_cell(ci);
        Eigen::VectorX<FloatType> ux_vecOld = u_phi_r*ux_fem_funcOld.eval_cell(ci);
        Eigen::VectorX<FloatType> uz_vecOld = u_phi_r*uz_fem_funcOld.eval_cell(ci);
        Eigen::VectorX<FloatType> ac_vecOld = h_phi_r*ac_fem_funcOld.eval_cell(ci);

        // Integrate over quadrature points to compute the RHS contributions.
        for (int q = 0; q < qpoints_r.rows(); ++q) {
            FloatType detJxW = qweights(q)*detJ_r(q);
            for (int i = 0; i < h_mesh.dofs_per_cell(); ++i) {
                rhs_vec(element_h(i)) += h_phi_r(q, i)*(
                    h0_vec(q) 
                    + weightk*dt*(uz_vec(q) - ux_vec(q)*dh_vec(q) + ac_vec(q))
                    + weightkm1*dt*(uz_vecOld(q) - ux_vecOld(q)*dh_vecOld(q) + ac_vecOld(q))
                )*detJxW;
            }
        }
    }
}



void FreeSurfaceProblem::assemble_lhs_simplicit(
    FEMFunction1D &ux_fem_func, FloatType dt
) {
    // Local variables for storing matrix and vector data during assembly.
    Eigen::MatrixX<FloatType> node_coords_h, node_coords_u, qpoints_x, h_phi_r, h_dphi_r,
        h_dphi_x, u_phi_r, u_dphi_r, u_dphi_x;
    Eigen::VectorX<FloatType> qweights, qpoints_r, detJ_r;
    Eigen::VectorXi element_h, element_u;

    // Perform Gauss-Legendre quadrature for the given precision.
    FEM1D::gauss_legendre_quadrature(gp_lhs, qpoints_r, qweights);

    // Generate Lagrange basis functions for the finite element degree.
    FEM1D::lagrange_basis(
        h_mesh.degree(), qpoints_r, h_phi_r, h_dphi_r
    );
    FEM1D::lagrange_basis(
        u_mesh.degree(), qpoints_r, u_phi_r, u_dphi_r
    );

    // Initialize matrices and vectors for mapping and calculation.
    qpoints_x = Eigen::MatrixX<FloatType>::Zero(qpoints_r.rows(), 2);
    detJ_r = Eigen::VectorX<FloatType>::Zero(qpoints_r.rows());
    h_dphi_x = Eigen::MatrixX<FloatType>::Zero(
        qpoints_r.rows(), h_mesh.dofs_per_cell()
    );
    u_dphi_x = Eigen::MatrixX<FloatType>::Zero(
        qpoints_r.rows(), u_mesh.dofs_per_cell()
    );

    // Initialize the matrix for each element.
    Eigen::MatrixX<FloatType> A = Eigen::MatrixX<FloatType>::Zero(
        h_mesh.dofs_per_cell(), h_mesh.dofs_per_cell()
    );

    // Loop over cells to compute the implicit contribution to the LHS matrix.
    for (int ci = 0; ci < h_mesh.nof_cells(); ++ci) {
        element_h = h_mesh.cmat(ci, Eigen::all);
        element_u = u_mesh.cmat(ci, Eigen::all);
        node_coords_h = h_mesh.pmat(element_h, Eigen::all);
        node_coords_u = u_mesh.pmat(element_u, Eigen::all);
        FEM1D::map_to_reference_cell(
            u_mesh.degree(), node_coords_u, qpoints_r,
            u_phi_r, u_dphi_r, detJ_r, qpoints_x, u_dphi_x
        );
        FEM1D::map_to_reference_cell(
            h_mesh.degree(), node_coords_h, qpoints_r,
            h_phi_r, h_dphi_r, detJ_r, qpoints_x, h_dphi_x
        );

        // Evaluate the velocity function at the quadrature points.
        Eigen::VectorX<FloatType> ux_vec = u_phi_r*ux_fem_func.eval_cell(ci);

        // Integrate over quadrature points to compute the LHS matrix.
        for (int q = 0; q < qpoints_r.rows(); ++q) {
            FloatType detJxW = qweights(q)*detJ_r(q);
            for (int i = 0; i < h_mesh.dofs_per_cell(); ++i) {
                for (int j = 0; j < h_mesh.dofs_per_cell(); ++j) {
                    A(i, j) += (
                        h_phi_r(q, i)*h_phi_r(q, j) +
                        dt*ux_vec(q)*h_phi_r(q, i)*h_dphi_x(q, j)// + 0.01*dt*h_dphi_x(q, i)*h_dphi_x(q, j)
                    )*detJxW;
                }
            }
        }

        // Store the computed coefficients into the triplet list.
        for (int i = 0; i < h_mesh.dofs_per_cell(); ++i) {
            for (int j = 0; j < h_mesh.dofs_per_cell(); ++j) {
                lhs_coeffs.push_back(Eigen::Triplet<FloatType>(
                    element_h(i),
                    element_h(j),
                    A(i, j)
                ));
            }
        }
        A.setZero();
    }
    nof_pushed_elements = lhs_coeffs.size();
}

void FreeSurfaceProblem::assemble_rhs_simplicit(
    FEMFunction1D &h0_fem_func,
    FEMFunction1D &uz_fem_func,
    FEMFunction1D &ac_fem_func,
    FloatType dt
) {
    // Local variables for storing matrix and vector data during assembly.
    Eigen::MatrixX<FloatType> node_coords_h, node_coords_u, qpoints_x, h_phi_r, h_dphi_r,
        h_dphi_x, u_phi_r, u_dphi_r, u_dphi_x;
    Eigen::VectorX<FloatType> qweights, qpoints_r, detJ_r;
    Eigen::VectorXi element_h, element_u;

    // Perform Gauss-Legendre quadrature for the given precision.
    FEM1D::gauss_legendre_quadrature(gp_rhs, qpoints_r, qweights);

    // Generate Lagrange basis functions for the finite element degree.
    FEM1D::lagrange_basis(
        h_mesh.degree(), qpoints_r, h_phi_r, h_dphi_r
    );
    FEM1D::lagrange_basis(
        u_mesh.degree(), qpoints_r, u_phi_r, u_dphi_r
    );

    // Initialize matrices and vectors for mapping and calculation.
    qpoints_x = Eigen::MatrixX<FloatType>::Zero(qpoints_r.rows(), 2);
    detJ_r = Eigen::VectorX<FloatType>::Zero(qpoints_r.rows());
    h_dphi_x = Eigen::MatrixX<FloatType>::Zero(
        qpoints_r.rows(), h_mesh.dofs_per_cell()
    );
    u_dphi_x = Eigen::MatrixX<FloatType>::Zero(
        qpoints_r.rows(), u_mesh.dofs_per_cell()
    );

    // Loop over cells to compute the implicit contribution to the RHS vector.
    for (int ci = 0; ci < h_mesh.nof_cells(); ++ci) {
        element_h = h_mesh.cmat(ci, Eigen::all);
        element_u = u_mesh.cmat(ci, Eigen::all);
        node_coords_h = h_mesh.pmat(element_h, Eigen::all);
        node_coords_u = u_mesh.pmat(element_u, Eigen::all);
        FEM1D::map_to_reference_cell(
            u_mesh.degree(), node_coords_u, qpoints_r,
            u_phi_r, u_dphi_r, detJ_r, qpoints_x, u_dphi_x
        );
        FEM1D::map_to_reference_cell(
            h_mesh.degree(), node_coords_h, qpoints_r,
            h_phi_r, h_dphi_r, detJ_r, qpoints_x, h_dphi_x
        );

        // Evaluate the functions at the quadrature points.
        Eigen::VectorX<FloatType> h0_vec = h_phi_r*h0_fem_func.eval_cell(ci);
        Eigen::VectorX<FloatType> uz_vec = u_phi_r*uz_fem_func.eval_cell(ci);
        Eigen::VectorX<FloatType> ac_vec = h_phi_r*ac_fem_func.eval_cell(ci);

        // Integrate over quadrature points to compute the RHS contributions.
        for (int q = 0; q < qpoints_r.rows(); ++q) {
            FloatType detJxW = qweights(q)*detJ_r(q);
            for (int i = 0; i < h_mesh.dofs_per_cell(); ++i) {
                rhs_vec(element_h(i)) += h_phi_r(q, i)*(
                    h0_vec(q) + dt*(uz_vec(q) + ac_vec(q))
                )*detJxW;
            }
        }
    }
}

void FreeSurfaceProblem::commit_lhs()
{
    lhs_mat.setFromTriplets(lhs_coeffs.begin(), lhs_coeffs.end());
}

void FreeSurfaceProblem::solve_linear_system()
{
    Eigen::SparseLU<Eigen::SparseMatrix<FloatType>> solver;
    solver.analyzePattern(lhs_mat);
    solver.factorize(lhs_mat);
    zs_vec = solver.solve(rhs_vec);
}

FEMFunction1D FreeSurfaceProblem::height()
{
    FEMFunction1D h_func(h_mesh);
    h_func.vals.noalias() = zs_vec;
    return h_func;
}

void FreeSurfaceProblem::reset_lhs()
{
    lhs_coeffs.clear();
    nof_pushed_elements = 0;
}

void FreeSurfaceProblem::reset_rhs()
{
    rhs_vec.setZero();
}

void FreeSurfaceProblem::reset_system()
{
    reset_lhs();
    reset_rhs();
}

void FreeSurfaceProblem::assemble_lhs_vBDF2(
    FEMFunction1D &ux_fem_func, FloatType dt, FloatType dtOld
) {
    // Local variables for storing matrix and vector data during assembly.
    Eigen::MatrixX<FloatType> node_coords_h, node_coords_u, qpoints_x, h_phi_r, h_dphi_r,
        h_dphi_x, u_phi_r, u_dphi_r, u_dphi_x;
    Eigen::VectorX<FloatType> qweights, qpoints_r, detJ_r;
    Eigen::VectorXi element_h, element_u;

    // Perform Gauss-Legendre quadrature for the given precision.
    FEM1D::gauss_legendre_quadrature(gp_lhs, qpoints_r, qweights);

    // Generate Lagrange basis functions for the finite element degree.
    FEM1D::lagrange_basis(
        h_mesh.degree(), qpoints_r, h_phi_r, h_dphi_r
    );
    FEM1D::lagrange_basis(
        u_mesh.degree(), qpoints_r, u_phi_r, u_dphi_r
    );

    // Initialize matrices and vectors for mapping and calculation.
    qpoints_x = Eigen::MatrixX<FloatType>::Zero(qpoints_r.rows(), 2);
    detJ_r = Eigen::VectorX<FloatType>::Zero(qpoints_r.rows());
    h_dphi_x = Eigen::MatrixX<FloatType>::Zero(
        qpoints_r.rows(), h_mesh.dofs_per_cell()
    );
    u_dphi_x = Eigen::MatrixX<FloatType>::Zero(
        qpoints_r.rows(), u_mesh.dofs_per_cell()
    );

    // Initialize the matrix for each element.
    Eigen::MatrixX<FloatType> A = Eigen::MatrixX<FloatType>::Zero(
        h_mesh.dofs_per_cell(), h_mesh.dofs_per_cell()
    );

    FloatType r = dtOld/dt;

    // Loop over cells to compute the implicit contribution to the LHS matrix.
    for (int ci = 0; ci < h_mesh.nof_cells(); ++ci) {
        element_h = h_mesh.cmat(ci, Eigen::all);
        element_u = u_mesh.cmat(ci, Eigen::all);
        node_coords_h = h_mesh.pmat(element_h, Eigen::all);
        node_coords_u = u_mesh.pmat(element_u, Eigen::all);
        FEM1D::map_to_reference_cell(
            u_mesh.degree(), node_coords_u, qpoints_r,
            u_phi_r, u_dphi_r, detJ_r, qpoints_x, u_dphi_x
        );
        FEM1D::map_to_reference_cell(
            h_mesh.degree(), node_coords_h, qpoints_r,
            h_phi_r, h_dphi_r, detJ_r, qpoints_x, h_dphi_x
        );

        // Evaluate the velocity function at the quadrature points.
        Eigen::VectorX<FloatType> ux_vec = u_phi_r*ux_fem_func.eval_cell(ci);

        // Integrate over quadrature points to compute the LHS matrix.
        for (int q = 0; q < qpoints_r.rows(); ++q) {
            FloatType detJxW = qweights(q)*detJ_r(q);
            for (int i = 0; i < h_mesh.dofs_per_cell(); ++i) {
                for (int j = 0; j < h_mesh.dofs_per_cell(); ++j) {
                    /*A(i, j) += (
                        h_phi_r(q, i)*h_phi_r(q, j) +
                        (2.0/3.0)*dt*ux_vec(q)*h_phi_r(q, i)*h_dphi_x(q, j)
                    )*detJxW;*/
                    A(i, j) += (
                        h_phi_r(q, i)*h_phi_r(q, j) +
                        ((r+1)/(r+2))*dt*ux_vec(q)*h_phi_r(q, i)*h_dphi_x(q, j)
                    )*detJxW;
                }
            }
        }

        // Store the computed coefficients into the triplet list.
        for (int i = 0; i < h_mesh.dofs_per_cell(); ++i) {
            for (int j = 0; j < h_mesh.dofs_per_cell(); ++j) {
                lhs_coeffs.push_back(Eigen::Triplet<FloatType>(
                    element_h(i),
                    element_h(j),
                    A(i, j)
                ));
            }
        }
        A.setZero();
    }
    nof_pushed_elements = lhs_coeffs.size();
}

void FreeSurfaceProblem::assemble_rhs_vBDF2(
    FEMFunction1D &h0_fem_func,
    FEMFunction1D &ux_fem_func,
    FEMFunction1D &uz_fem_func,
    FEMFunction1D &ac_fem_func,
    FEMFunction1D &h0_fem_funcOld,
    FloatType dt,
    FloatType dtOld
)
{
    // Local variables for storing matrix and vector data during assembly.
    Eigen::MatrixX<FloatType> node_coords_h, node_coords_u, qpoints_x, h_phi_r, h_dphi_r,
        h_dphi_x, u_phi_r, u_dphi_r, u_dphi_x;
    Eigen::VectorX<FloatType> qweights, qpoints_r, detJ_r;
    Eigen::VectorXi element_h, element_u;

    // Perform Gauss-Legendre quadrature for the given precision.
    FEM1D::gauss_legendre_quadrature(gp_rhs, qpoints_r, qweights);

    // Generate Lagrange basis functions for the finite element degree.
    FEM1D::lagrange_basis(
        h_mesh.degree(), qpoints_r, h_phi_r, h_dphi_r
    );
    FEM1D::lagrange_basis(
        u_mesh.degree(), qpoints_r, u_phi_r, u_dphi_r
    );

    // Initialize matrices and vectors for mapping and calculation.
    qpoints_x = Eigen::MatrixX<FloatType>::Zero(qpoints_r.rows(), 2);
    detJ_r = Eigen::VectorX<FloatType>::Zero(qpoints_r.rows());
    h_dphi_x = Eigen::MatrixX<FloatType>::Zero(
        qpoints_r.rows(), h_mesh.dofs_per_cell()
    );
    u_dphi_x = Eigen::MatrixX<FloatType>::Zero(
        qpoints_r.rows(), u_mesh.dofs_per_cell()
    );

    FloatType r = dtOld/dt;

    // Loop over cells to compute the contribution to the RHS vector.
    for (int ci = 0; ci < h_mesh.nof_cells(); ++ci) {
        element_h = h_mesh.cmat(ci, Eigen::all);
        element_u = u_mesh.cmat(ci, Eigen::all);
        node_coords_h = h_mesh.pmat(element_h, Eigen::all);
        node_coords_u = u_mesh.pmat(element_u, Eigen::all);
        FEM1D::map_to_reference_cell(
            u_mesh.degree(), node_coords_u, qpoints_r,
            u_phi_r, u_dphi_r, detJ_r, qpoints_x, u_dphi_x
        );
        FEM1D::map_to_reference_cell(
            h_mesh.degree(), node_coords_h, qpoints_r,
            h_phi_r, h_dphi_r, detJ_r, qpoints_x, h_dphi_x
        );

        // Evaluate the functions at the quadrature points.
        Eigen::VectorX<FloatType> h0_vec = h_phi_r*h0_fem_func.eval_cell(ci);
        Eigen::VectorX<FloatType> dh_vec = h_dphi_x*h0_fem_func.eval_cell(ci);
        Eigen::VectorX<FloatType> ux_vec = u_phi_r*ux_fem_func.eval_cell(ci);
        Eigen::VectorX<FloatType> uz_vec = u_phi_r*uz_fem_func.eval_cell(ci);
        Eigen::VectorX<FloatType> ac_vec = h_phi_r*ac_fem_func.eval_cell(ci);
        Eigen::VectorX<FloatType> h0_vecOld = h_phi_r*h0_fem_funcOld.eval_cell(ci);


        // Integrate over quadrature points to compute the RHS contributions.
        for (int q = 0; q < qpoints_r.rows(); ++q) {
            FloatType detJxW = qweights(q)*detJ_r(q);
            for (int i = 0; i < h_mesh.dofs_per_cell(); ++i) {
                /*rhs_vec(element_h(i)) += h_phi_r(q, i)*(
                    (4.0/3.0)*h0_vec(q)-(1.0/3.0)*h0_vecOld(q) + (2.0/3.0)*dt*(uz_vec(q) + ac_vec(q))
                )*detJxW;*/
                rhs_vec(element_h(i)) += h_phi_r(q, i)*(
                    (((r+1)*(r+1))/(r*(r+2)))*h0_vec(q)-(1.0/(r*(r+2)))*h0_vecOld(q) + ((r+1)/(r+2))*dt*(uz_vec(q) + ac_vec(q))
                )*detJxW;
                
            }
        }
    }
}