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
#include <fem_function_1d.hpp>

FEMFunction1D::FEMFunction1D(IntervalMesh &mesh) : mesh(mesh)
{
    vals = Eigen::VectorX<FloatType>::Zero(mesh.nof_dofs());
}

void FEMFunction1D::assemble_mass_matrix()
{
    M = FEM1D::assemble_mass_matrix(mesh, mesh.degree()+1);
    mass_mat_is_assembled = true;
}

void FEMFunction1D::set_mass_matrix(Eigen::SparseMatrix<FloatType> &M_sp)
{
    M = M_sp;
    mass_mat_is_assembled = true;
}

Eigen::SparseMatrix<FloatType> FEMFunction1D::mass_matrix() {
    return M;
}

void FEMFunction1D::assign(std::function<FloatType(FloatType, FloatType)> func)
{
    for (int i = 0; i < mesh.nof_dofs(); i++) {
        FloatType x = mesh.pmat(i, 0);
        FloatType z = mesh.pmat(i, 1);
        vals(i) = func(x, z);
    }
}

void FEMFunction1D::assign(const Eigen::VectorX<FloatType> &vec)
{
    if (vec.size() != mesh.nof_dofs()) {
        throw std::invalid_argument("FEMFunction1D::assign: Size mismatch vec.size() != this->mesh.nof_dofs()");
    }
    for (int dof = 0; dof < vec.size(); dof++) {
        vals(dof) = vec(dof);
    }
}

Eigen::VectorX<FloatType> FEMFunction1D::eval_cell(int ci)
{
    return vals(mesh.cmat.row(ci), Eigen::all);
}

FloatType FEMFunction1D::integrate(int gp) {
    Eigen::MatrixX<FloatType> node_coords, qpoints_x, phi_r, dphi_r, dphi_x;
    Eigen::VectorX<FloatType> qweights, qpoints_r, detJ_r;
    Eigen::VectorXi element;

    FEM1D::gauss_legendre_quadrature(gp, qpoints_r, qweights);

    FEM1D::lagrange_basis(
        mesh.degree(), qpoints_r, phi_r, dphi_r
    );

    qpoints_x = Eigen::MatrixX<FloatType>::Zero(qpoints_r.rows(), 2);
    detJ_r = Eigen::VectorX<FloatType>::Zero(qpoints_r.rows());
    dphi_x = Eigen::MatrixX<FloatType>::Zero(
        qpoints_r.rows(), mesh.dofs_per_cell()
    );

    FloatType integral_sum = 0.0;
    for (int ci = 0; ci < mesh.nof_cells(); ci++) {
        element = mesh.cmat(ci, Eigen::all);
        node_coords = mesh.pmat(element, Eigen::all);
        FEM1D::map_to_reference_cell(
            mesh.degree(), node_coords, qpoints_r,
            phi_r, dphi_r, detJ_r, qpoints_x, dphi_x
        );
        Eigen::VectorX<FloatType> f_vec = phi_r*eval_cell(ci);

        for (int q = 0; q < qpoints_r.rows(); q++) {
            integral_sum += f_vec(q)*qweights(q)*detJ_r(q);
        }
    }
    return integral_sum;
}

Eigen::MatrixX<FloatType> FEMFunction1D::extract_vertex_subvec(int domain_id)
{
    std::vector<int> subdomain_inds = mesh.extract_vertex_dof_inds(domain_id);
    Eigen::MatrixX<FloatType> vector_xzf(subdomain_inds.size(), 3);
    vector_xzf(Eigen::all, Eigen::seq(0, 1)) = mesh.pmat(subdomain_inds, Eigen::all);
    vector_xzf(Eigen::all, 2) = vals(subdomain_inds);
    return vector_xzf;
}

Eigen::MatrixX<FloatType> FEMFunction1D::extract_dof_subvec(int domain_id)
{
    std::vector<int> subdomain_inds = mesh.extract_dof_inds(domain_id);
    Eigen::MatrixX<FloatType> vector_xzf(subdomain_inds.size(), 3);
    vector_xzf(Eigen::all, Eigen::seq(0, 1)) = mesh.pmat(subdomain_inds, Eigen::all);
    vector_xzf(Eigen::all, 2) = vals(subdomain_inds);
    return vector_xzf;
}

FEMFunction1D FEMFunction1D::operator+(const FEMFunction1D &f)
{
    if (vals.size() != f.vals.size()) {
        throw std::invalid_argument("FEMFunction1D::operator+: Size mismatch this->vals.size() != f.vals.size()");
    }
    FEMFunction1D func(mesh);
    func.vals = vals + f.vals;
    return func;
}

FEMFunction1D FEMFunction1D::operator-(const FEMFunction1D &f)
{
    if (vals.size() != f.vals.size()) {
        throw std::invalid_argument("FEMFunction1D::operator-: Size mismatch this->vals.size() != f.vals.size()");
    }
    FEMFunction1D func(mesh);
    func.vals = vals - f.vals;
    return func;
}

FEMFunction1D FEMFunction1D::operator*(const FEMFunction1D &f)
{
    if (vals.size() != f.vals.size()) {
        throw std::invalid_argument("FEMFunction1D::operator*: Size mismatch this->vals.size() != f.vals.size()");
    }
    FEMFunction1D func(mesh);
    func.vals = vals * f.vals;
    return func;
}

FEMFunction1D FEMFunction1D::operator+(FloatType b)
{
    FEMFunction1D func(mesh);
    func.vals = this->vals.array() + b;
    return func;
}

FEMFunction1D FEMFunction1D::operator-(FloatType b)
{
    FEMFunction1D func(mesh);
    func.vals = this->vals.array() - b;
    return func;
}

FEMFunction1D FEMFunction1D::operator*(FloatType b)
{
    FEMFunction1D func(mesh);
    func.vals = this->vals * b;
    return func;
}

FloatType FEMFunction1D::L2_norm()
{
    if (!mass_mat_is_assembled)
        throw std::runtime_error(
            "FEMFunction1D::L2_norm: Mass matrix has to be assembled before the L2 norm can be computed."
        );
    return SQRT_FUNC(FEM1D::inner(vals, M, vals));
}
