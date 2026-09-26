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
#include <boost/python.hpp>
#include <Eigen/Dense>
#include <eigenpy/eigenpy.hpp>
#include <enums.hpp>
#include <interval_mesh.hpp>
#include <logger.hpp>
#include <structured_mesh.hpp>
#include <fem_function_1d.hpp>
#include <fem_function_2d.hpp>
#include <poisson_fem.hpp>
#include <pstokes_fem.hpp>
#include <free_surface_fem.hpp>
#include <time_integrator.hpp>

namespace py = boost::python;

// Convert std::vector to Eigen::Vector as the latter automatically gets converted to numpy array by eigenpy
template<typename T>
Eigen::VectorX<T> stdvec_to_eigvec(const std::vector<T> &std_vec)
{
    return Eigen::Map<const Eigen::VectorX<T>>(std_vec.data(), std_vec.size());
}

// Convert Python function signatures to C++ signatures so that one can directly pass python functions to e.g., class methods
template<typename T, typename... Args>
std::function<T(Args...)> pyfunc_to_cppfunc(py::object py_func)
{
    if (!PyCallable_Check(py_func.ptr())) {
        throw std::runtime_error("Provided Python object is not callable.");
    }

    return [py_func](Args... args) -> T {
        return py::extract<T>(py_func(args...));
    };
}

BOOST_PYTHON_MODULE(biceps)
{
    // Expose Eigen
    eigenpy::enableEigenPy();

    // Expose enums and constants
    py::scope().attr("SEC_PER_YEAR") = SEC_PER_YEAR;
    py::scope().attr("PA_TO_MPA") = PA_TO_MPA;
    py::enum_<FSSA_VERSION>("FSSA_VERSION")
        .value("FSSA_NONE", FSSA_NONE)
        .value("FSSA_NORMAL", FSSA_NORMAL)
        .value("FSSA_VERTICAL", FSSA_VERTICAL);
    py::enum_<MESH1D::DOMAIN_IDS>("DOMAIN_IDS_1D")
        .value("EMPTY_ID", MESH1D::EMPTY_ID)
        .value("INTERIOR_ID", MESH1D::INTERIOR_ID)
        .value("WEST_ID", MESH1D::WEST_ID)
        .value("EAST_ID", MESH1D::EAST_ID)
        .value("BOUNDARY_ID", MESH1D::BOUNDARY_ID)
        .value("DOMAIN_ID", MESH1D::DOMAIN_ID);
    py::enum_<MESH2D::CELL_TYPE>("CELL_TYPE_2D")
        .value("QUADRILATERAL", MESH2D::QUADRILATERAL)
        .value("TRIANGLE_LEFT", MESH2D::TRIANGLE_LEFT)
        .value("TRIANGLE_RIGHT", MESH2D::TRIANGLE_RIGHT);
    py::enum_<MESH2D::DOMAIN_IDS>("DOMAIN_IDS_2D")
        .value("EMPTY_ID", MESH2D::EMPTY_ID)
        .value("INTERIOR_ID", MESH2D::INTERIOR_ID)
        .value("NORTH_ID", MESH2D::NORTH_ID)
        .value("NORTH_WEST_ID", MESH2D::NORTH_WEST_ID)
        .value("WEST_ID", MESH2D::WEST_ID)
        .value("SOUTH_WEST_ID", MESH2D::SOUTH_WEST_ID)
        .value("SOUTH_ID", MESH2D::SOUTH_ID)
        .value("SOUTH_EAST_ID", MESH2D::SOUTH_EAST_ID)
        .value("EAST_ID", MESH2D::EAST_ID)
        .value("NORTH_EAST_ID", MESH2D::NORTH_EAST_ID)
        .value("SURFACE_ID", MESH2D::SURFACE_ID)
        .value("BED_ID", MESH2D::BED_ID)
        .value("BOUNDARY_ID", MESH2D::BOUNDARY_ID)
        .value("DOMAIN_ID", MESH2D::DOMAIN_ID);

    // Expose IntervalMesh
    py::class_<IntervalMesh>("IntervalMesh", py::init<FloatType, FloatType, int, int>())
        .def(py::init<const Eigen::MatrixX<FloatType> &, int>())
        .def("extract_dof_inds", +[](IntervalMesh &self, int id) {
            return stdvec_to_eigvec<int>(self.extract_dof_inds(id));
        })
        .def("nof_cells", &IntervalMesh::nof_cells)
        .def("nof_verts", &IntervalMesh::nof_verts)
        .def("nof_dofs", &IntervalMesh::nof_dofs)
        .def("dofs_per_cell", &IntervalMesh::dofs_per_cell)
        .def("degree", &IntervalMesh::degree)
        .def_readwrite("pmat", &IntervalMesh::pmat)
        .def_readwrite("cmat", &IntervalMesh::cmat)
        .def_readwrite("dimat", &IntervalMesh::dimat);

    // Expose StructuredMesh
    py::class_<StructuredMesh>("StructuredMesh", py::init<int, int, int, int>())
        .def("extrude_x", static_cast<void (StructuredMesh::*)(const Eigen::VectorX<FloatType>&)>(&StructuredMesh::extrude_x))
        .def("extrude_x", static_cast<void (StructuredMesh::*)(FloatType, FloatType)>(&StructuredMesh::extrude_x))
        .def("extrude_z", static_cast<void (StructuredMesh::*)(const Eigen::VectorX<FloatType>&)>(&StructuredMesh::extrude_z))
        .def("extrude_z", static_cast<void (StructuredMesh::*)(const Eigen::VectorX<FloatType>&, const Eigen::VectorX<FloatType>&)>(&StructuredMesh::extrude_z))
        .def("extrude_z", +[](StructuredMesh &self, py::object zb, py::object zs) {
            self.extrude_z(pyfunc_to_cppfunc<FloatType, FloatType>(zb), pyfunc_to_cppfunc<FloatType, FloatType>(zs));
        })
        .def("extract_cell_inds", +[](StructuredMesh &self, int id) {
            return stdvec_to_eigvec<int>(self.extract_cell_inds(id));
        })
        .def("extract_edge_inds", +[](StructuredMesh &self, int id) {
            return stdvec_to_eigvec<int>(self.extract_edge_inds(id));
        })
        .def("extract_dof_inds", +[](StructuredMesh &self, int id) {
            return stdvec_to_eigvec<int>(self.extract_dof_inds(id));
        })
        .def("extract_vertex_dof_inds", +[](StructuredMesh &self, int id) {
            return stdvec_to_eigvec<int>(self.extract_vertex_dof_inds(id));
        })
        .def("nx", &StructuredMesh::nx)
        .def("nz", &StructuredMesh::nz)
        .def("hl", &StructuredMesh::hl)
        .def("vl", &StructuredMesh::vl)
        .def("nof_cells", &StructuredMesh::nof_cells)
        .def("nof_edges", &StructuredMesh::nof_edges)
        .def("dofs_per_cell", &StructuredMesh::dofs_per_cell)
        .def("dofs_per_edge", &StructuredMesh::dofs_per_edge)
        .def("cell_type", &StructuredMesh::cell_type)
        .def("area", &StructuredMesh::area)
        .def_readwrite("pmat", &StructuredMesh::pmat)
        .def_readwrite("cmat", &StructuredMesh::cmat)
        .def_readwrite("emat", &StructuredMesh::emat)
        .def_readwrite("dimat", &StructuredMesh::dimat);

    // Expose FEMFunction1D
    py::class_<FEMFunction1D>("FEMFunction1D", py::init<IntervalMesh &>())
        .def("assign", static_cast<void (FEMFunction1D::*)(const Eigen::VectorX<FloatType> &)>(&FEMFunction1D::assign))
        .def("assign", +[](FEMFunction1D &self, py::object f) {
            self.assign(pyfunc_to_cppfunc<FloatType, FloatType, FloatType>(f));
        })
        .def("integrate", &FEMFunction1D::integrate)
        .def("eval_cell", &FEMFunction1D::eval_cell)
        .def("extract_vertex_subvec", &FEMFunction1D::extract_vertex_subvec)
        .def("extract_dof_subvec", &FEMFunction1D::extract_dof_subvec)
        .def("assemble_mass_matrix", &FEMFunction1D::assemble_mass_matrix)
        .def("mass_matrix", &FEMFunction1D::mass_matrix)
        .def("L2_norm", &FEMFunction1D::L2_norm)
        .def(py::self + FloatType())
        .def(py::self - FloatType())
        .def(py::self * FloatType())
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def_readwrite("vals", &FEMFunction1D::vals);

    // Expose FEMFunction2D
    py::class_<FEMFunction2D>("FEMFunction2D", py::init<StructuredMesh &>())
        .def("assign", static_cast<void (FEMFunction2D::*)(const Eigen::VectorX<FloatType> &)>(&FEMFunction2D::assign))
        .def("assign", +[](FEMFunction2D &self, py::object f) {
            self.assign(pyfunc_to_cppfunc<FloatType, FloatType, FloatType>(f));
        })
        .def("eval_cell", &FEMFunction2D::eval_cell)
        .def("extract_vertex_subvec", &FEMFunction2D::extract_vertex_subvec)
        .def("extract_dof_subvec", &FEMFunction2D::extract_dof_subvec)
        .def("assemble_mass_matrix", &FEMFunction2D::assemble_mass_matrix)
        .def("mass_matrix", &FEMFunction2D::mass_matrix)
        .def("diff_x_interp", &FEMFunction2D::diff_x_interp)
        .def("diff_z_interp", &FEMFunction2D::diff_z_interp)
        .def("diff_x_proj", &FEMFunction2D::diff_x_interp)
        .def("diff_z_proj", &FEMFunction2D::diff_z_interp)
        .def("L2_norm", &FEMFunction2D::L2_norm)
        .def(py::self + FloatType())
        .def(py::self - FloatType())
        .def(py::self * FloatType())
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def_readwrite("vals", &FEMFunction2D::vals);

    // Expose PoissonProblem
    py::class_<PoissonProblem>("PoissonProblem", py::init<StructuredMesh &>())
        .def("assemble_stiffness_block", +[](PoissonProblem &self, py::object alpha, py::object beta, int gp) {
            self.assemble_stiffness_block(
                pyfunc_to_cppfunc<FloatType, FloatType, FloatType>(alpha),
                pyfunc_to_cppfunc<FloatType, FloatType, FloatType>(beta),
                gp
            );
        })
        .def("assemble_mass_block", +[](PoissonProblem &self, py::object gamma, int gp) {
            self.assemble_mass_block(
                pyfunc_to_cppfunc<FloatType, FloatType, FloatType>(gamma), gp
            );
        })
        .def("assemble_robin_block", +[](PoissonProblem &self, py::object a_robin, py::object b_robin, py::object g_robin, int boundary_id, int gp) {
            self.assemble_robin_block(
                pyfunc_to_cppfunc<FloatType, FloatType, FloatType>(a_robin),
                pyfunc_to_cppfunc<FloatType, FloatType, FloatType>(b_robin),
                pyfunc_to_cppfunc<FloatType, FloatType, FloatType>(g_robin),
                boundary_id,
                gp
            );
        })
        .def("commit_lhs_mat", &PoissonProblem::commit_lhs_mat)
        .def("assemble_force_rhs", +[](PoissonProblem &self, py::object f, int gp) {
            self.assemble_force_rhs(pyfunc_to_cppfunc<FloatType, FloatType, FloatType>(f), gp);
        })
        .def("assemble_force_rhs", static_cast<void (PoissonProblem::*)(FEMFunction2D &, int gp)>(&PoissonProblem::assemble_force_rhs))
        .def("assemble_neumann_rhs", +[](PoissonProblem &self, py::object g_neumann, int boundary_id, int gp) {
            self.assemble_neumann_rhs(
                pyfunc_to_cppfunc<FloatType, FloatType, FloatType>(g_neumann),
                boundary_id,
                gp
            );
        })
        .def("assemble_robin_rhs", +[](PoissonProblem &self, py::object b_robin, py::object g_robin, int boundary_id, int gp) {
            self.assemble_robin_rhs(
                pyfunc_to_cppfunc<FloatType, FloatType, FloatType>(b_robin),
                pyfunc_to_cppfunc<FloatType, FloatType, FloatType>(g_robin),
                boundary_id,
                gp
            );
        })
        .def("apply_dirichlet_bc", +[](PoissonProblem &self, py::object g_dirichlet, int boundary_id) {
            self.apply_dirichlet_bc(
                pyfunc_to_cppfunc<FloatType, FloatType, FloatType>(g_dirichlet),
                boundary_id
            );
        })
        .def("solution", &PoissonProblem::solution)
        .def("solve_linear_system", &PoissonProblem::solve_linear_system)
        .def("reset_system", &PoissonProblem::reset_system)
        .def_readonly("free_dofs", &PoissonProblem::free_dofs)
        .def_readonly("fixed_dofs", &PoissonProblem::fixed_dofs)
        .def_readonly("lhs_mat", &PoissonProblem::lhs_mat)
        .def_readonly("lhs_mat_free", &PoissonProblem::lhs_mat_free)
        .def_readonly("lhs_mat_fixed", &PoissonProblem::lhs_mat_fixed)
        .def_readonly("rhs_vec", &PoissonProblem::rhs_vec)
        .def_readonly("rhs_vec_free", &PoissonProblem::rhs_vec_free)
        .def_readonly("bc_vec", &PoissonProblem::bc_vec)
        .def_readonly("sol_vec", &PoissonProblem::sol_vec)
        .def_readonly("sol_vec_free", &PoissonProblem::sol_vec_free);

    // Expose pStokesProblem
    py::class_<pStokesProblem>("pStokesProblem", py::no_init)
    .def(
        "__init__",
        py::make_constructor(
            +[](FloatType rate_factor,
                FloatType glen_exponent,
                FloatType eps_reg_2,
                py::object func_x,
                py::object func_z,
                StructuredMesh &u_mesh,
                StructuredMesh &p_mesh
            ) {
                return new pStokesProblem(
                    rate_factor,
                    glen_exponent,
                    eps_reg_2,
                    pyfunc_to_cppfunc<FloatType, FloatType, FloatType>(func_x),
                    pyfunc_to_cppfunc<FloatType, FloatType, FloatType>(func_z),
                    u_mesh,
                    p_mesh
                );
            },
            py::default_call_policies(),
            (py::arg("rate_factor"), py::arg("glen_exponent"), py::arg("eps_reg_2"), py::arg("func_x"), py::arg("func_z"), py::arg("u_mesh"), py::arg("p_mesh"))
        )
    )
    .def("assemble_stress_block", &pStokesProblem::assemble_stress_block)
    .def("assemble_incomp_block", &pStokesProblem::assemble_incomp_block)
    .def("assemble_fssa_normal_block", &pStokesProblem::assemble_fssa_normal_block)
    .def("assemble_fssa_vertical_block", &pStokesProblem::assemble_fssa_vertical_block)
    .def("assemble_rhs_vec", &pStokesProblem::assemble_rhs_vec)
    .def("assemble_fssa_vertical_rhs_vec", &pStokesProblem::assemble_fssa_vertical_rhs)
    .def("commit_lhs_mat", &pStokesProblem::commit_lhs_mat)
    .def("prune_lhs", &pStokesProblem::prune_lhs)
    .def("apply_zero_dirichlet_bc", &pStokesProblem::apply_zero_dirichlet_bc)
    .def("apply_dirichlet_bc", +[](pStokesProblem &self, int boundary_id, int velocity_component, py::object ub_func) {
        self.apply_dirichlet_bc(
            boundary_id,
            velocity_component,
            pyfunc_to_cppfunc<FloatType, FloatType, FloatType>(ub_func)
        );
    })
    .def("solve_linear_system", &pStokesProblem::solve_linear_system)
    .def("solve_nonlinear_system", &pStokesProblem::solve_nonlinear_system)
    .def("velocity_x", &pStokesProblem::velocity_x)
    .def("velocity_z", &pStokesProblem::velocity_z)
    .def("pressure", &pStokesProblem::pressure)
    .def("resize_lhs", &pStokesProblem::resize_lhs)
    .def("reset_lhs", &pStokesProblem::reset_lhs)
    .def("reset_rhs", &pStokesProblem::reset_rhs)
    .def("reset_system", &pStokesProblem::reset_system)
    .def_readwrite("rate_factor", &pStokesProblem::rate_factor)
    .def_readwrite("glen_exponent", &pStokesProblem::glen_exponent)
    .def_readwrite("eps_reg_2", &pStokesProblem::eps_reg_2)
    .def_readwrite("fssa_version", &pStokesProblem::fssa_version)
    .def_readwrite("fssa_param", &pStokesProblem::fssa_param)
    .def_readwrite("gp_stress", &pStokesProblem::gp_stress)
    .def_readwrite("gp_incomp", &pStokesProblem::gp_incomp)
    .def_readwrite("gp_fssa_lhs", &pStokesProblem::gp_fssa_lhs)
    .def_readwrite("gp_fssa_rhs", &pStokesProblem::gp_fssa_rhs)
    .def_readwrite("gp_rhs", &pStokesProblem::gp_rhs)
    .def_readwrite("ux_dirichlet_bc_mask", &pStokesProblem::ux_dirichlet_bc_mask)
    .def_readwrite("uz_dirichlet_bc_mask", &pStokesProblem::uz_dirichlet_bc_mask)
    .def_readwrite("picard_stol", &pStokesProblem::picard_stol)
    .def_readwrite("picard_max_iter", &pStokesProblem::picard_max_iter)
    .def_readonly("nof_pushed_elements", &pStokesProblem::nof_pushed_elements)
    .def_readonly("rhs_vec", &pStokesProblem::rhs_vec)
    .def_readonly("w_vec", &pStokesProblem::w_vec);

    // Expose FreeSurfaceProblem
    py::class_<FreeSurfaceProblem>("FreeSurfaceProblem", py::init<IntervalMesh &, IntervalMesh &>())
        .def("assemble_lhs_explicit", &FreeSurfaceProblem::assemble_lhs_explicit)
        .def("assemble_rhs_explicit", &FreeSurfaceProblem::assemble_rhs_explicit)
        .def("assemble_lhs_simplicit", &FreeSurfaceProblem::assemble_lhs_simplicit)
        .def("assemble_rhs_simplicit", &FreeSurfaceProblem::assemble_rhs_simplicit)
        .def("commit_lhs", &FreeSurfaceProblem::commit_lhs)
        .def("solve_linear_system", &FreeSurfaceProblem::solve_linear_system)
        .def("height", &FreeSurfaceProblem::height)
        .def("reset_lhs", &FreeSurfaceProblem::reset_lhs)
        .def("reset_rhs", &FreeSurfaceProblem::reset_rhs)
        .def("reset_system", &FreeSurfaceProblem::reset_system)
        .def_readonly("zs_vec", &FreeSurfaceProblem::zs_vec);

    // Expose TimeIntegrator
    py::class_<TimeIntegrator>("TimeIntegrator", py::init<pStokesProblem &, FreeSurfaceProblem &>())
        .def("step_explicit", &TimeIntegrator::step_explicit)
        .def("extrude_mesh_z", &TimeIntegrator::extrude_mesh_z);
}
