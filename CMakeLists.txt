#
# Copyright (C) 2025 André Löfgren
#
# This file is part of Biceps.
#
# Biceps is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# Biceps is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Biceps. If not, see <https://www.gnu.org/licenses/>.
#
cmake_minimum_required(VERSION 3.10)
project(biceps VERSION 1.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
set(ENABLE_PYTHON ON CACHE BOOL "Enable Python support")
set(BUILD_SHARED_LIBS ON CACHE BOOL "Enable shared libs")
set(USE_LONG_DOUBLE OFF CACHE BOOL "Whether to use long double")
set(ENABLE_DOCS ON CACHE BOOL "Build documentation")
set(DOXYGEN_HTML ON CACHE BOOL "Build html documentation")
set(DOXYGEN_LATEX ON CACHE BOOL "Build tex documentation")
set(ENABLE_TESTS ON CACHE BOOL "Enable unit testing")

# Check for Eigen3 and Boost (mandatory)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(Boost 1.74 REQUIRED)

# Check for Python interface dependencies (optional)
find_package(Python COMPONENTS Development NumPy QUIET)
find_package(Boost 1.74 COMPONENTS python QUIET)
find_package(eigenpy QUIET)
if(
    BUILD_SHARED_LIBS AND
    ENABLE_PYTHON AND
    Python_FOUND AND
    Python_Development_FOUND AND
    Python_NumPy_FOUND AND
    Boost_PYTHON_FOUND AND
    eigenpy_FOUND
)
    set(PYTHON_WRAPPER ${CMAKE_SOURCE_DIR}/src/pywrapper.cpp)
    message(STATUS "Building Python interface.")
elseif(ENABLE_PYTHON AND NOT BUILD_SHARED_LIBS)
    set(ENABLE_PYTHON OFF)
    message(WARNING "BUILD_SHARED_LIBS is OFF, but ENABLE_PYTHON is ON; Python interface will not be built.")
elseif(ENABLE_PYTHON AND NOT Python_FOUND)
    set(ENABLE_PYTHON OFF)
    message(WARNING "Python not found, but ENABLE_PYTHON is ON; Python interface will not be built.")
elseif(ENABLE_PYTHON AND NOT Python_Development_FOUND)
    set(ENABLE_PYTHON OFF)
    message(WARNING "Python development files not found, but ENABLE_PYTHON is ON; Python interface will not be built.")
elseif(ENABLE_PYTHON AND NOT Python_NumPy_FOUND)
    set(ENABLE_PYTHON OFF)
    message(WARNING "NumPy not found, but ENABLE_PYTHON is ON; disabling Python; Python interface will not be built.")
elseif(ENABLE_PYTHON AND NOT Boost_PYTHON_FOUND)
    set(ENABLE_PYTHON OFF)
    message(WARNING "Boost Python module not found, but ENABLE_PYTHON is ON; Python interface will not be built.")
elseif(ENABLE_PYTHON AND NOT eigenpy_FOUND)
    set(ENABLE_PYTHON OFF)
    message(WARNING "EigenPy not found, but ENABLE_PYTHON is ON; Python interface will not be built.")
else()
    message(STATUS "Python interface will not be built.")
endif()

# Check for documentation dependencies (optional)
find_package(Doxygen QUIET)
if(ENABLE_DOCS AND DOXYGEN_FOUND)
    message(STATUS "Doxygen found, enabling documentation target.")

    set(DOXYGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/docs)
    set(DOXYGEN_CONFIG_FILE ${CMAKE_BINARY_DIR}/Doxyfile)

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${DOXYGEN_CONFIG_FILE} @ONLY)

    add_custom_target(
        docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_CONFIG_FILE}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generate documentation using Doxygen"
        VERBATIM
    )

    # Install documentation (HTML and LaTeX if enabled)
    if(DOXYGEN_HTML)
        install(
            DIRECTORY ${DOXYGEN_OUTPUT_DIR}/html
            DESTINATION share/doc/biceps
            PATTERN ".*" EXCLUDE
        )
    endif()

    if(DOXYGEN_LATEX)
        install(
            DIRECTORY ${DOXYGEN_OUTPUT_DIR}/latex
            DESTINATION share/doc/biceps
            PATTERN ".*" EXCLUDE
        )
    endif()
elseif(ENABLE_DOCS AND NOT DOXYGEN_FOUND)
    message(WARNING "Doxygen not found, but ENABLE_DOCS is ON. Documentation target disabled.")
else()
    message(STATUS "Documentation will not be built.")
endif()

# Check for unit testing dependency (optional)
find_package(Boost 1.74 COMPONENTS unit_test_framework QUIET)
if(ENABLE_TESTS AND Boost_UNIT_TEST_FRAMEWORK_FOUND)
    message(STATUS "Boost unit test framework found; enabling test target.")
    add_subdirectory(unit_tests)
elseif(ENABLE_TESTS AND NOT Boost_UNIT_TEST_FRAMEWORK_FOUND)
    message(WARNING "Boost unit test framework not found, but ENABLE_TESTS is ON; disabling test target.")
else()
    message(STATUS "test target disabled.")
endif()

# Project source files
add_subdirectory(src)

install(
    FILES 
        ${CMAKE_BINARY_DIR}/float_type.hpp
        include/eigen_spmat_addons.hpp
        include/enums.hpp
        include/logger.hpp
        include/interval_mesh.hpp
        include/structured_mesh.hpp
        include/fem_1d.hpp
        include/fem_2d.hpp
        include/fem_function_1d.hpp
        include/fem_function_2d.hpp
        include/pstokes_fem.hpp
        include/poisson_fem.hpp
        include/free_surface_fem.hpp
        include/time_integrator.hpp
    DESTINATION
        include
)
