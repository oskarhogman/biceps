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
#define HEADER_LINE1 #pragma once
#ifdef USE_LONG_DOUBLE
    #define HEADER_LINE2 #define ABS_FUNC fabsl
    #define HEADER_LINE3 #define SQRT_FUNC sqrtl
    #define HEADER_LINE4 #define POW_FUNC powl
    #define HEADER_LINE5 #define SIN_FUNC sinl
    #define HEADER_LINE6 #define COS_FUNC cosl
    #define HEADER_LINE7 #define PI_CONST M_PIl
    #define HEADER_LINE8 typedef long double FloatType;
#else
    #define HEADER_LINE2 #define ABS_FUNC fabs
    #define HEADER_LINE3 #define SQRT_FUNC sqrt
    #define HEADER_LINE4 #define POW_FUNC pow
    #define HEADER_LINE5 #define SIN_FUNC sin
    #define HEADER_LINE6 #define COS_FUNC cos
    #define HEADER_LINE7 #define PI_CONST M_PI
    #define HEADER_LINE8 typedef double FloatType;
#endif
HEADER_LINE1
HEADER_LINE2
HEADER_LINE3
HEADER_LINE4
HEADER_LINE5
HEADER_LINE6
HEADER_LINE7
HEADER_LINE8
