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
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Biceps. If not, see <https://www.gnu.org/licenses/>.
 */
private:
int binary_find(const Eigen::VectorXi &vec, int e)
{
    int low = 0;
    int high = vec.size() - 1;
    int mid;

    while (high - low > 1) {
        int mid = (high + low) >> 1;
        if (vec[mid] < e) {
            low = mid + 1;
        }
        else {
            high = mid;
        }
    }
    if (vec[low] == e) {
        return low;
    } else if (vec[high] == e) {
        return high;
    } else {
        return -1;
    }
}

int binary_find(const std::vector<int> &vec, int e)
{
    int low = 0;
    int high = vec.size() - 1;
    int mid;

    while (high - low > 1) {
        int mid = (high + low) >> 1;
        if (vec[mid] < e)
        {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    if (vec[low] == e) {
        return low;
    } else if (vec[high] == e) {
        return high;
    } else {
        return -1;
    }
}

public:
inline SparseMatrix<FloatType> extract_block(
    const Eigen::VectorXi &row_inds, const Eigen::VectorXi &col_inds
) {
    SparseMatrix<FloatType> sp_block(row_inds.size(), col_inds.size());
    std::vector<Eigen::Triplet<FloatType>> block_coeffs;
    block_coeffs.reserve(nonZeros());
    for (int col = 0; col < this->outerSize(); col++) {
        int block_col = binary_find(col_inds, col);
        for (
            SparseMatrix<FloatType>::InnerIterator it(*this, col); it; ++it
        ) {
        int row = it.row();
        FloatType val = it.value();
        int block_row = binary_find(row_inds, row);
            if (block_row != -1 && block_col != -1) {
                block_coeffs.push_back(
                    Eigen::Triplet<FloatType>(block_row, block_col, val)
                );
            }
        }
    }

    sp_block.setFromTriplets(block_coeffs.begin(), block_coeffs.end());
    return sp_block;
}

inline SparseMatrix<FloatType> extract_block(
    const std::vector<int> &row_inds, const std::vector<int> &col_inds
) {
    SparseMatrix<FloatType> sp_block(row_inds.size(), col_inds.size());
    std::vector<Eigen::Triplet<FloatType>> block_coeffs;
    block_coeffs.reserve(nonZeros());
    for (int col = 0; col < this->outerSize(); col++) {
        int block_col = binary_find(col_inds, col);
        for (
            SparseMatrix<FloatType>::InnerIterator it(*this, col); it; ++it
        ) {
            int row = it.row();
            FloatType val = it.value();
            int block_row = binary_find(row_inds, row);
            if (block_row != -1 && block_col != -1) {
                block_coeffs.push_back(
                    Eigen::Triplet<FloatType>(block_row, block_col, val)
                );
            }
        }
    }
    sp_block.setFromTriplets(block_coeffs.begin(), block_coeffs.end());
    return sp_block;
}
