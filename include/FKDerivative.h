#ifndef FK_DERIVATIVE
#define FK_DERIVATIVE
#undef __CUDACC__
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <totalmodel.h>

template <typename T>
void AngleAxisToRotationMatrix_Derivative(const T* pose, T* dR_data, const int idj, const int numberColumns=TotalModel::NUM_JOINTS * 3);

void EulerAnglesToRotationMatrix_Derivative(const double* pose, double* dR_data, const int idj, const int numberColumns=TotalModel::NUM_JOINTS * 3);

template <typename T>
void Product_Derivative(const T* const A_data, const T* const dA_data, const T* const B_data,
                        const T* const dB_data, T* dAB_data, const int B_col=3);
template <typename T>
void SparseProductDerivative(const T* const A_data, const T* const dA_data, const T* const B_data,
                             const T* const dB_data, const int colIndex,
                             const std::vector<int>& parentIndexes, T* dAB_data, const int numberColumns=TotalModel::NUM_JOINTS * 3, bool debug=0);
template <typename T>
void SparseProductDerivative(const T* const dA_data, const T* const B_data,
                             const std::vector<int>& parentIndexes, T* dAB_data, const int numberColumns=TotalModel::NUM_JOINTS * 3, bool debug=0);
template <typename T>
void SparseProductDerivativeConstA(const T* const A_data, const T* const dB_data,
                             const std::vector<int>& parentIndexes, T* dAB_data, const int numberColumns=TotalModel::NUM_JOINTS * 3, bool debug=0);
template <typename T>
void SparseAdd(const T* const B_data, const std::vector<int>& parentIndexes, T* A_data, const int numberColumns=TotalModel::NUM_JOINTS * 3);
template <typename T>
void SparseSubtract(const T* const B_data, const std::vector<int>& parentIndexes, T* A_data, const int numberColumns=TotalModel::NUM_JOINTS * 3);

#endif
