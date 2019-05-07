#include <iostream>

#include <adamcuda.h>

#include "ceres/ceres.h"
#include "ceres/rotation.h"

#undef EIGEN_MPL2_ONLY
#include <FKDerivative.h>

void printSize(torch::Tensor& tensor){
    for(auto i : tensor.sizes()){
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

void AdamCuda::loadAdamData(){
    std::string path = std::string(CMAKE_SOURCE_DIR) + "/data/";

    Eigen::read_binary_special(std::string(path + "mat_J_mu.dat").c_str(), J_mu_);
    allocAndCopyCUDAFloatMat(J_mu_tensor, J_mu_);

    Eigen::read_binary_special(std::string(path + "mat_dJdc.dat").c_str(), dJdc_);
    allocAndCopyCUDAFloatMat(dJdc_tensor, dJdc_);

    Eigen::read_binary_vec(std::string(path + "vec_m_parent.dat").c_str(), m_parent);
    allocAndCopyCUDAIntVec(m_parent_tensor, m_parent);

    std::vector<int> total_vertex;
    Eigen::read_binary_vec(std::string(path + "vec_total_vertex.dat").c_str(), total_vertex);
    if(total_vertex.size() != TotalModel::NUM_FAST_VERTICES) throw std::runtime_error("Fast Vertices wrong size");

    Eigen::MatrixXi parentIndexesMat;
    Eigen::read_binary(std::string(path + "mat_parentIndexesMat.dat").c_str(), parentIndexesMat);
    for(int i=0; i<parentIndexes.size(); i++){
        for(int j=0; j<parentIndexesMat.cols(); j++){
            int output = parentIndexesMat(i, j);
            if(output == -1) continue;
            parentIndexes.at(i).emplace_back(j);
        }
    }
    allocAndCopyCUDAIntMat(parentIndexes_tensor, parentIndexesMat);

    Eigen::MatrixXf m_meanshape;
    Eigen::read_binary_special(std::string(path + "mat_m_meanshape.dat").c_str(), m_meanshape);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> m_meanshape_fast;
    m_meanshape_fast.resize(TotalModel::NUM_FAST_VERTICES*3, 1);
    int i=0;
    for(auto tv_index : total_vertex){
        m_meanshape_fast(i*3 + 0) = (float)m_meanshape(tv_index*3 + 0);
        m_meanshape_fast(i*3 + 1) = (float)m_meanshape(tv_index*3 + 1);
        m_meanshape_fast(i*3 + 2) = (float)m_meanshape(tv_index*3 + 2);
        i++;
    }
    allocAndCopyCUDAFloatMat(m_meanshape_fast_tensor, m_meanshape_fast);

    Eigen::MatrixXf m_shapespace_u;
    Eigen::read_binary_special(std::string(path + "mat_m_shapespace_u.dat").c_str(), m_shapespace_u);
    Eigen::Matrix<float, TotalModel::NUM_FAST_VERTICES*3, TotalModel::NUM_SHAPE_COEFFICIENTS, Eigen::RowMajor> m_shapespace_u_fast;
    i=0;
    for(auto tv_index : total_vertex){
        m_shapespace_u_fast.block(i*3 + 0, 0, 1, TotalModel::NUM_SHAPE_COEFFICIENTS) = m_shapespace_u.block(tv_index*3 + 0, 0, 1, TotalModel::NUM_SHAPE_COEFFICIENTS);
        m_shapespace_u_fast.block(i*3 + 1, 0, 1, TotalModel::NUM_SHAPE_COEFFICIENTS) = m_shapespace_u.block(tv_index*3 + 1, 0, 1, TotalModel::NUM_SHAPE_COEFFICIENTS);
        m_shapespace_u_fast.block(i*3 + 2, 0, 1, TotalModel::NUM_SHAPE_COEFFICIENTS) = m_shapespace_u.block(tv_index*3 + 2, 0, 1, TotalModel::NUM_SHAPE_COEFFICIENTS);
        i++;
    }
    allocAndCopyCUDAFloatMat(m_shapespace_u_fast_tensor, m_shapespace_u_fast);

    Eigen::MatrixXf m_blendW;
    Eigen::read_binary_special(std::string(path + "mat_m_blendW.dat").c_str(), m_blendW);
    Eigen::Matrix<float, TotalModel::NUM_FAST_VERTICES, TotalModel::NUM_JOINTS, Eigen::RowMajor> m_blendW_fast;
    i=0;
    for(auto tv_index : total_vertex){
        m_blendW_fast.block(i, 0, 1, TotalModel::NUM_JOINTS) = m_blendW.block(tv_index, 0, 1, TotalModel::NUM_JOINTS);
        i++;
    }
    allocAndCopyCUDAFloatMat(m_blendW_fast_tensor, m_blendW_fast);

    Eigen::MatrixXf m_small_coco_reg;
    Eigen::read_binary_special(std::string(path + "mat_m_small_coco_reg.dat").c_str(), m_small_coco_reg);
    Eigen::Matrix<float, AdamCuda::NUM_COCO_KP, TotalModel::NUM_FAST_VERTICES, Eigen::RowMajor> m_small_coco_reg_fast;
    i=0;
    for(auto tv_index : total_vertex){
        m_small_coco_reg_fast.block(0,i,AdamCuda::NUM_COCO_KP,1) = m_small_coco_reg.block(0,tv_index,AdamCuda::NUM_COCO_KP,1);
        i++;
    }
    allocAndCopyCUDAFloatMat(m_small_coco_reg_fast_tensor, m_small_coco_reg_fast);

    Eigen::MatrixXf m_dVdFaceEx;
    Eigen::read_binary_special(std::string(path + "mat_m_dVdFaceEx.dat").c_str(), m_dVdFaceEx);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> m_dVdFaceEx_fast;
    m_dVdFaceEx_fast.resize(AdamCuda::NUM_FAST_VERTICES*3, AdamCuda::NUM_EXP_BASIS_COEFFICIENTS);
    i=0;
    for(auto tv_index : total_vertex){
        m_dVdFaceEx_fast.block(i*3 + 0, 0, 1, AdamCuda::NUM_EXP_BASIS_COEFFICIENTS) = m_dVdFaceEx.block(tv_index*3 + 0, 0, 1, AdamCuda::NUM_EXP_BASIS_COEFFICIENTS);
        m_dVdFaceEx_fast.block(i*3 + 1, 0, 1, AdamCuda::NUM_EXP_BASIS_COEFFICIENTS) = m_dVdFaceEx.block(tv_index*3 + 1, 0, 1, AdamCuda::NUM_EXP_BASIS_COEFFICIENTS);
        m_dVdFaceEx_fast.block(i*3 + 2, 0, 1, AdamCuda::NUM_EXP_BASIS_COEFFICIENTS) = m_dVdFaceEx.block(tv_index*3 + 2, 0, 1, AdamCuda::NUM_EXP_BASIS_COEFFICIENTS);
        i++;
    }
    allocAndCopyCUDAFloatMat(m_dVdFaceEx_fast_tensor, m_dVdFaceEx_fast);

    Eigen::MatrixXf posePrior_A;
    Eigen::MatrixXf posePrior_mu;
    Eigen::MatrixXf handPrior_A;
    Eigen::MatrixXf handPrior_mu;
    Eigen::MatrixXf facePrior_A;
    Eigen::MatrixXf facePrior_mu;
    Eigen::read_binary_special(std::string(path + "mat_posePrior_A.dat").c_str(), posePrior_A);
    Eigen::read_binary_special(std::string(path + "mat_posePrior_mu.dat").c_str(), posePrior_mu);
    Eigen::read_binary_special(std::string(path + "mat_handPrior_A.dat").c_str(), handPrior_A);
    Eigen::read_binary_special(std::string(path + "mat_handPrior_mu.dat").c_str(), handPrior_mu);
    Eigen::read_binary_special(std::string(path + "mat_facePrior_A.dat").c_str(), facePrior_A);
    Eigen::read_binary_special(std::string(path + "mat_facePrior_mu.dat").c_str(), facePrior_mu);
    allocAndCopyCUDAFloatMat(facePrior_A_tensor, facePrior_A);
    allocAndCopyCUDAFloatMat(facePrior_mu_tensor, facePrior_mu);

    const int BODY_COUNT = 22;
    const int HAND_COUNT = 40;
    if(BODY_COUNT + HAND_COUNT != AdamCuda::NUM_JOINTS) throw std::runtime_error("Incorrect Joint Size");
    torch::Tensor bodyPrior_A_tensor;
    torch::Tensor bodyPrior_mu_tensor;
    torch::Tensor handPrior_A_tensor;
    torch::Tensor handPrior_mu_tensor;
    allocAndCopyCUDAFloatMat(bodyPrior_A_tensor, posePrior_A);
    allocAndCopyCUDAFloatMat(bodyPrior_mu_tensor, posePrior_mu);
    allocAndCopyCUDAFloatMat(handPrior_A_tensor, handPrior_A);
    allocAndCopyCUDAFloatMat(handPrior_mu_tensor, handPrior_mu);
    bodyPrior_A_tensor = bodyPrior_A_tensor.slice(0, 0, BODY_COUNT*3);
    posePrior_A_tensor = torch::cat({bodyPrior_A_tensor, handPrior_A_tensor}, 0);
    posePrior_mu_tensor = torch::cat({bodyPrior_mu_tensor.slice(0, 0, BODY_COUNT*3), handPrior_mu_tensor.slice(0, BODY_COUNT*3, BODY_COUNT*3 + HAND_COUNT*3)});

    dposepriorlossdP_tensor = wPosePrior * posePrior_A_tensor;
    dfacepriorlossdf_tensor = wFacePrior * facePrior_A_tensor;
}

////const auto CUDA_NUM_THREADS = 512u;
unsigned int getNumberCudaBlocks(const unsigned int totalRequired, const unsigned int numberCudaThreads)
{
    return (totalRequired + numberCudaThreads - 1) / numberCudaThreads+1;
}

const dim3 THREADS_PER_BLOCK{4, 4, 4};

__global__ void k3(const float* a1, const float* a2, const float* a3, float* bPtr)
{
    const auto i = (blockIdx.x * blockDim.x) + threadIdx.x;
    const auto j = (blockIdx.y * blockDim.y) + threadIdx.y;
    bPtr[i*4 + j] = a1[i*3 + j] + a2[i*3 + j] + a3[i*3 + j];
}

__global__ void dCdB(const float* A, float* dCdB, const int ROWS, const int COLS, const int NUM)
{
    // ROW= 20 COL=180
    const auto i = (blockIdx.x * blockDim.x) + threadIdx.x; // 20
    const auto j = (blockIdx.y * blockDim.y) + threadIdx.y; // 180 Blocks

    // True jacobian is [20*3, 180*3] - so its NUM*NUM Blocks
    dCdB[i*COLS*NUM*NUM + 0*COLS*3 + j*3 + 0] = A[i*COLS + j];
    dCdB[i*COLS*NUM*NUM + 1*COLS*3 + j*3 + 1] = A[i*COLS + j];
    dCdB[i*COLS*NUM*NUM + 2*COLS*3 + j*3 + 2] = A[i*COLS + j];
}

__global__ void dVtodVti(const float* blended_transforms, float* dVtodVti, const int NUM_VERTICES)
{
    const auto i = (blockIdx.x * blockDim.x) + threadIdx.x; // 180
    const auto j = (blockIdx.y * blockDim.y) + threadIdx.y; // 9

    const int a_offset = NUM_VERTICES*3*3;
    const int b_offset = NUM_VERTICES*3;

    dVtodVti[i*a_offset + 0*b_offset + i*3 + 0] = blended_transforms[i*12 + 0];
    dVtodVti[i*a_offset + 0*b_offset + i*3 + 1] = blended_transforms[i*12 + 1];
    dVtodVti[i*a_offset + 0*b_offset + i*3 + 2] = blended_transforms[i*12 + 2];

    dVtodVti[i*a_offset + 1*b_offset + i*3 + 0] = blended_transforms[i*12 + 4];
    dVtodVti[i*a_offset + 1*b_offset + i*3 + 1] = blended_transforms[i*12 + 5];
    dVtodVti[i*a_offset + 1*b_offset + i*3 + 2] = blended_transforms[i*12 + 6];

    dVtodVti[i*a_offset + 2*b_offset + i*3 + 0] = blended_transforms[i*12 + 8];
    dVtodVti[i*a_offset + 2*b_offset + i*3 + 1] = blended_transforms[i*12 + 9];
    dVtodVti[i*a_offset + 2*b_offset + i*3 + 2] = blended_transforms[i*12 + 10];
}

__global__ void dVtodTr(const float* Vti, const float* blend_W, float* dVtodTr, const int NUM_JOINTS)
{
    const auto i = (blockIdx.x * blockDim.x) + threadIdx.x; // 180
    const auto j = (blockIdx.y * blockDim.y) + threadIdx.y; // 62
    const auto k = (blockIdx.z * blockDim.z) + threadIdx.z; // 4

    auto w = blend_W[i*NUM_JOINTS + j];
    auto a = Vti[i*4 + k] * w;

    const int a_offset = NUM_JOINTS*12*3;
    const int b_offset = NUM_JOINTS*12;

    dVtodTr[i*a_offset + 0*b_offset + j*12 + 0 + k] = a;
    dVtodTr[i*a_offset + 1*b_offset + j*12 + 4 + k] = a;
    dVtodTr[i*a_offset + 2*b_offset + j*12 + 8 + k] = a;
}


__global__ void OJkernel(
        float* OJ, float* dOJdf, float* dOJdc, float* dOJdP,
        const float* CocoV, const float* Vto, const float* Jn,
        const float* dCocoVdc, const float* dCocoVdP, const float* dVtodf, const float* dVtodc, const float* dVtodP, const float* dJndP, const float* dJndc,
        const int* REG, const int NUM_C, const int NUM_P, const int NUM_F)
{
    const auto i = (blockIdx.x * blockDim.x) + threadIdx.x; // 112
    const auto j = (blockIdx.y * blockDim.y) + threadIdx.y; // 200
    const auto k = (blockIdx.z * blockDim.z) + threadIdx.z; // 3

    const int TYPE = REG[i*2 + 0];
    const int INDEX = REG[i*2 + 1];

    if(TYPE == 0){
        if(j == 0){
            OJ[i*3 + k] = CocoV[INDEX*3 + k];
            OJ[i*3 + k] = CocoV[INDEX*3 + k];
            OJ[i*3 + k] = CocoV[INDEX*3 + k];
        }

        if(j<NUM_C) dOJdc[i*NUM_C*3 + NUM_C*k + j] = dCocoVdc[INDEX*NUM_C*3 + NUM_C*k + j];
        if(j<NUM_P) dOJdP[i*NUM_P*3 + NUM_P*k + j] = dCocoVdP[INDEX*NUM_P*3 + NUM_P*k + j];

    }else if(TYPE == 1){
        if(j == 0){
            OJ[i*3 + k] = Vto[INDEX*3 + k];
            OJ[i*3 + k] = Vto[INDEX*3 + k];
            OJ[i*3 + k] = Vto[INDEX*3 + k];
        }

        if(j<NUM_C) dOJdc[i*NUM_C*3 + NUM_C*k + j] = dVtodc[INDEX*NUM_C*3 + NUM_C*k + j];
        if(j<NUM_P) dOJdP[i*NUM_P*3 + NUM_P*k + j] = dVtodP[INDEX*NUM_P*3 + NUM_P*k + j];
        if(j<NUM_F) dOJdf[i*NUM_F*3 + NUM_F*k + j] = dVtodf[INDEX*NUM_F*3 + NUM_F*k + j];

    }else if(TYPE == 2){
        if(j == 0){
            OJ[i*3 + k] = Jn[INDEX*3 + k];
            OJ[i*3 + k] = Jn[INDEX*3 + k];
            OJ[i*3 + k] = Jn[INDEX*3 + k];
        }

        if(j<NUM_C) dOJdc[i*NUM_C*3 + NUM_C*k + j] = dJndc[INDEX*NUM_C*3 + NUM_C*k + j];
        if(j<NUM_P) dOJdP[i*NUM_P*3 + NUM_P*k + j] = dJndP[INDEX*NUM_P*3 + NUM_P*k + j];
    }
}

__global__ void projRes(float* ploss, float* dplossdOJ, const float* OJ_tensor, const float* proj_truth_tensor, const float* calib_tensor, const int NUM_OJ)
{
    const auto i = (blockIdx.x * blockDim.x) + threadIdx.x; // 112
    const auto j = (blockIdx.y * blockDim.y) + threadIdx.y; // 6 // Make this 6

    const float X = OJ_tensor[i*3 + 0];
    const float Y = OJ_tensor[i*3 + 1];
    const float Z = OJ_tensor[i*3 + 2];
    const float fx = calib_tensor[0];
    const float cx = calib_tensor[2];
    const float fy = calib_tensor[4];
    const float cy = calib_tensor[5];
    const float u = proj_truth_tensor[i*3 + 0];
    const float v = proj_truth_tensor[i*3 + 1];
    const float w = proj_truth_tensor[i*3 + 2];

    // elseif
    if(j == 0){
        ploss[i*2 + 0] = w*(((X*fx + Z*cx)/Z) - u);
    }
    if(j == 1){
        ploss[i*2 + 1] = w*(((Y*fy + Z*cy)/Z) - v);
    }
    if(j == 2){
        dplossdOJ[i*NUM_OJ*3*2 + i*3 + 0] = w*(fx/Z);
    }
    if(j == 3){
        dplossdOJ[i*NUM_OJ*3*2 + i*3 + 2] = w*((cx/Z) - ((X*fx + Z*cx)/(Z*Z)));
    }
    if(j == 4){
        dplossdOJ[i*NUM_OJ*3*2 + NUM_OJ*3 + i*3 + 1] = w*(fy/Z);
    }
    if(j == 5){
        dplossdOJ[i*NUM_OJ*3*2 + NUM_OJ*3 + i*3 + 2] = w*((cy/Z) - ((Y*fy + Z*cy)/(Z*Z)));
    }
}

__global__ void pofRes(const float* pof_truth, const float* OJ, const int* REG_POF_A, const int* REG_POF_B,
                       float* pofloss, float* pof, float* theta, const int NUM_POFS)
{
    const auto i = (blockIdx.x * blockDim.x) + threadIdx.x; // 53
    if(i >= NUM_POFS) return;

    const int IND_A = REG_POF_A[i];
    const int IND_B = REG_POF_B[i];
    const float POFX = OJ[IND_B*3 + 0] - OJ[IND_A*3 + 0];
    const float POFY = OJ[IND_B*3 + 1] - OJ[IND_A*3 + 1];
    const float POFZ = OJ[IND_B*3 + 2] - OJ[IND_A*3 + 2];
    const float th = sqrt(POFX*POFX + POFY*POFY + POFZ*POFZ);

    const float POFNX = POFX/th;
    const float POFNY = POFY/th;
    const float POFNZ = POFZ/th;

    const float pof_truth_x = pof_truth[i*4 + 0];
    const float pof_truth_y = pof_truth[i*4 + 1];
    const float pof_truth_z = pof_truth[i*4 + 2];
    const float w = pof_truth[i*4 + 3];
    const float tht = sqrt(pof_truth_x*pof_truth_x + pof_truth_y*pof_truth_y + pof_truth_z*pof_truth_z);

    pofloss[i*3 + 0] = w*(POFNX - pof_truth_x/tht);
    pofloss[i*3 + 1] = w*(POFNY - pof_truth_y/tht);
    pofloss[i*3 + 2] = w*(POFNZ - pof_truth_z/tht);

    theta[i] = th;

    pof[i*3 + 0] = POFNX;
    pof[i*3 + 1] = POFNY;
    pof[i*3 + 2] = POFNZ;

    if(w==0)
    {
        pofloss[i*3 + 0] = 0;
        pofloss[i*3 + 1] = 0;
        pofloss[i*3 + 2] = 0;

        theta[i] = 0;

        pof[i*3 + 0] = 0;
        pof[i*3 + 1] = 0;
        pof[i*3 + 2] = 0;
    }
}

__global__ void pofGrad(const float* pof_truth, const float* pof, const float* theta, const int* REG_POF_A, const int* REG_POF_B, float* dpoflossdpof, float* dpofdOJ, const int NUM_POFS, const int NUM_OJ)
{
    const auto i = (blockIdx.x * blockDim.x) + threadIdx.x; // 53
    const auto j = (blockIdx.y * blockDim.y) + threadIdx.y; // 10
    if(i >= NUM_POFS) return;

    float inv_theta = 1/theta[i];
    const float X = pof[i*3 + 0];
    const float Y = pof[i*3 + 1];
    const float Z = pof[i*3 + 2];
    const float w = pof_truth[i*4 + 3];

    if(w == 0)
    {
        inv_theta = 0;
    }

    if(j == 0){
        dpoflossdpof[i*NUM_POFS*3*3 + (0*NUM_POFS*3 + 0) + i*3] = w*(X*X*(-inv_theta) + inv_theta);
    }
    if(j == 1){
        dpoflossdpof[i*NUM_POFS*3*3 + (0*NUM_POFS*3 + 1) + i*3] = w*(X*Y*(-inv_theta));
    }
    if(j == 2){
        dpoflossdpof[i*NUM_POFS*3*3 + (0*NUM_POFS*3 + 2) + i*3] = w*(X*Z*(-inv_theta));
    }

    if(j == 3){
        dpoflossdpof[i*NUM_POFS*3*3 + (1*NUM_POFS*3 + 0) + i*3] = w*(Y*X*(-inv_theta));
    }
    if(j == 4){
        dpoflossdpof[i*NUM_POFS*3*3 + (1*NUM_POFS*3 + 1) + i*3] = w*(Y*Y*(-inv_theta) + inv_theta);
    }
    if(j == 5){
        dpoflossdpof[i*NUM_POFS*3*3 + (1*NUM_POFS*3 + 2) + i*3] = w*(Y*Z*(-inv_theta));
    }

    if(j == 6){
        dpoflossdpof[i*NUM_POFS*3*3 + (2*NUM_POFS*3 + 0) + i*3] = w*(Z*X*(-inv_theta));
    }
    if(j == 7){
        dpoflossdpof[i*NUM_POFS*3*3 + (2*NUM_POFS*3 + 1) + i*3] = w*(Z*Y*(-inv_theta));
    }
    if(j == 8){
        dpoflossdpof[i*NUM_POFS*3*3 + (2*NUM_POFS*3 + 2) + i*3] = w*(Z*Z*(-inv_theta) + inv_theta);
    }

    if(j == 9){
        const int IND_A = REG_POF_A[i];
        const int IND_B = REG_POF_B[i];

        dpofdOJ[i*NUM_OJ*3*3 + (0*NUM_OJ*3 + 0) + IND_A*3] = -1;
        dpofdOJ[i*NUM_OJ*3*3 + (1*NUM_OJ*3 + 1) + IND_A*3] = -1;
        dpofdOJ[i*NUM_OJ*3*3 + (2*NUM_OJ*3 + 2) + IND_A*3] = -1;

        dpofdOJ[i*NUM_OJ*3*3 + (0*NUM_OJ*3 + 0) + IND_B*3] = 1;
        dpofdOJ[i*NUM_OJ*3*3 + (1*NUM_OJ*3 + 1) + IND_B*3] = 1;
        dpofdOJ[i*NUM_OJ*3*3 + (2*NUM_OJ*3 + 2) + IND_B*3] = 1;
    }

}

__global__ void updatePofWeight(float* pof_truth, float w, const int* range)
{
    const auto i = (blockIdx.x * blockDim.x) + threadIdx.x; // 53
    float startRange = range[0];
    float endRange = range[1];
    if(i >= startRange && i <= endRange){
        float x = pof_truth[i*4 + 0];
        float y = pof_truth[i*4 + 1];
        float z = pof_truth[i*4 + 2];
        if(x==0 && y==0 && z==0) return;
        pof_truth[i*4 + 3] = w;
    }
}

__global__ void updateProjWeight(float* proj_truth, float w, const int* range)
{
    const auto i = (blockIdx.x * blockDim.x) + threadIdx.x; // 53
    float startRange = range[0];
    float endRange = range[1];
    if(i >= startRange && i <= endRange){
        float x = proj_truth[i*3 + 0];
        float y = proj_truth[i*3 + 1];
        if(x==0 && y==0) return;
        proj_truth[i*3 + 2] = w;
    }
}

void FKFunction(AdamCuda::FKData& fkData, const Eigen::MatrixXf& eulerAngles, const Eigen::MatrixXf& joints,
                const std::vector<int>& m_parent, const std::array<std::vector<int>, AdamCuda::NUM_JOINTS>& mParentIndexes,
                bool jac=false){
    fkData.zeroOutJac(jac); // Takes 0.1ms with jacobian :(

    Eigen::Map<Eigen::Matrix<float, 4 * AdamCuda::NUM_JOINTS * 3, AdamCuda::NUM_JOINTS * 3, Eigen::RowMajor> > dTrdP(fkData.dTrdP_ptr);
    Eigen::Map<Eigen::Matrix<float, AdamCuda::NUM_JOINTS * 3, AdamCuda::NUM_JOINTS * 3, Eigen::RowMajor> > dJdP(fkData.dTrdP_ptr + AdamCuda::NUM_JOINTS * AdamCuda::NUM_JOINTS * 36);

    Eigen::Map<Eigen::Matrix<float, 4 * AdamCuda::NUM_JOINTS * 3, AdamCuda::NUM_JOINTS * 3, Eigen::RowMajor> > dTrdJ(fkData.dTrdJ_ptr);
    Eigen::Map<Eigen::Matrix<float, AdamCuda::NUM_JOINTS * 3, AdamCuda::NUM_JOINTS * 3, Eigen::RowMajor> > dJdJ(fkData.dTrdJ_ptr + AdamCuda::NUM_JOINTS * AdamCuda::NUM_JOINTS * 36);

    Eigen::Map<Eigen::Matrix<float, 3 * AdamCuda::NUM_JOINTS, 4, Eigen::RowMajor> > outT(fkData.transforms_joint_ptr);
    Eigen::Map<Eigen::Matrix<float, AdamCuda::NUM_JOINTS, 3, Eigen::RowMajor> > outJoint(fkData.transforms_joint_ptr + 3 * AdamCuda::NUM_JOINTS * 4);
    Eigen::Map<const Eigen::Matrix<float, AdamCuda::NUM_JOINTS, 3, Eigen::RowMajor>>J0(joints.data());

    auto& R = fkData.R;
    auto& MR = fkData.MR;
    auto& Mt = fkData.Mt;
    const auto& pose = eulerAngles.data();
    auto& dMRdP = fkData.dMRdP;
    auto& dMtdJ = fkData.dMtdJ;
    auto& dMtdP = fkData.dMtdP;
    auto& offset = fkData.offset;
    auto& dRdP = fkData.dRdP;
    auto& dtdJ = fkData.dtdJ;
    auto& dtdJ2 = fkData.dtdJ2;
    auto& dtdP = fkData.dtdP;

    ceres::AngleAxisToRotationMatrix(pose, R.data());
    outJoint.row(0) = J0.row(0);
    MR.at(0) = R;
    Mt.at(0) = J0.row(0).transpose();
    outT.block(0, 0, 3, 3) = MR[0];
    outT.block(0, 3, 3, 1) = Mt[0];

    // May not need Fill and Zero op

    if(jac)
    {
        AngleAxisToRotationMatrix_Derivative(pose, dMRdP.at(0).data(), 0);
        std::fill(dMtdP[0].data(), dMtdP[0].data() + 9 * AdamCuda::NUM_JOINTS, 0.0); // dMtdP.at(0).setZero();
        std::fill(dMtdJ[0].data(), dMtdJ[0].data() + 9 * AdamCuda::NUM_JOINTS, 0.0); // dMtdJ.at(0).setZero();
        dMtdJ.at(0).block(0, 0, 3, 3).setIdentity();
        std::copy(dMtdP[0].data(), dMtdP[0].data() + 9 * AdamCuda::NUM_JOINTS, dJdP.data()); // dJdP.block(0, 0, 3, TotalModel::NUM_JOINTS * 3) = dMtdP[0];
        std::copy(dMtdJ[0].data(), dMtdJ[0].data() + 9 * AdamCuda::NUM_JOINTS, dJdJ.data()); // dJdJ.block(0, 0, 3, TotalModel::NUM_JOINTS * 3) = dMtdJ[0];
    }

    for (int idj = 1; idj < AdamCuda::NUM_JOINTS; idj++)
    {
        const int ipar = m_parent[idj];
        const auto baseIndex = idj * 3;
        float angles[3] = {pose[baseIndex], pose[baseIndex+1], pose[baseIndex+2]};

        //Freezing joints here  //////////////////////////////////////////////////////
        if (idj == 10 || idj == 11) //foot ends
        {
            angles[0] = 0;
            angles[1] = 0;
            angles[2] = 0;
        }
        else
        {
            ceres::AngleAxisToRotationMatrix(angles, R.data());
        }
        MR.at(idj) = MR.at(ipar) * R;
        offset = (J0.row(idj) - J0.row(ipar)).transpose();
        Mt.at(idj) = Mt.at(ipar) + MR.at(ipar) * offset;
        outJoint.row(idj) = Mt.at(idj).transpose();
        outT.block(0, 0, 3, 3) = MR[0];
        outT.block(0, 3, 3, 1) = Mt[0];

        if(jac)
        {
            AngleAxisToRotationMatrix_Derivative(angles, dRdP.data(), idj);

            if (idj == 10 || idj == 11) //foot ends
                dMRdP.at(idj) = dMRdP.at(ipar);
            else
            {
                // Sparse derivative
                SparseProductDerivative(MR.at(ipar).data(), dMRdP.at(ipar).data(), R.data(), dRdP.data(), idj, mParentIndexes.at(idj), dMRdP.at(idj).data(), TotalModel::NUM_JOINTS * 3);
                // // Slower but equivalent - Dense derivative
                // Product_Derivative(MR.at(ipar).data(), dMRdP.at(ipar).data(), R.data(), dRdP.data(), dMRdP.at(idj).data()); // Compute the product of matrix multiplication
            }

            SparseProductDerivative(dMRdP.at(ipar).data(), offset.data(), mParentIndexes.at(ipar), dMtdP.at(idj).data(), TotalModel::NUM_JOINTS * 3);
            // the following line is equivalent to dMtdP.at(idj) = dMtdP.at(idj) + dMtdP.at(ipar);
            SparseAdd(dMtdP.at(ipar).data(), mParentIndexes.at(ipar), dMtdP.at(idj).data());

            std::fill(dtdJ.data(), dtdJ.data() + 9 * AdamCuda::NUM_JOINTS, 0.0); // dtdJ.setZero();
            // the following two lines are equiv to: dtdJ.block(0, 3 * idj, 3, 3).setIdentity(); dtdJ.block(0, 3 * ipar, 3, 3) -= Matrix<double, 3, 3>::Identity(); // derivative of offset wrt J
            dtdJ.data()[3 * idj] = 1; dtdJ.data()[3 * idj + 3 * AdamCuda::NUM_JOINTS + 1] = 1; dtdJ.data()[3 * idj + 6 * TotalModel::NUM_JOINTS + 2] = 1;
            dtdJ.data()[3 * ipar] = -1; dtdJ.data()[3 * ipar + 3 * AdamCuda::NUM_JOINTS + 1] = -1; dtdJ.data()[3 * ipar + 6 * TotalModel::NUM_JOINTS + 2] = -1;

            // the following line is equivalent to Product_Derivative(MR.at(ipar).data(), NULL, offset.data(), dtdJ.data(), dMtdJ.at(idj).data(), 1); // dA_data is NULL since rotation is not related to joint
            SparseProductDerivativeConstA(MR.at(ipar).data(), dtdJ.data(), mParentIndexes.at(idj), dMtdJ.at(idj).data(), TotalModel::NUM_JOINTS*3);

            // the following line is equivalent to dMtdJ.at(idj) = dMtdJ.at(idj) + dMtdJ.at(ipar);
            SparseAdd(dMtdJ.at(ipar).data(), mParentIndexes.at(idj), dMtdJ.at(idj).data());

            std::copy(dMtdP[idj].data(), dMtdP[idj].data() + 9 * AdamCuda::NUM_JOINTS, dJdP.data() + 9 * idj * TotalModel::NUM_JOINTS); // dJdP.block(3 * idj, 0, 3, TotalModel::NUM_JOINTS * 3) = dMtdP[idj];
            std::copy(dMtdJ[idj].data(), dMtdJ[idj].data() + 9 * AdamCuda::NUM_JOINTS, dJdJ.data() + 9 * idj * TotalModel::NUM_JOINTS); // dJdJ.block(3 * idj, 0, 3, TotalModel::NUM_JOINTS * 3) = dMtdJ[idj];
        }
    }

    for (int idj = 0; idj < AdamCuda::NUM_JOINTS; idj++) {
        offset = J0.row(idj).transpose();
        Mt.at(idj) -= MR.at(idj) * offset;

        outT.block(3 * idj, 0, 3, 3) = MR.at(idj);
        outT.block(3 * idj, 3, 3, 1) = Mt.at(idj);

        if (jac)
        {
            // The following line is equivalent to Product_Derivative(MR.at(idj).data(), dMRdP.at(idj).data(), offset.data(), NULL, dtdP.data(), 1);
            SparseProductDerivative(dMRdP.at(idj).data(), offset.data(), mParentIndexes.at(idj), dtdP.data());
            // The following line is equivalent to dMtdP.at(idj) -= dtdP;
            SparseSubtract(dtdP.data(), mParentIndexes.at(idj), dMtdP.at(idj).data());

            std::fill(dtdJ.data(), dtdJ.data() + 9 * TotalModel::NUM_JOINTS, 0.0); // dtdJ.setZero();
            // The follwing line is equivalent to dtdJ.block(0, 3 * idj, 3, 3).setIdentity();
            dtdJ.data()[3 * idj] = 1; dtdJ.data()[3 * idj + 3 * TotalModel::NUM_JOINTS + 1] = 1; dtdJ.data()[3 * idj + 6 * TotalModel::NUM_JOINTS + 2] = 1;
            // The following line is equivalent to Product_Derivative(MR.at(idj).data(), NULL, offset.data(), dtdJ.data(), dtdJ2.data(), 1);
            SparseProductDerivativeConstA(MR.at(idj).data(), dtdJ.data(), mParentIndexes.at(idj), dtdJ2.data());
            // The following line is equivalent to dMtdJ.at(idj) -= dtdJ2;
            SparseSubtract(dtdJ2.data(), mParentIndexes.at(idj), dMtdJ.at(idj).data());

            // The following lines are copying jacobian from dMRdP and dMtdP to dTrdP, equivalent to
            // dTrdP.block(12 * idj + 0, 0, 3, TotalModel::NUM_JOINTS * 3) = dMRdP.at(idj).block(0, 0, 3, TotalModel::NUM_JOINTS * 3);
            // dTrdP.block(12 * idj + 4, 0, 3, TotalModel::NUM_JOINTS * 3) = dMRdP.at(idj).block(3, 0, 3, TotalModel::NUM_JOINTS * 3);
            // dTrdP.block(12 * idj + 8, 0, 3, TotalModel::NUM_JOINTS * 3) = dMRdP.at(idj).block(6, 0, 3, TotalModel::NUM_JOINTS * 3);
            // dTrdP.block(12 * idj + 3, 0, 1, TotalModel::NUM_JOINTS * 3) = dMtdP.at(idj).block(0, 0, 1, TotalModel::NUM_JOINTS * 3);
            // dTrdP.block(12 * idj + 7, 0, 1, TotalModel::NUM_JOINTS * 3) = dMtdP.at(idj).block(1, 0, 1, TotalModel::NUM_JOINTS * 3);
            // dTrdP.block(12 * idj + 11, 0, 1, TotalModel::NUM_JOINTS * 3) = dMtdP.at(idj).block(2, 0, 1, TotalModel::NUM_JOINTS * 3);
            std::copy(dMRdP.at(idj).data(), dMRdP.at(idj).data() + 9 * TotalModel::NUM_JOINTS, dTrdP.data() + 12 * idj * TotalModel::NUM_JOINTS * 3);
            std::copy(dMtdP.at(idj).data(), dMtdP.at(idj).data() + 3 * TotalModel::NUM_JOINTS, dTrdP.data() + (12 * idj + 3) * TotalModel::NUM_JOINTS * 3);
            std::copy(dMRdP.at(idj).data() + 9 * TotalModel::NUM_JOINTS, dMRdP.at(idj).data() + 18 * TotalModel::NUM_JOINTS,
                dTrdP.data() + (12 * idj + 4)* TotalModel::NUM_JOINTS * 3);
            std::copy(dMtdP.at(idj).data() + 3 * TotalModel::NUM_JOINTS, dMtdP.at(idj).data() + 6 * TotalModel::NUM_JOINTS,
                dTrdP.data() + (12 * idj + 7) * TotalModel::NUM_JOINTS * 3);
            std::copy(dMRdP.at(idj).data() + 18 * TotalModel::NUM_JOINTS, dMRdP.at(idj).data() + 27 * TotalModel::NUM_JOINTS,
                dTrdP.data() + (12 * idj + 8)* TotalModel::NUM_JOINTS * 3);
            std::copy(dMtdP.at(idj).data() + 6 * TotalModel::NUM_JOINTS, dMtdP.at(idj).data() + 9 * TotalModel::NUM_JOINTS,
                dTrdP.data() + (12 * idj + 11) * TotalModel::NUM_JOINTS * 3);

            // The following lines are copying jacobian from and dMtdJ to dTrdJ, equivalent to
            // dTrdJ.block(12 * idj + 3, 0, 1, TotalModel::NUM_JOINTS * 3) = dMtdJ.at(idj).block(0, 0, 1, TotalModel::NUM_JOINTS * 3);
            // dTrdJ.block(12 * idj + 7, 0, 1, TotalModel::NUM_JOINTS * 3) = dMtdJ.at(idj).block(1, 0, 1, TotalModel::NUM_JOINTS * 3);
            // dTrdJ.block(12 * idj + 11, 0, 1, TotalModel::NUM_JOINTS * 3) = dMtdJ.at(idj).block(2, 0, 1, TotalModel::NUM_JOINTS * 3);
            std::copy(dMtdJ.at(idj).data(), dMtdJ.at(idj).data() + 3 * TotalModel::NUM_JOINTS, dTrdJ.data() + (12 * idj + 3) * TotalModel::NUM_JOINTS * 3);
            std::copy(dMtdJ.at(idj).data() + 3 * TotalModel::NUM_JOINTS, dMtdJ.at(idj).data() + 6 * TotalModel::NUM_JOINTS,
                dTrdJ.data() + (12 * idj + 7) * TotalModel::NUM_JOINTS * 3);
            std::copy(dMtdJ.at(idj).data() + 6 * TotalModel::NUM_JOINTS, dMtdJ.at(idj).data() + 9 * TotalModel::NUM_JOINTS,
                dTrdJ.data() + (12 * idj + 11) * TotalModel::NUM_JOINTS * 3);
        }
    }

}

inline
cudaError_t checkCuda(cudaError_t result)
{
//#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n",
            cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
//#endif
  return result;
}

void AdamCuda::setPofWeightRange(const std::vector<int>& range, float w){
    torch::Tensor range_tensor;
    allocAndCopyCUDAIntVec(range_tensor, range);
    updatePofWeight<<<dim3(NUM_POFS), dim3(1)>>>((float*)pof_truth_tensor.data_ptr(), w, (int*)range_tensor.data_ptr());
    cudaDeviceSynchronize();
}

void AdamCuda::setProjWeightRange(const std::vector<int>& range, float w){
    torch::Tensor range_tensor;
    allocAndCopyCUDAIntVec(range_tensor, range);
    updateProjWeight<<<dim3(NUM_OJ), dim3(1)>>>((float*)proj_truth_tensor.data_ptr(), w, (int*)range_tensor.data_ptr());
    cudaDeviceSynchronize();
}

void AdamCuda::run(const torch::Tensor& t_tensor, const torch::Tensor& eulers_tensor, const torch::Tensor& bodyshape_tensor, const torch::Tensor& faceshape_tensor, bool jac)
{
    //startTime();

    // Copy Bodyshape and Eulers to CPU
    cudaMemcpyAsync(eulers_pinned, eulers_tensor.data_ptr(), NUM_POSE_PARAMETERS * sizeof(float),
               cudaMemcpyDeviceToHost, streams[0].stream());
    cudaMemcpyAsync(bodyshape_pinned, bodyshape_tensor.data_ptr(), NUM_SHAPE_COEFFICIENTS * sizeof(float),
               cudaMemcpyDeviceToHost, streams[0].stream());
    cudaStreamSynchronize(streams[0]);
    Eigen::Map<Eigen::Matrix<float, NUM_SHAPE_COEFFICIENTS, 1> > bodyshape_mat(bodyshape_pinned);
    Eigen::Map<Eigen::Matrix<float, NUM_JOINTS, 3, Eigen::RowMajor> > eulers_mat(eulers_pinned);

    // Forward Kinematics on CPU
    Eigen::MatrixXf J_vec = J_mu_ + dJdc_ * bodyshape_mat;
    FKFunction(fkData, eulers_mat, J_vec, m_parent, parentIndexes, jac);

    // Start FK Gradient Upload
    cudaMemcpyAsync(transforms_tensor.data_ptr(), fkData.transforms_joint_ptr,
               transforms_tensor.size(0)*transforms_tensor.size(1)*sizeof(float),
                    cudaMemcpyHostToDevice, streams[0].stream());
    cudaMemcpyAsync(jn_tensor.data_ptr(), fkData.transforms_joint_ptr + (3 * NUM_JOINTS * 4),
               jn_tensor.size(0)*jn_tensor.size(1)*sizeof(float),
                    cudaMemcpyHostToDevice, streams[0].stream());
    if(jac)
    {
        cudaMemcpyAsync(dTrdP_tensor.data_ptr(), fkData.dTrdP_ptr,
                   dTrdP_tensor.size(0)*dTrdP_tensor.size(1)*sizeof(float),
                        cudaMemcpyHostToDevice, streams[1].stream());
        cudaMemcpyAsync(dJndP_tensor.data_ptr(), fkData.dTrdP_ptr + (12*NUM_JOINTS * 3*NUM_JOINTS),
                   dJndP_tensor.size(0)*dJndP_tensor.size(1)*sizeof(float),
                        cudaMemcpyHostToDevice, streams[1].stream());
        cudaMemcpyAsync(dTrdJ_tensor.data_ptr(), fkData.dTrdJ_ptr,
                   dTrdJ_tensor.size(0)*dTrdJ_tensor.size(1)*sizeof(float),
                        cudaMemcpyHostToDevice);
        cudaMemcpyAsync(dJndJ_tensor.data_ptr(), fkData.dTrdJ_ptr + (12*NUM_JOINTS * 3*NUM_JOINTS),
                   dJndJ_tensor.size(0)*dJndJ_tensor.size(1)*sizeof(float),
                        cudaMemcpyHostToDevice);
    }

    // Compute Vertices from Parameters
    torch::mm_out(vt_temp_body, m_shapespace_u_fast_tensor, bodyshape_tensor);
    torch::mm_out(vt_temp_face, m_dVdFaceEx_fast_tensor, faceshape_tensor);
    cudaDeviceSynchronize();
    k3<<<dim3(NUM_FAST_VERTICES/3,3), dim3(3,1)>>>((float*)m_meanshape_fast_tensor.data_ptr(), (float*)vt_temp_body.data_ptr(), (float*)vt_temp_face.data_ptr(), (float*)Vt_in_homo.data_ptr());
    cudaDeviceSynchronize();

    // Compute Gradients for Transform/Joints to Bodyshape
    if(jac)
    {
        torch::mm_out(dTrdc_tensor, dTrdJ_tensor, dJdc_tensor);
        torch::mm_out(dJndc_tensor, dJndJ_tensor, dJdc_tensor);
        dVtodTr<<<dim3(NUM_FAST_VERTICES/2, NUM_JOINTS/2, 4), dim3(2,2,1), 0, streams[0].stream()>>>((float*)Vt_in_homo.data_ptr(), (float*)m_blendW_fast_tensor.data_ptr(), (float*)dVtodTr_tensor.data_ptr(), NUM_JOINTS);
    }

    // LBS Function
    torch::mm_out(blended_transforms, m_blendW_fast_tensor, transforms_tensor);
    blended_transforms.resize_({NUM_FAST_VERTICES, 3, 4});
    torch::matmul_out(Vt_out_homo, blended_transforms, Vt_in_homo);
    Vt_out_homo.resize_({NUM_FAST_VERTICES, 3});
    Vto_tensor = Vt_out_homo.slice(1,0,3);
    cudaDeviceSynchronize();

    // Add Translation to Joints and Vertices (Inefficient add sucks)
    torch::add_out(jn_tensor, jn_tensor, t_tensor);
    torch::add_out(Vto_tensor, Vto_tensor, t_tensor);

    // LBS Gradient
    if(jac)
    {
        dVtodVti<<<dim3(NUM_FAST_VERTICES), dim3(1), 0>>>((float*)blended_transforms.data_ptr(), (float*)dVtodVti_tensor.data_ptr(), NUM_FAST_VERTICES);
        cudaDeviceSynchronize();
        torch::Tensor& dVtidc_tensor = m_shapespace_u_fast_tensor;
        torch::Tensor& dVtidf_tensor = m_dVdFaceEx_fast_tensor;
        {
            at::cuda::CUDAStreamGuard guard1(streams[0]);
            torch::mm_out(dVtodP_tensor, dVtodTr_tensor, dTrdP_tensor);
            at::cuda::CUDAStreamGuard guard2(streams[1]);
            dVtodc_tensor = torch::mm(dVtodTr_tensor, dTrdc_tensor) + torch::mm(dVtodVti_tensor, dVtidc_tensor); // THIS IS NOT EFFICIENT
            at::cuda::CUDAStreamGuard guard3(streams[2]);
            torch::mm_out(dVtodf_tensor, dVtodVti_tensor, dVtidf_tensor);
        }
    }

    // COCO Regressor
    torch::mm_out(CocoV_tensor, m_small_coco_reg_fast_tensor, Vto_tensor);

    // COCO Regressor Gradient
    if(jac)
    {
        cudaDeviceSynchronize();
        dCdB<<<dim3(NUM_COCO_KP,NUM_FAST_VERTICES), dim3(1,1)>>>((float*)m_small_coco_reg_fast_tensor.data_ptr(), (float*)dCocoVdVto_tensor.data_ptr(), NUM_COCO_KP, NUM_FAST_VERTICES, 3);
        cudaDeviceSynchronize();
        torch::mm_out(dCocoVdc_tensor, dCocoVdVto_tensor, dVtodc_tensor);
        torch::mm_out(dCocoVdP_tensor, dCocoVdVto_tensor, dVtodP_tensor);
        cudaDeviceSynchronize();
    }

    // Reorder Joints/Verticies/CocoPoints into new datastructure OJ
    OJkernel<<<dim3(NUM_OJ/4, NUM_EXP_BASIS_COEFFICIENTS/4, 3), dim3(4,4,1)>>>(
                                      (float*)OJ_tensor.data_ptr(), (float*)dOJdf_tensor.data_ptr(), (float*)dOJdc_tensor.data_ptr(), (float*)dOJdP_tensor.data_ptr(),
                                      (float*)CocoV_tensor.data_ptr(), (float*)Vto_tensor.data_ptr(), (float*)jn_tensor.data_ptr(),
                                      (float*)dCocoVdc_tensor.data_ptr(), (float*)dCocoVdP_tensor.data_ptr(), (float*)dVtodf_tensor.data_ptr(), (float*)dVtodc_tensor.data_ptr(), (float*)dVtodP_tensor.data_ptr(), (float*)dJndP_tensor.data_ptr(), (float*)dJndc_tensor.data_ptr(),
                                      (int*)REG_POINTS_tensor.data_ptr(),
                                      NUM_SHAPE_COEFFICIENTS, NUM_POSE_PARAMETERS, NUM_EXP_BASIS_COEFFICIENTS);
    cudaDeviceSynchronize();

    // Proj Loss Function and Gradient
    projRes<<<dim3(NUM_OJ/4, 6/3), dim3(4,3)>>>((float*)ploss_tensor.data_ptr(), (float*)dplossdOJ_tensor.data_ptr(), (float*)OJ_tensor.data_ptr(), (float*)proj_truth_tensor.data_ptr(), (float*)calib_tensor.data_ptr(), NUM_OJ);
    cudaDeviceSynchronize();

    // Proj Loss Sub-Gradients
    if(jac)
    {
        {
            torch::mm_out(dplossdt_tensor, dplossdOJ_tensor, dOJdt_tensor);
            at::cuda::CUDAStreamGuard g0(streams[0]);
            torch::mm_out(dplossdP_tensor, dplossdOJ_tensor, dOJdP_tensor);
            at::cuda::CUDAStreamGuard g1(streams[1]);
            torch::mm_out(dplossdc_tensor, dplossdOJ_tensor, dOJdc_tensor);
            at::cuda::CUDAStreamGuard g2(streams[2]);
            torch::mm_out(dplossdf_tensor, dplossdOJ_tensor, dOJdf_tensor);
        }
    }

    // POF Loss Function
    pofRes<<<dim3(54/3), dim3(3)>>>((float*)pof_truth_tensor.data_ptr(), (float*)OJ_tensor.data_ptr(), (int*)REG_POF_A_tensor.data_ptr(), (int*)REG_POF_B_tensor.data_ptr(),
           (float*)pofloss_tensor.data_ptr(), (float*)pof_tensor.data_ptr(), (float*)theta_tensor.data_ptr(), NUM_POFS);
    cudaDeviceSynchronize();

    // POF Loss Gradient and Sub-Gradients
    if(jac)
    {
        pofGrad<<<dim3(54/3, 10/5), dim3(3,5)>>>((float*)pof_truth_tensor.data_ptr(), (float*)pof_tensor.data_ptr(), (float*)theta_tensor.data_ptr(), (int*)REG_POF_A_tensor.data_ptr(), (int*)REG_POF_B_tensor.data_ptr(), (float*)dpoflossdpof_tensor.data_ptr(), (float*)dpofdOJ_tensor.data_ptr(), NUM_POFS, NUM_OJ);
        cudaDeviceSynchronize();
        torch::mm_out(dpoflossdOJ_tensor, dpoflossdpof_tensor, dpofdOJ_tensor);
        cudaDeviceSynchronize();

        {
            torch::mm_out(dpoflossdt_tensor, dpoflossdOJ_tensor, dOJdt_tensor);
            at::cuda::CUDAStreamGuard g0(streams[0]);
            torch::mm_out(dpoflossdP_tensor, dpoflossdOJ_tensor, dOJdP_tensor);
            at::cuda::CUDAStreamGuard g1(streams[1]);
            torch::mm_out(dpoflossdc_tensor, dpoflossdOJ_tensor, dOJdc_tensor);
            at::cuda::CUDAStreamGuard g2(streams[2]);
            torch::mm_out(dpoflossdf_tensor, dpoflossdOJ_tensor, dOJdf_tensor);
        }
        cudaDeviceSynchronize();
    }

    // Priors (Inefficient)
    //posepriorloss_tensor - NUM_POSE_PARAMETERS
    torch::Tensor eulers_resized = eulers_tensor.reshape({NUM_POSE_PARAMETERS, 1});
    torch::mm_out(posepriorloss_tensor, posePrior_A_tensor, wPosePrior*(eulers_resized - posePrior_mu_tensor));
    torch::mm_out(facepriorloss_tensor, facePrior_A_tensor, wFacePrior*(faceshape_tensor - facePrior_mu_tensor));
    shapecoeffloss_tensor = wCoeffRg * bodyshape_tensor;

    // Combine Everything
    ploss_tensor.resize_({NUM_OJ*2});
    pofloss_tensor.resize_({NUM_POFS*3});
    torch::cat_out(r_tensor, {ploss_tensor, pofloss_tensor}, 0);
    r_tensor.resize_({NUM_OJ*2 + NUM_POFS*3, 1});
    if(jac)
    {
        torch::cat_out(drdt_tensor, {dplossdt_tensor, dpoflossdt_tensor}, 0);
        torch::cat_out(drdP_tensor, {dplossdP_tensor, dpoflossdP_tensor}, 0);
        torch::cat_out(drdc_tensor, {dplossdc_tensor, dpoflossdc_tensor}, 0);
        torch::cat_out(drdf_tensor, {dplossdf_tensor, dpoflossdf_tensor}, 0);
    }

    //endTime();
}
