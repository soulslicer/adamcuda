#ifndef ADAMCUDA_H_INCLUDED
#define ADAMCUDA_H_INCLUDED

#include <iostream>

#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#undef __CUDACC__
#include <Eigen/Core>

#include <totalmodel.h>


const auto CUDA_NUM_THREADS = 512u;
unsigned int getNumberCudaBlocks(const unsigned int totalRequired, const unsigned int numberCudaThreads = CUDA_NUM_THREADS);


namespace Eigen{
template<class T>
void write_binary_vec(const char* filename, const std::vector<T>& vector){
    std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
    size_t size = vector.size();
    out.write((char*) (&size), sizeof(size_t));
    out.write((char*) vector.data(), vector.size()*sizeof(T) );
    out.close();
}
template<class T>
void read_binary_vec(const char* filename, std::vector<T>& vector){
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    size_t size;
    in.read((char*) (&size),sizeof(size_t));
    vector.resize(size);
    in.read( (char *) vector.data() , vector.size()*sizeof(T));
    in.close();
}

template<class Matrix>
void write_binary(const char* filename, const Matrix& matrix){
    std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
    typename Matrix::Index rows=matrix.rows(), cols=matrix.cols();
    out.write((char*) (&rows), sizeof(typename Matrix::Index));
    out.write((char*) (&cols), sizeof(typename Matrix::Index));
    out.write((char*) matrix.data(), rows*cols*sizeof(typename Matrix::Scalar) );
    out.close();
}
template<class Matrix>
void read_binary(const char* filename, Matrix& matrix){
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    typename Matrix::Index rows=0, cols=0;
    in.read((char*) (&rows),sizeof(typename Matrix::Index));
    in.read((char*) (&cols),sizeof(typename Matrix::Index));
    matrix.resize(rows, cols);
    in.read( (char *) matrix.data() , rows*cols*sizeof(typename Matrix::Scalar) );
    in.close();
}
template<class Matrix>
void read_binary_special(const char* filename, Matrix& matrix){
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    typename Matrix::Index rows=0, cols=0;
    in.read((char*) (&rows),sizeof(typename Matrix::Index));
    in.read((char*) (&cols),sizeof(typename Matrix::Index));
    matrix.resize(rows, cols);
    for(int r=0; r<rows; r++){
        for(int c=0; c<cols; c++){
            float num = 0;
            in.read(reinterpret_cast<char*>( &num ), sizeof(typename Matrix::Scalar));
            matrix(r,c) = num;
        }
    }
    in.close();
}
} // Eigen::

class TestClass{
public:
    TestClass(){

    }
};

class AdamCuda{
public:
    static const int NUM_VERTICES = TotalModel::NUM_VERTICES;
    static const int NUM_FACES = TotalModel::NUM_FACES;// 111462;
    static const int NUM_FAST_VERTICES = 180;
    static const int NUM_COCO_KP = 20;

    static const int NUM_JOINTS = TotalModel::NUM_JOINTS;		//(SMPL-hands)22 + lhand20 + rhand20
    static const int NUM_POSE_PARAMETERS = NUM_JOINTS * 3;
    static const int NUM_SHAPE_COEFFICIENTS = TotalModel::NUM_SHAPE_COEFFICIENTS;
    static const int NUM_EXP_BASIS_COEFFICIENTS = TotalModel::NUM_EXP_BASIS_COEFFICIENTS; //Facial expression

    const std::vector<int> REG_COCO_POINTS = {0,1,2,3,4,5,6,7,8,9,10,11,12};
    const std::vector<int> REG_VTO_POINTS =  {5, 6, 2, 7, 3,
                                              126, 127, 128, 129, 130, 131, 132, 133,
                                              134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179};
    const std::vector<int> REG_JN_POINTS =   {22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61};
    const int NUM_OJ = REG_COCO_POINTS.size() + REG_VTO_POINTS.size() + REG_JN_POINTS.size(); // 112
    const std::vector<int> REG_POF_A = {12, 2, 1, 12, 3, 4, 12, 8, 7, 12, 9, 10, 12, 11, 72, 73, 74, 11, 76, 77, 78, 11, 80, 81, 82, 11, 84, 85, 86, 11, 88, 89, 90, 6, 92, 93, 94, 6, 96, 97, 98, 6, 100, 101, 102, 6, 104, 105, 106, 6, 108, 109, 110};
    const std::vector<int> REG_POF_B = {2, 1, 0, 3, 4, 5, 8, 7, 6, 9, 10, 11, 13, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111};
    const int NUM_POFS = REG_POF_A.size();
    const int NUM_RESIDUALS = NUM_POFS*3 + NUM_OJ*2;
    const std::vector<int> JNB_INDEXES = {6, 0, 12};

    const float blw = 5;
    const std::vector<float> dblossdJnb =
    {
        blw, 0, 0, -blw/2, 0, 0, blw/2, 0, 0,
        0, blw, 0, 0, -blw/2, 0, 0, blw/2, 0,
        0, 0, blw, 0, 0, -blw/2, 0, 0, blw/2
    };

    struct FKData{
        // Hold Data
        Eigen::Matrix<float, 3, 3, Eigen::RowMajor> R; // Interface with ceres
        Eigen::Matrix<float, 9, 3 * NUM_JOINTS, Eigen::RowMajor> dRdP;
        Eigen::Matrix<float, 3, 1> offset; // a buffer for 3D vector
        Eigen::Matrix<float, 3, 3 * NUM_JOINTS, Eigen::RowMajor> dtdP; // a buffer for the derivative
        Eigen::Matrix<float, 3, 3 * NUM_JOINTS, Eigen::RowMajor> dtdJ; // a buffer for the derivative
        Eigen::Matrix<float, 3, 3 * NUM_JOINTS, Eigen::RowMajor> dtdJ2; // a buffer for the derivative
        std::vector<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> MR;
        std::vector<Eigen::Matrix<float, 3, 1>> Mt;
        std::vector<Eigen::Matrix<float, 9, 3 * NUM_JOINTS, Eigen::RowMajor>> dMRdP;
        std::vector<Eigen::Matrix<float, 3, 3 * NUM_JOINTS, Eigen::RowMajor>> dMtdP;
        std::vector<Eigen::Matrix<float, 3, 3 * NUM_JOINTS, Eigen::RowMajor>> dMtdJ;
        static const int num_joints = NUM_JOINTS;
        static const int num_t = (NUM_JOINTS) * 3 * 5;
        static const int transforms_t = 3 * NUM_JOINTS * 4 + 3 * NUM_JOINTS;
        float* dTrdP_ptr;
        float* dTrdJ_ptr;
        float* transforms_joint_ptr;

        FKData(){
            MR = std::vector<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(NUM_JOINTS, Eigen::Matrix<float, 3, 3, Eigen::RowMajor>(3, 3));
            Mt = std::vector<Eigen::Matrix<float, 3, 1>>(NUM_JOINTS, Eigen::Matrix<float, 3, 1>(3, 1));
            dMRdP = std::vector<Eigen::Matrix<float, 9, 3 * NUM_JOINTS, Eigen::RowMajor>>(NUM_JOINTS, Eigen::Matrix<float, 9, 3 * NUM_JOINTS, Eigen::RowMajor>(9, 3 * NUM_JOINTS));
            dMtdP = std::vector<Eigen::Matrix<float, 3, 3 * NUM_JOINTS, Eigen::RowMajor>>(NUM_JOINTS, Eigen::Matrix<float, 3, 3 * NUM_JOINTS, Eigen::RowMajor>(3, 3 * NUM_JOINTS));
            dMtdJ = std::vector<Eigen::Matrix<float, 3, 3 * NUM_JOINTS, Eigen::RowMajor>>(NUM_JOINTS, Eigen::Matrix<float, 3, 3 * NUM_JOINTS, Eigen::RowMajor>(3, 3 * NUM_JOINTS));
            cudaMallocHost((void**)&dTrdP_ptr, num_t*3*NUM_JOINTS * sizeof(float));
            cudaMallocHost((void**)&dTrdJ_ptr, num_t*3*NUM_JOINTS * sizeof(float));
            cudaMallocHost((void**)&transforms_joint_ptr, transforms_t * sizeof(float));
        }

        void zeroOutJac(bool jac){
            std::memset(R.data(), 0.0, R.rows() * R.cols() * sizeof(float));
            std::memset(&MR[0], 0.0, MR.size() * MR[0].rows() * MR[0].cols() * sizeof(float));
            std::memset(&Mt[0], 0.0, Mt.size() * Mt[0].rows() * Mt[0].cols() * sizeof(float));
            std::memset(offset.data(), 0.0, offset.rows() * offset.cols() * sizeof(float));
            std::memset((void*)transforms_joint_ptr, 0.0, transforms_t * sizeof(float));

            if(jac){
                std::memset(dRdP.data(), 0.0, dRdP.rows() * dRdP.cols() * sizeof(float));
                std::memset(dtdP.data(), 0.0, dtdP.rows() * dtdP.cols() * sizeof(float));
                std::memset(dtdJ.data(), 0.0, dtdJ.rows() * dtdJ.cols() * sizeof(float));
                std::memset(dtdJ2.data(), 0.0, dtdJ2.rows() * dtdJ2.cols() * sizeof(float));
                std::memset(&dMRdP[0], 0.0, dMRdP.size() * dMRdP[0].rows() * dMRdP[0].cols() * sizeof(float));
                std::memset(&dMtdP[0], 0.0, dMtdP.size() * dMtdP[0].rows() * dMtdP[0].cols() * sizeof(float));
                std::memset(&dMtdJ[0], 0.0, dMtdJ.size() * dMtdJ[0].rows() * dMtdJ[0].cols() * sizeof(float));
                std::memset((void*)dTrdP_ptr, 0.0, num_t*3*NUM_JOINTS * sizeof(float));
                std::memset((void*)dTrdJ_ptr, 0.0, num_t*3*NUM_JOINTS * sizeof(float));
            }
        }
    };


    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();

    // Timing
    void startTime(bool cuda=true){
        if(cuda) cudaDeviceSynchronize();
        begin = std::chrono::steady_clock::now();
    }

    void endTime(bool cuda=true, std::string name="name"){
        if(cuda) cudaDeviceSynchronize();
        end= std::chrono::steady_clock::now();
        std::cout << "Time difference = " << name << " " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000. <<std::endl;
    }

    void run(const torch::Tensor& t_tensor, const torch::Tensor& eulers_tensor, const torch::Tensor& bodyshape_tensor, const torch::Tensor& faceshape_tensor, bool jac=false);

    void setPofBodyWeight(float w);

    void setPofHandWeight(float w);

    void setProjCocoWeight(float w);

    void setProjFootWeight(float w);

    void setProjHandWeight(float w);

    void setProjFaceWeight(float w);

    void setProjWeightRange(const std::vector<int>& range, float w);

    void setPofWeightRange(const std::vector<int>& range, float w);

    double* r_d;
    double* drdt_d;
    double* drdP_d;
    double* drdc_d;
    double* drdf_d;

    void loadAdamData();

    AdamCuda(){
        at::init();

        rigidBodydrdP_tensor = torch::cat({torch::ones({NUM_RESIDUALS, 3}).cuda(), torch::zeros({NUM_RESIDUALS, NUM_POSE_PARAMETERS-3}).cuda()}, 1);

        r_d = new double[NUM_RESIDUALS];
        drdt_d = new double[NUM_RESIDUALS * 3];
        drdP_d = new double[NUM_RESIDUALS * NUM_POSE_PARAMETERS];
        drdc_d = new double[NUM_RESIDUALS * NUM_SHAPE_COEFFICIENTS];
        drdf_d = new double[NUM_RESIDUALS * NUM_EXP_BASIS_COEFFICIENTS];

        std::vector<long> jnb_full;
        for(auto item : JNB_INDEXES){
            jnb_full.emplace_back(item*3 + 0);
            jnb_full.emplace_back(item*3 + 1);
            jnb_full.emplace_back(item*3 + 2);
        }
        allocAndCopyCUDALongVec(DJNB_SELECT_tensor, jnb_full);
        allocAndCopyCUDAFloatVec(dblossdJnb_tensor, dblossdJnb);
        dblossdJnb_tensor.resize_({3, 9});

        cudaMallocHost((void**)&bodyshape_pinned, (NUM_SHAPE_COEFFICIENTS) * sizeof(float));
        cudaMallocHost((void**)&eulers_pinned, (NUM_POSE_PARAMETERS) * sizeof(float));

        std::vector<int> REG_POINTS;
        for(auto p : REG_COCO_POINTS){
            REG_POINTS.emplace_back(0);
            REG_POINTS.emplace_back(p);
        }
        for(auto p : REG_VTO_POINTS){
            REG_POINTS.emplace_back(1);
            REG_POINTS.emplace_back(p);
        }
        for(auto p : REG_JN_POINTS){
            REG_POINTS.emplace_back(2);
            REG_POINTS.emplace_back(p);
        }
        allocAndCopyCUDAIntVec(REG_POINTS_tensor, REG_POINTS);
        REG_POINTS_tensor.resize_({NUM_OJ,2});

        allocAndCopyCUDAIntVec(REG_POF_A_tensor, REG_POF_A);
        allocAndCopyCUDAIntVec(REG_POF_B_tensor, REG_POF_B);

        //runKernel();
        c = torch::zeros({62, 9, 62*3}, {torch::requires_grad(false)}).cuda();
        thetaTensor = torch::zeros({62,1}).cuda();
        cosTensor = torch::zeros({62,1}).cuda();
        sinTensor = torch::zeros({62,1}).cuda();
        eulersNormTensor = torch::zeros({62,3}).cuda();
        rotTensor = torch::zeros({62,9}).cuda();

        transforms_tensor = torch::zeros({NUM_JOINTS, 12}).cuda();
        jn_tensor = torch::zeros({NUM_JOINTS, 3}).cuda();
        dTrdP_tensor = torch::zeros({NUM_JOINTS*12, NUM_JOINTS*3}).cuda();
        dJndP_tensor = torch::zeros({NUM_JOINTS*3, NUM_JOINTS*3}).cuda();
        dTrdJ_tensor = torch::zeros({NUM_JOINTS*12, NUM_JOINTS*3}).cuda();
        dJndJ_tensor = torch::zeros({NUM_JOINTS*3, NUM_JOINTS*3}).cuda();

        dTrdc_tensor = torch::zeros({NUM_JOINTS*12, NUM_SHAPE_COEFFICIENTS}).cuda();

        m_meanshape_fast_tensor = torch::zeros({NUM_FAST_VERTICES*3, 1}).cuda();
        m_shapespace_u_fast_tensor = torch::zeros({NUM_FAST_VERTICES*3, NUM_SHAPE_COEFFICIENTS}).cuda();

        bodyshape_tensor = torch::zeros({NUM_SHAPE_COEFFICIENTS, 1}).cuda();


        Vt_in = torch::zeros({NUM_FAST_VERTICES*3, 1}).cuda();
        vt_temp_body = torch::zeros({NUM_FAST_VERTICES*3, 1}).cuda();
        vt_temp_face = torch::zeros({NUM_FAST_VERTICES*3, 1}).cuda();
        vt_ones = torch::ones({NUM_FAST_VERTICES, 1}).cuda();
        Vt_in_homo = torch::ones({NUM_FAST_VERTICES, 4, 1}).cuda();

        m_blendW_fast_tensor = torch::zeros({NUM_FAST_VERTICES, NUM_JOINTS}).cuda();

        blended_transforms = torch::zeros({NUM_FAST_VERTICES, 12}).cuda();

        Vt_out_homo = torch::ones({NUM_FAST_VERTICES, 4, 1}).cuda();

        dVtodTr_tensor = torch::zeros({NUM_FAST_VERTICES*3, NUM_JOINTS*12}).cuda();

        dVtodVti_tensor= torch::zeros({NUM_FAST_VERTICES*3, NUM_FAST_VERTICES*3}).cuda();

        dVtodP_tensor = torch::zeros({NUM_FAST_VERTICES*3, NUM_JOINTS*3}).cuda();
        dVtodc_tensor = torch::zeros({NUM_FAST_VERTICES*3, NUM_SHAPE_COEFFICIENTS}).cuda();

        m_small_coco_reg_fast_tensor = torch::zeros({NUM_COCO_KP, NUM_FAST_VERTICES}).cuda();

        CocoV_tensor = torch::zeros({NUM_COCO_KP, 3}).cuda();

        t_tensor = torch::zeros({3}).cuda();

        dCocoVdVto_tensor = torch::zeros({NUM_COCO_KP*3, NUM_FAST_VERTICES*3}).cuda();

        m_dVdFaceEx_fast_tensor = torch::zeros({NUM_FAST_VERTICES*3, NUM_EXP_BASIS_COEFFICIENTS}).cuda();

        faceshape_tensor = torch::zeros({NUM_EXP_BASIS_COEFFICIENTS, 1}).cuda();

        dVtodf_tensor = torch::zeros({NUM_FAST_VERTICES*3, NUM_EXP_BASIS_COEFFICIENTS}).cuda();

        dCocoVdc_tensor = torch::zeros({NUM_COCO_KP*3, NUM_SHAPE_COEFFICIENTS}).cuda();
        dCocoVdP_tensor = torch::zeros({NUM_COCO_KP*3, NUM_JOINTS*3}).cuda();

        dJndc_tensor = torch::zeros({NUM_JOINTS*3, NUM_SHAPE_COEFFICIENTS}).cuda();

        OJ_tensor = torch::zeros({NUM_OJ, 3}).cuda();
        dOJdf_tensor = torch::zeros({NUM_OJ*3, NUM_EXP_BASIS_COEFFICIENTS}).cuda();
        dOJdc_tensor = torch::zeros({NUM_OJ*3, NUM_SHAPE_COEFFICIENTS}).cuda();
        dOJdP_tensor = torch::zeros({NUM_OJ*3, NUM_JOINTS*3}).cuda();

        ploss_tensor = torch::zeros({NUM_OJ, 2}).cuda();
        dplossdOJ_tensor = torch::zeros({NUM_OJ*2, NUM_OJ*3}).cuda();

        dOJdt_tensor = torch::eye(3).reshape({1,3,3}).repeat({NUM_OJ,1,1}).reshape({NUM_OJ*3, 3}).cuda();

        dplossdt_tensor = torch::zeros({NUM_OJ*2, 3}).cuda();
        dplossdP_tensor = torch::zeros({NUM_OJ*2, NUM_POSE_PARAMETERS}).cuda();
        dplossdc_tensor = torch::zeros({NUM_OJ*2, NUM_SHAPE_COEFFICIENTS}).cuda();
        dplossdf_tensor = torch::zeros({NUM_OJ*2, NUM_EXP_BASIS_COEFFICIENTS}).cuda();

        pofloss_tensor = torch::zeros({NUM_POFS, 3}).cuda();
        pof_tensor = torch::zeros({NUM_POFS, 3}).cuda();
        theta_tensor = torch::zeros({NUM_POFS}).cuda();

        dpoflossdpof_tensor = torch::zeros({NUM_POFS*3, NUM_POFS*3}).cuda();
        dpofdOJ_tensor = torch::zeros({NUM_POFS*3, NUM_OJ*3}).cuda();
        dpoflossdOJ_tensor = torch::zeros({NUM_POFS*3, NUM_OJ*3}).cuda();

        dpoflossdt_tensor = torch::zeros({NUM_POFS*3, 3}).cuda();
        dpoflossdP_tensor = torch::zeros({NUM_POFS*3, NUM_POSE_PARAMETERS}).cuda();
        dpoflossdc_tensor = torch::zeros({NUM_POFS*3, NUM_SHAPE_COEFFICIENTS}).cuda();
        dpoflossdf_tensor = torch::zeros({NUM_POFS*3, NUM_EXP_BASIS_COEFFICIENTS}).cuda();

        dJnbdc_tensor = torch::zeros({3*3, NUM_SHAPE_COEFFICIENTS}).cuda();
        dJnbdP_tensor = torch::zeros({3*3, NUM_POSE_PARAMETERS}).cuda();

        dblossdc_tensor = torch::zeros({3, NUM_SHAPE_COEFFICIENTS}).cuda();
        dblossdP_tensor = torch::zeros({3, NUM_POSE_PARAMETERS}).cuda();

        r_tensor = torch::zeros({NUM_RESIDUALS}).cuda();
        drdt_tensor = torch::zeros({NUM_RESIDUALS, 3}).cuda();
        drdP_tensor = torch::zeros({NUM_RESIDUALS, NUM_POSE_PARAMETERS}).cuda();
        drdc_tensor = torch::zeros({NUM_RESIDUALS, NUM_SHAPE_COEFFICIENTS}).cuda();
        drdf_tensor = torch::zeros({NUM_RESIDUALS, NUM_EXP_BASIS_COEFFICIENTS}).cuda();

        streams.emplace_back(at::cuda::getStreamFromPool());
        streams.emplace_back(at::cuda::getStreamFromPool());
        streams.emplace_back(at::cuda::getStreamFromPool());

        // Load
        loadAdamData();
    }

    bool ridigBody = false;

    float* bodyshape_pinned;
    float* eulers_pinned;

    torch::Tensor rigidBodydrdP_tensor;

    torch::Tensor r_tensor;
    torch::Tensor drdt_tensor;
    torch::Tensor drdP_tensor;
    torch::Tensor drdc_tensor;
    torch::Tensor drdf_tensor;

    torch::Tensor dJnbdc_tensor, dJnbdP_tensor;
    torch::Tensor DJNB_SELECT_tensor;
    torch::Tensor dblossdJnb_tensor;
    torch::Tensor dblossdc_tensor, dblossdP_tensor;

    torch::Tensor dpoflossdt_tensor, dpoflossdP_tensor, dpoflossdc_tensor, dpoflossdf_tensor;

    torch::Tensor dpoflossdpof_tensor; //NUM_POFS*3 x NUM_POFS*3
    torch::Tensor dpofdOJ_tensor; //NUM_POFS*3 x NUM_OJ*3
    torch::Tensor dpoflossdOJ_tensor; //NUM_POFS*3 x NUM_OJ*3

    torch::Tensor pofloss_tensor; //NUM_POFS x 3
    torch::Tensor pof_tensor; //NUM_POFS x 3
    torch::Tensor theta_tensor; //NUM_POFS x 1

    torch::Tensor dplossdt_tensor, dplossdP_tensor, dplossdc_tensor, dplossdf_tensor;

    torch::Tensor dCocoVdt_tensor, dVtodt_tensor, dJndt_tensor;
    torch::Tensor dOJdt_tensor;

    torch::Tensor REG_POINTS_tensor;
    torch::Tensor REG_POF_A_tensor, REG_POF_B_tensor;

    torch::Tensor J_mu_tensor, dJdc_tensor, m_parent_tensor, parentIndexes_tensor;
    Eigen::MatrixXf J_mu_, dJdc_;
    std::vector<int> m_parent; std::array<std::vector<int>, NUM_JOINTS> parentIndexes;
    torch::Tensor t_tensor, eulers_tensor, bodyshape_tensor;
    torch::Tensor thetaTensor, eulersNormTensor, cosTensor, sinTensor, rotTensor;
    torch::Tensor c;

    torch::Tensor faceshape_tensor;

    torch::Tensor transforms_tensor, jn_tensor;
    torch::Tensor dTrdP_tensor, dJndP_tensor;
    torch::Tensor dTrdJ_tensor, dJndJ_tensor;
    torch::Tensor dTrdc_tensor;

    torch::Tensor m_meanshape_fast_tensor, m_shapespace_u_fast_tensor;

    torch::Tensor vt_temp_body, vt_temp_face;
    torch::Tensor Vt_in, Vt_in_homo, Vt_out_homo;
    torch::Tensor vt_ones;

    torch::Tensor m_blendW_fast_tensor;
    torch::Tensor blended_transforms;

    torch::Tensor dVtodTr_tensor;
    torch::Tensor dVtodVti_tensor;

    torch::Tensor dVtodP_tensor, dVtodc_tensor;

    torch::Tensor m_small_coco_reg_fast_tensor;

    torch::Tensor CocoV_tensor;
    torch::Tensor dCocoVdVto_tensor;

    torch::Tensor m_dVdFaceEx_fast_tensor;

    torch::Tensor dVtodf_tensor;

    torch::Tensor dCocoVdc_tensor;
    torch::Tensor dCocoVdP_tensor;

    torch::Tensor dJndc_tensor;
    torch::Tensor Vto_tensor;

    torch::Tensor OJ_tensor;
    torch::Tensor dOJdf_tensor, dOJdc_tensor, dOJdP_tensor;

    torch::Tensor ploss_tensor, dplossdOJ_tensor;

    torch::Tensor calib_tensor;
    torch::Tensor proj_truth_tensor, pof_truth_tensor;

    std::vector<at::cuda::CUDAStream> streams;

    FKData fkData;

    bool sameSize(const torch::Tensor& tensor, const std::vector<int>& cmpsize){
        if(!tensor.defined()) return false;
        torch::IntArrayRef tsize = tensor.sizes();
        if(tsize.size() != cmpsize.size()) return false;
        for(int i=0; i<tsize.size(); i++) if(tsize[i] != cmpsize[i]) return false;
        return true;
    }

    void allocAndCopyCUDAFloatMat(torch::Tensor& outputTensor, const Eigen::MatrixXf& inputMatrix){
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> M = inputMatrix;
        if(!sameSize(outputTensor, {(int)M.rows(), (int)M.cols()}))
            outputTensor = torch::zeros({M.rows(), M.cols()}).cuda();
        cudaMemcpy(outputTensor.data_ptr(), M.data(), M.rows() * M.cols() * sizeof(float),
                   cudaMemcpyHostToDevice);
    }

    void allocAndCopyCUDAIntMat(torch::Tensor& outputTensor, const Eigen::MatrixXi& inputMatrix){
        Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> M = inputMatrix;
        if(!sameSize(outputTensor, {(int)M.rows(), (int)M.cols()}))
            outputTensor = torch::zeros({M.rows(), M.cols()}, at::kInt).cuda();
        cudaMemcpy(outputTensor.data_ptr(), M.data(), M.rows() * M.cols() * sizeof(int),
                   cudaMemcpyHostToDevice);
    }

    void allocAndCopyCUDAIntVec(torch::Tensor& outputTensor, const std::vector<int>& inputVector){
        if(!sameSize(outputTensor, {(int)inputVector.size()}))
            outputTensor = torch::zeros({(int)inputVector.size()}, at::kInt).cuda();
        cudaMemcpy(outputTensor.data_ptr(), inputVector.data(), inputVector.size() * sizeof(int),
                   cudaMemcpyHostToDevice);
    }

    void allocAndCopyCUDAFloatVec(torch::Tensor& outputTensor, const std::vector<float>& inputVector){
        if(!sameSize(outputTensor, {(int)inputVector.size()}))
            outputTensor = torch::zeros({(int)inputVector.size()}).cuda();
        cudaMemcpy(outputTensor.data_ptr(), inputVector.data(), inputVector.size() * sizeof(float),
                   cudaMemcpyHostToDevice);
    }

    void allocAndCopyCUDALongVec(torch::Tensor& outputTensor, const std::vector<long>& inputVector){
        if(!sameSize(outputTensor, {(int)inputVector.size()}))
            outputTensor = torch::zeros({(int)inputVector.size()}, at::kLong).cuda();
        cudaMemcpy(outputTensor.data_ptr(), inputVector.data(), inputVector.size() * sizeof(long),
                   cudaMemcpyHostToDevice);
    }

};

#endif // CPPTL_JSON_READER_H_INCLUDED
