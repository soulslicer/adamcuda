#ifndef PYADAMCUDA_HPP
#define PYADAMCUDA_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <opencv2/core/core.hpp>
#include <stdexcept>

#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

//#include <torch/csrc/jit/pybind.h>
#include <torch/csrc/utils/pybind.h>

#include <adamcuda.h>

namespace op{

namespace py = pybind11;


PYBIND11_MODULE(pyadamcuda, m) {

    py::class_<AdamCuda>(m, "AdamCuda")
        .def(py::init<>())
        .def("run", &AdamCuda::run)
        .def_readwrite("proj_truth_tensor", &AdamCuda::proj_truth_tensor)
        .def_readwrite("calib_tensor", &AdamCuda::calib_tensor)
        .def_readwrite("pof_truth_tensor", &AdamCuda::pof_truth_tensor)
        .def_readwrite("r_tensor", &AdamCuda::r_tensor)
        .def_readwrite("drdt_tensor", &AdamCuda::drdt_tensor)
        .def_readwrite("drdP_tensor", &AdamCuda::drdP_tensor)
        .def_readwrite("drdc_tensor", &AdamCuda::drdc_tensor)
        .def_readwrite("drdf_tensor", &AdamCuda::drdf_tensor)
        .def_readwrite("posepriorloss_tensor", &AdamCuda::posepriorloss_tensor)
        .def_readwrite("facepriorloss_tensor", &AdamCuda::facepriorloss_tensor)
        .def_readwrite("shapecoeffloss_tensor", &AdamCuda::shapecoeffloss_tensor)
        .def_readwrite("dposepriorlossdP_tensor", &AdamCuda::dposepriorlossdP_tensor)
        .def_readwrite("dshapecoefflossdc_tensor", &AdamCuda::dshapecoefflossdc_tensor)
        .def_readwrite("dfacepriorlossdf_tensor", &AdamCuda::dfacepriorlossdf_tensor)
        ;

    #ifdef VERSION_INFO
        m.attr("__version__") = VERSION_INFO;
    #else
        m.attr("__version__") = "dev";
    #endif
}

}

#endif

