#ifndef PYADAMCUDA_HPP
#define PYADAMCUDA_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <opencv2/core/core.hpp>
#include <stdexcept>

namespace op{

namespace py = pybind11;

void parse_gflags(const std::vector<std::string>& argv)
{
    std::vector<char*> argv_vec;
    for(auto& arg : argv) argv_vec.emplace_back((char*)arg.c_str());
    char** cast = &argv_vec[0];
    int size = argv_vec.size();
    gflags::ParseCommandLineFlags(&size, &cast, true);
}

void init_int(py::dict d)
{
    std::vector<std::string> argv;
    argv.emplace_back("openpose.py");
    for (auto item : d){
        argv.emplace_back("--" + std::string(py::str(item.first)));
        argv.emplace_back(py::str(item.second));
    }
    parse_gflags(argv);
}

void init_argv(std::vector<std::string> argv)
{
    argv.insert(argv.begin(), "openpose.py");
    parse_gflags(argv);
}

PYBIND11_MODULE(pyadamcuda, m) {

//    // Functions for Init Params
//    m.def("init_int", &init_int, "Init Function");
//    m.def("init_argv", &init_argv, "Init Function");

//    // OpenposePython
//    py::class_<WrapperPython>(m, "WrapperPython")
//        .def(py::init<>())
//        .def(py::init<int>())
//        .def("configure", &WrapperPython::configure)
//        .def("start", &WrapperPython::start)
//        .def("stop", &WrapperPython::stop)
//        .def("execute", &WrapperPython::exec)
//        .def("emplaceAndPop", &WrapperPython::emplaceAndPop)
//        ;

//    // Rectangle
//    py::class_<op::Rectangle<float>>(m, "Rectangle")
//        .def("__repr__", [](op::Rectangle<float> &a) { return a.toString(); })
//        .def(py::init<>())
//        .def(py::init<float, float, float, float>())
//        .def_readwrite("x", &op::Rectangle<float>::x)
//        .def_readwrite("y", &op::Rectangle<float>::y)
//        .def_readwrite("width", &op::Rectangle<float>::width)
//        .def_readwrite("height", &op::Rectangle<float>::height)
//        ;

//    // Point
//    py::class_<op::Point<int>>(m, "Point")
//        .def("__repr__", [](op::Point<int> &a) { return a.toString(); })
//        .def(py::init<>())
//        .def(py::init<int, int>())
//        .def_readwrite("x", &op::Point<int>::x)
//        .def_readwrite("y", &op::Point<int>::y)
//        ;

    #ifdef VERSION_INFO
        m.attr("__version__") = VERSION_INFO;
    #else
        m.attr("__version__") = "dev";
    #endif
}

}

//// Numpy - op::Array<float> interop
//namespace pybind11 { namespace detail {

//template <> struct type_caster<op::Array<float>> {
//    public:

//        PYBIND11_TYPE_CASTER(op::Array<float>, _("numpy.ndarray"));

//        // Cast numpy to op::Array<float>
//        bool load(handle src, bool imp)
//        {
//            // array b(src, true);
//            array b = reinterpret_borrow<array>(src);
//            buffer_info info = b.request();

//            if (info.format != format_descriptor<float>::format())
//                throw std::runtime_error("op::Array only supports float32 now");

//            //std::vector<int> a(info.shape);
//            std::vector<int> shape(std::begin(info.shape), std::end(info.shape));

//            // No copy
//            value = op::Array<float>(shape, (float*)info.ptr);
//            // Copy
//            //value = op::Array<float>(shape);
//            //memcpy(value.getPtr(), info.ptr, value.getVolume()*sizeof(float));

//            return true;
//        }

//        // Cast op::Array<float> to numpy
//        static handle cast(const op::Array<float> &m, return_value_policy, handle defval)
//        {
//            std::string format = format_descriptor<float>::format();
//            return array(buffer_info(
//                m.getPseudoConstPtr(),/* Pointer to buffer */
//                sizeof(float),        /* Size of one scalar */
//                format,               /* Python struct-style format descriptor */
//                m.getSize().size(),   /* Number of dimensions */
//                m.getSize(),          /* Buffer dimensions */
//                m.getStride()         /* Strides (in bytes) for each index */
//                )).release();
//        }

//    };
//}} // namespace pybind11::detail

//// Numpy - cv::Mat interop
//namespace pybind11 { namespace detail {

//template <> struct type_caster<cv::Mat> {
//    public:

//        PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));

//        // Cast numpy to cv::Mat
//        bool load(handle src, bool)
//        {
//            /* Try a default converting into a Python */
//            //array b(src, true);
//            array b = reinterpret_borrow<array>(src);
//            buffer_info info = b.request();

//            int ndims = info.ndim;

//            decltype(CV_32F) dtype;
//            size_t elemsize;
//            if (info.format == format_descriptor<float>::format()) {
//                if (ndims == 3) {
//                    dtype = CV_32FC3;
//                } else {
//                    dtype = CV_32FC1;
//                }
//                elemsize = sizeof(float);
//            } else if (info.format == format_descriptor<double>::format()) {
//                if (ndims == 3) {
//                    dtype = CV_64FC3;
//                } else {
//                    dtype = CV_64FC1;
//                }
//                elemsize = sizeof(double);
//            } else if (info.format == format_descriptor<unsigned char>::format()) {
//                if (ndims == 3) {
//                    dtype = CV_8UC3;
//                } else {
//                    dtype = CV_8UC1;
//                }
//                elemsize = sizeof(unsigned char);
//            } else {
//                throw std::logic_error("Unsupported type");
//                return false;
//            }

//            std::vector<int> shape = {(int)info.shape[0], (int)info.shape[1]};

//            value = cv::Mat(cv::Size(shape[1], shape[0]), dtype, info.ptr, cv::Mat::AUTO_STEP);
//            return true;
//        }

//        // Cast cv::Mat to numpy
//        static handle cast(const cv::Mat &m, return_value_policy, handle defval)
//        {
//            std::string format = format_descriptor<unsigned char>::format();
//            size_t elemsize = sizeof(unsigned char);
//            int dim;
//            switch(m.type()) {
//                case CV_8U:
//                    format = format_descriptor<unsigned char>::format();
//                    elemsize = sizeof(unsigned char);
//                    dim = 2;
//                    break;
//                case CV_8UC3:
//                    format = format_descriptor<unsigned char>::format();
//                    elemsize = sizeof(unsigned char);
//                    dim = 3;
//                    break;
//                case CV_32F:
//                    format = format_descriptor<float>::format();
//                    elemsize = sizeof(float);
//                    dim = 2;
//                    break;
//                case CV_64F:
//                    format = format_descriptor<double>::format();
//                    elemsize = sizeof(double);
//                    dim = 2;
//                    break;
//                default:
//                    throw std::logic_error("Unsupported type");
//            }

//            std::vector<size_t> bufferdim;
//            std::vector<size_t> strides;
//            if (dim == 2) {
//                bufferdim = {(size_t) m.rows, (size_t) m.cols};
//                strides = {elemsize * (size_t) m.cols, elemsize};
//            } else if (dim == 3) {
//                bufferdim = {(size_t) m.rows, (size_t) m.cols, (size_t) 3};
//                strides = {(size_t) elemsize * m.cols * 3, (size_t) elemsize * 3, (size_t) elemsize};
//            }
//            return array(buffer_info(
//                m.data,         /* Pointer to buffer */
//                elemsize,       /* Size of one scalar */
//                format,         /* Python struct-style format descriptor */
//                dim,            /* Number of dimensions */
//                bufferdim,      /* Buffer dimensions */
//                strides         /* Strides (in bytes) for each index */
//                )).release();
//        }

//    };
//}} // namespace pybind11::detail

#endif

