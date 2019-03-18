set(PYTHON_FILES
    pyadamcuda.cpp)

pybind11_add_module(pyadamcuda pyadamcuda.cpp)

target_link_libraries(pyadamcuda PRIVATE pybind11::module openpose ${OpenPose_3rdparty_libraries})
SET_TARGET_PROPERTIES(pyadamcuda PROPERTIES PREFIX "")
