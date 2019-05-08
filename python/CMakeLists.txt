set(PYTHON_FILES
    test_pyadamcuda.py
    test_opt.py
    util.py
    pyadamcuda.cpp)

pybind11_add_module(pyadamcuda pyadamcuda.cpp)

target_link_libraries(pyadamcuda PRIVATE pybind11::module MYLIB ${3rdparty_libraries})
SET_TARGET_PROPERTIES(pyadamcuda PROPERTIES PREFIX "")
configure_file(test_pyadamcuda.py test_pyadamcuda.py)
configure_file(test_opt.py test_opt.py)
configure_file(util.py util.py)
