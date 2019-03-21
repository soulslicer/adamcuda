# adamcuda
adamcuda

CUDA implementation of Jacobian Computation and Residual for Adam Model

Requirements: 

- CUDA 9.0 and above and CUDNN
- Pytorch 1.1 Compiled from Source
- Eigen
- Python Dev Libraries

To Test:

Go to adamcuda/build/python look at test_pyadamcuda.py

Runtime for Model per iteration on 1080 Ti: 0.9ms per iteration for 1 person
