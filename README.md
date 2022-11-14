## GOAL 
- Beat Numpy and Matlab in matrix iversion.
- Write guide to be able to use this inside matlab. 


# Single vs Double precision in Matrix Multiplication
Most consumer GPUS have great FP32 performance but behave really bad in FP64 (1/16 avg with AMD and 1/64 avg with NVIDIA).

FP64 allows us to get a medium error of e-16 while the medium error for FP32 is e-5.

We will be focusing on FP32 since it is where GPU's shine. 

We might switch to FP64 if the error starts to add up during the recursion process.


matmul results
https://1drv.ms/x/s!AoEhz_j--clChxYHudQoL2wH6p07?e=laQzH0


# Square matrix matmul vs non-square matrix
Since matmul for non square matrix in slower, we are going to use 2 kernels. One for square matrix and one for rectangular ones. The decision of which kernel to use is decided from the host. 


##TODO: 
- Transpose B ? to have coalesced reads 
- Add support for matrices where N is not a multiple of DIM
- Add support for non square matrices
- use float4 
- assign more work to each thread ?? 






