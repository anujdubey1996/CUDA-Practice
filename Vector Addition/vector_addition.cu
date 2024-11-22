#include <cuda.h>
#include <stdio.h>

__global__ void vectorAddKernel(float *A, float *B, float *C, int N){

  int id = gridDim.x * blockIdx.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  //printf("Entered Kernel\n");
  for(int i=id; i<N; i+=stride){
    //printf("A[%d]:%f, B[%d]:%f\n",i,A[i],i,B[i]);
    C[i] = A[i] + B[i];
  }  
}

// Function to process inputs and return outputs to Python
extern "C" void vectorAdd(float *A, float *B, float *C, int N) {
    // This function is called from Python
    // Input:
    // - A, B: Input vectors
    // - C: Output vector (will be modified in-place)
    // - N: Size of the vectors
  float *A_gpu, *B_gpu, *C_gpu;
  int size = sizeof(int)*N;
  
  cudaMalloc(&A_gpu, size);
  cudaMalloc(&B_gpu, size);
  cudaMalloc(&C_gpu, size);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  cudaEventRecord(start);

  cudaMemcpy(A_gpu,  A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_gpu,  B, size, cudaMemcpyHostToDevice);
  cudaMemcpy(C_gpu,  C, size, cudaMemcpyHostToDevice);
  
  vectorAddKernel<<<128,512>>>(A_gpu,B_gpu,C_gpu,N);
  
  cudaMemcpy(A, A_gpu, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(B, B_gpu, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(C, C_gpu, size, cudaMemcpyDeviceToHost);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);  
  
  cudaDeviceSynchronize();

  cudaFree(A_gpu);
  cudaFree(B_gpu);
  cudaFree(C_gpu);


  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Elapsed time: %f ms\n", milliseconds);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}
