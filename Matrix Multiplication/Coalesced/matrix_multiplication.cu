#include <cuda.h>
#include <stdio.h>

__global__ void matrixMulKernel(float *A,float *B, float *C, int M, int N, int K)
{
  int col_id = blockDim.x * blockIdx.x + threadIdx.x;
  int row_id = blockDim.y * blockIdx.y + threadIdx.y;
  //int stride = blockDim.x *gridDim.x;
  

  if(row_id <M && col_id<N){
      //printf("R,C,S: %d %d\n",row_id,col_id);

      for(int k=0;k<N;k+=1)
      {
          C[row_id*N + col_id] += A[row_id*N + k] * B[col_id + k*N]; 
          
      } 
      //printf("Row, Col, C[row_id*N + col_id] :%d, %d, %f\n",row_id, col_id, C[row_id*N + col_id]);
  }
}

extern "C" void matrixMultiply(float *A, float *B, float *C, int M, int N, int K) {
    // Declare CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing
    cudaEventRecord(start);
    int size_a = sizeof(int) * M * N;
    int size_b = sizeof(int) * N * K;
    int size_c = sizeof(int) * M * K;
    //printf("Entered Here!\n");
    float *A_gpu, *B_gpu, *C_gpu;
    cudaMalloc(&A_gpu, size_a);
    cudaMalloc(&B_gpu, size_b);
    cudaMalloc(&C_gpu, size_c);

    cudaMemcpy(A_gpu,A,size_a,cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu,B,size_b,cudaMemcpyHostToDevice);
    cudaMemcpy(C_gpu,C,size_c,cudaMemcpyHostToDevice);
    dim3 threads(32,32);
    dim3 blocks((((M+threads.x-1)/threads.x) + 1), (((N+threads.y-1)/threads.y) + 1));
    //printf("Entered Here too!\n");
    matrixMulKernel<<<blocks, threads>>>(A_gpu,B_gpu,C_gpu,M,N,K);

    cudaMemcpy(C,C_gpu,size_c,cudaMemcpyDeviceToHost);

    cudaError_t lastError,async_error;

    lastError = cudaGetLastError();
    if(lastError != cudaSuccess){
      printf("LE: %s\n",cudaGetErrorString(lastError));
    }


    async_error = cudaDeviceSynchronize();
    if(async_error != cudaSuccess){
      printf("AE: %s\n",cudaGetErrorString(async_error));
    }
    
    printf("Entered Here as well!\n");
    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Elapsed time: %f ms\n", milliseconds);

    // Clean up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}