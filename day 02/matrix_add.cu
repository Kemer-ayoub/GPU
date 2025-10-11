#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

__global__ void matrixAdd(const int *a, const int *b, int *c, int N){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  index += (blockIdx.y * blockDim.y + threadIdx.y) * N;

  if(index < N*N){
    c[index] = a[index] + b[index];
  }

}

int getRandomValue(int min, int max) {
    return min + rand() % (max - min + 1);
}

int main() {
  // Array size of 2^10 (65536 elements)
  constexpr int N = 1 << 10;

  std::vector<int> matrix_A(N * N);
  std::vector<int> matrix_B(N * N);
  std::vector<int> matrix_C(N * N);

  for(int i = 0; i < N * N; i++) {
      matrix_A[i] = getRandomValue(1, 100);
      matrix_B[i] = getRandomValue(1, 100);
  }

  int *d_a, *d_b, *d_c;
  cudaMalloc((void **)&d_a, sizeof(int)*N*N);
  cudaMalloc((void **)&d_b, sizeof(int)*N*N);
  cudaMalloc((void **)&d_c, sizeof(int)*N*N);

  cudaMemcpy(d_a, matrix_A.data(), sizeof(int)*N*N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, matrix_B.data(), sizeof(int)*N*N, cudaMemcpyHostToDevice);

  dim3 dimBlock(32,16);
  dim3 dimGrid(ceil(N/32),ceil(N/16));


  matrixAdd<<<dimGrid,dimBlock>>>(d_a,d_b,d_c,N);

  cudaMemcpy(matrix_C.data(), d_c, sizeof(int)*N*N, cudaMemcpyDeviceToHost);


  // Free memory on device
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  std::cout << "COMPLETED SUCCESSFULLY\n";

  return 0;
}