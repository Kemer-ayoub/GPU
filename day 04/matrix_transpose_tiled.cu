#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

#define TILE_DIM 32

// Naive transpose kernel
__global__ void matrixTransposeNaive(const int *a, int *c, int N){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if(i < N && j < N){
    c[i*N+j] = a[j*N+i];  // Direct transpose - uncoalesced writes!
  }
}

// Tiled transpose kernel
__global__ void matrixTransposeTiled(const int *a, int *c, int N){
  __shared__ int as[TILE_DIM][TILE_DIM+1];

  int i = blockIdx.x*TILE_DIM + threadIdx.x;
  int j = blockIdx.y*TILE_DIM + threadIdx.y;

  // Load into shared memory
  if(i < N && j < N){
    as[threadIdx.y][threadIdx.x] = a[j*N+i];
  }

  __syncthreads();

  // Recalculate for transposed position
  i = blockIdx.y*TILE_DIM + threadIdx.x;
  j = blockIdx.x*TILE_DIM + threadIdx.y;

  // Write transposed
  if(i < N && j < N){
    c[j*N+i] = as[threadIdx.x][threadIdx.y];
  }
}

int getRandomValue(int min, int max) {
    return min + rand() % (max - min + 1);
}

bool verifyTranspose(std::vector<int>& input, std::vector<int>& output, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (input[i * width + j] != output[j * height + i]) {
                printf("Mismatch at (%d, %d): expected %d, got %d\n", 
                       i, j, input[i * width + j], output[j * height + i]);
                return false;
            }
        }
    }
    return true;
}

int main() {
  constexpr int N = 1 << 12;  // 4096x4096 for better benchmarking
  const int numRuns = 100;

  std::vector<int> matrix_A(N * N);
  std::vector<int> matrix_C(N * N);

  // Initialize with random values
  srand(time(NULL));
  for(int i = 0; i < N * N; i++) {
      matrix_A[i] = getRandomValue(1, 100);
  }

  // Allocate device memory
  int *d_a, *d_c;
  cudaMalloc((void **)&d_a, sizeof(int)*N*N);
  cudaMalloc((void **)&d_c, sizeof(int)*N*N);

  cudaMemcpy(d_a, matrix_A.data(), sizeof(int)*N*N, cudaMemcpyHostToDevice);

  // Setup execution configuration
  dim3 dimBlock(TILE_DIM, TILE_DIM);
  dim3 dimGrid((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

  // Create CUDA events for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // ========== NAIVE TRANSPOSE ==========
  printf("Running Naive Transpose...\n");
  
  // Warm up
  matrixTransposeNaive<<<dimGrid, dimBlock>>>(d_a, d_c, N);
  cudaDeviceSynchronize();

  // Benchmark
  cudaEventRecord(start);
  for(int i = 0; i < numRuns; i++){
    matrixTransposeNaive<<<dimGrid, dimBlock>>>(d_a, d_c, N);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float naiveTime;
  cudaEventElapsedTime(&naiveTime, start, stop);
  naiveTime /= numRuns;  // Average time

  // Copy back and verify
  cudaMemcpy(matrix_C.data(), d_c, sizeof(int)*N*N, cudaMemcpyDeviceToHost);
  
  if (verifyTranspose(matrix_A, matrix_C, N, N)) {
    printf("Naive transpose: PASSED ✓\n");
  } else {
    printf("Naive transpose: FAILED ✗\n");
  }

  // ========== TILED TRANSPOSE ==========
  printf("\nRunning Tiled Transpose...\n");
  
  // Warm up
  matrixTransposeTiled<<<dimGrid, dimBlock>>>(d_a, d_c, N);
  cudaDeviceSynchronize();

  // Benchmark
  cudaEventRecord(start);
  for(int i = 0; i < numRuns; i++){
    matrixTransposeTiled<<<dimGrid, dimBlock>>>(d_a, d_c, N);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float tiledTime;
  cudaEventElapsedTime(&tiledTime, start, stop);
  tiledTime /= numRuns;  // Average time

  // Copy back and verify
  cudaMemcpy(matrix_C.data(), d_c, sizeof(int)*N*N, cudaMemcpyDeviceToHost);
  
  if (verifyTranspose(matrix_A, matrix_C, N, N)) {
    printf("Tiled transpose: PASSED ✓\n");
  } else {
    printf("Tiled transpose: FAILED ✗\n");
  }

  // ========== PERFORMANCE ANALYSIS ==========
  float speedup = naiveTime / tiledTime;
  size_t bytes = 2 * N * N * sizeof(int);  // Read + Write
  float bandwidthNaive = bytes / (naiveTime * 1e6);  // GB/s
  float bandwidthTiled = bytes / (tiledTime * 1e6);  // GB/s

  printf("\n========================================\n");
  printf("         PERFORMANCE ANALYSIS\n");
  printf("========================================\n");
  printf("Matrix size:           %d x %d\n", N, N);
  printf("Tile dimension:        %d\n", TILE_DIM);
  printf("Number of runs:        %d\n", numRuns);
  printf("----------------------------------------\n");
  printf("Naive transpose time:  %.3f ms\n", naiveTime);
  printf("Tiled transpose time:  %.3f ms\n", tiledTime);
  printf("----------------------------------------\n");
  printf("Speedup:               %.2fx\n", speedup);
  printf("----------------------------------------\n");
  printf("Naive bandwidth:       %.2f GB/s\n", bandwidthNaive);
  printf("Tiled bandwidth:       %.2f GB/s\n", bandwidthTiled);
  printf("Bandwidth improvement: %.2fx\n", bandwidthTiled / bandwidthNaive);
  printf("========================================\n");

  // Cleanup
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_a);
  cudaFree(d_c);

  std::cout << "\nCOMPLETED SUCCESSFULLY\n";

  return 0;
}