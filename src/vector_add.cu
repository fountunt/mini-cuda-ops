#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

constexpr int N = 1 << 24;
constexpr int BLOCK = 256;
constexpr int REPEAT = 100;

#define CUDA_CHECK(call)\
	do{\
	   cudaError_t err = (call);\
	   if(err != cudaSuccess) {\
		printf("CUDA error at %s:%d -> %s\n",__FILE__,__LINE__,\
			cudaGetErrorString(err));\
		exit(1);\
	   }\
	} while(0)

__global__ void vector_add_kernel(const float* a,const float* b,float* c,int n){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= n) return;
	c[i] = a[i] + b[i];
}

int main(){
	const size_t bytes = (size_t)N * sizeof(float);

	float* h_a = (float*)malloc(bytes);
	float* h_b = (float*)malloc(bytes);
	float* h_c = (float*)malloc(bytes);
	for(int i = 0;i < N;i++){
		h_a[i] = i * 0.5f;
		h_b[i] = i * 0.25f;
	}

	float *d_a,*d_b,*d_c;
	CUDA_CHECK(cudaMalloc((void**)&d_a,bytes));
	CUDA_CHECK(cudaMalloc((void**)&d_b,bytes));
	CUDA_CHECK(cudaMalloc((void**)&d_c,bytes));

	CUDA_CHECK(cudaMemcpy(d_a,h_a,bytes,cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_b,h_b,bytes,cudaMemcpyHostToDevice));

	int grid = (N + BLOCK - 1) / BLOCK;

	cudaEvent_t start,stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	vector_add_kernel<<<grid,BLOCK>>>(d_a,d_b,d_c,N);
	CUDA_CHECK(cudaGetLastError());

	CUDA_CHECK(cudaEventRecord(start));
	for(int r = 0;r < REPEAT;r++){
		vector_add_kernel<<<grid,BLOCK>>>(d_a,d_b,d_c,N);
	}
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));

	float total_ms = 0.0f;
	CUDA_CHECK(cudaEventElapsedTime(&total_ms,start,stop));
	float ms = total_ms / REPEAT;

	CUDA_CHECK(cudaMemcpy(h_c,d_c,bytes,cudaMemcpyDeviceToHost));

	double gb = 3.0 * (double)bytes / 1e9;
	printf("N = %d BLOCK = %d grid = %d REPEAT = %d\n",N,BLOCK,grid,REPEAT);
	printf("total = %.3f ms  per-call = %.4f ms\n",total_ms,ms);
	printf("bandwidth = %.1f GB/s\n",gb / ((double)ms * 1e-3));

	for(int i = 0;i < 5;i++){
		printf("c[%d] = %.2f + %.2f = %.2f\n",i,h_a[i],h_b[i],h_c[i]);
	}

	FILE* fp = fopen("c.bin","wb");
	if(fp == NULL) {
		printf("open c.bin failed\n");
		return 1;
	}
	fwrite(h_c,sizeof(float),(size_t)N,fp);
	fclose(fp);
	printf("wrote c.bin (%zu bytes)\n",bytes);

	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));
	CUDA_CHECK(cudaFree(d_a));
	CUDA_CHECK(cudaFree(d_b));
	CUDA_CHECK(cudaFree(d_c));
	free(h_a);
	free(h_b);
	free(h_c);

	printf("ok\n");
	return 0;
}
