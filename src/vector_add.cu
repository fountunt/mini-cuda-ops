/*
这个程序在做什么？
在GPU上并行计算 检测环境是否正常
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

constexpr int GUARD = 256;
constexpr int N = (1 << 24)+1;
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

//GPU计算加法
__global__ void vector_add_kernel(const float* a,const float* b,float* c,int n){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= n) return;
	c[i] = a[i] + b[i];
}

int main(){
	//定义字节内存
	const size_t bytes = (size_t)N * sizeof(float);

	//分配内存
	float* h_a = (float*)malloc(bytes);
	float* h_b = (float*)malloc(bytes);
	float* h_c = (float*)malloc(bytes);

	//host直接计算
	for(int i = 0;i < N;i++){
		h_a[i] = i * 0.5f;
		h_b[i] = i * 0.25f;
	}

	//device计算
	float *d_a,*d_b,*d_c;
	CUDA_CHECK(cudaMalloc((void**)&d_a,bytes));
	CUDA_CHECK(cudaMalloc((void**)&d_b,bytes));
	CUDA_CHECK(cudaMalloc((void**)&d_c,(size_t)(N + GUARD) * sizeof(float)));
	CUDA_CHECK(cudaMemset(d_c + N,0xAB,(size_t)GUARD * sizeof(float)));

	//拷贝数据 host->device
	CUDA_CHECK(cudaMemcpy(d_a,h_a,bytes,cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_b,h_b,bytes,cudaMemcpyHostToDevice));

	//定义块大小
	int grid = (N + BLOCK - 1) / BLOCK;

	//创建计时标记start，stop
	cudaEvent_t start,stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	//预热，让GPU进入稳定状态，避免冷启动开销影响时间
	vector_add_kernel<<<grid,BLOCK>>>(d_a,d_b,d_c,N);
	CUDA_CHECK(cudaGetLastError());

	//重复跑
	CUDA_CHECK(cudaEventRecord(start));
	for(int r = 0;r < REPEAT;r++){
		vector_add_kernel<<<grid,BLOCK>>>(d_a,d_b,d_c,N);
	}
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));

	//哨兵检测
	unsigned char guard_host[GUARD * sizeof(float)];
	CUDA_CHECK(cudaMemcpy(guard_host,d_c + N,(size_t)GUARD *
	sizeof(float),cudaMemcpyDeviceToHost));

	int dirty = 0;
	for(int k = 0;k < (int)(GUARD * sizeof(float));k++){
        	if(guard_host[k] != 0xAB) dirty++;
	}
	printf("guard: %s  (%d / %d bytes changed)\n",
        	dirty ? "FAIL" : "ok",dirty,(int)(GUARD * sizeof(float)));

	//计算时间
	float total_ms = 0.0f;
	CUDA_CHECK(cudaEventElapsedTime(&total_ms,start,stop));
	float ms = total_ms / REPEAT;

	//把结果拷回host
	CUDA_CHECK(cudaMemcpy(h_c,d_c,bytes,cudaMemcpyDeviceToHost));

	//输出结果
	double gb = 3.0 * (double)bytes / 1e9;
	printf("N = %d BLOCK = %d grid = %d REPEAT = %d\n",N,BLOCK,grid,REPEAT);
	printf("total = %.3f ms  per-call = %.4f ms\n",total_ms,ms);
	printf("bandwidth = %.1f GB/s\n",gb / ((double)ms * 1e-3));

	for(int i = 0;i < 5;i++){
		printf("c[%d] = %.2f + %.2f = %.2f\n",i,h_a[i],h_b[i],h_c[i]);
	}

	//结果写入文件
	FILE* fp = fopen("c.bin","wb");
	if(fp == NULL) {
		printf("open c.bin failed\n");
		return 1;
	}
	fwrite(h_c,sizeof(float),(size_t)N,fp);
	fclose(fp);
	printf("wrote c.bin (%zu bytes)\n",bytes);

	//释放内存
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
