#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

constexpr int N = (1 << 24)+1;
constexpr int REPEAT = 10;

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
__global__ void reduce_kernel(const float* x,float* out,int n){
	float sum = 0.0f;
	for(int i = 0;i < n;i++){
		sum += x[i];
	}
	out[0] = sum;
}

int main(){
	//输入数组字节数
	const size_t bytes = (size_t)N * sizeof(float);

	//分配内存
	float* h_x = (float*)malloc(bytes);
	float* h_out = (float*)malloc(sizeof(float));

	//host直接计算
	for(int i = 0;i < N;i++){
		h_x[i] = (float)i;
	}

	//device计算
	float *d_x,*d_out;
	CUDA_CHECK(cudaMalloc(&d_x,bytes));
	CUDA_CHECK(cudaMalloc(&d_out,sizeof(float)));

	//拷贝数据 host->device
	CUDA_CHECK(cudaMemcpy(d_x,h_x,bytes,cudaMemcpyHostToDevice));

	//创建计时标记start，stop
	cudaEvent_t start,stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	//预热，让GPU进入稳定状态，避免冷启动开销影响时间
	reduce_kernel<<<1,1>>>(d_x,d_out,N);
	CUDA_CHECK(cudaGetLastError());

	//重复跑
	CUDA_CHECK(cudaEventRecord(start));
	for(int r = 0;r < REPEAT;r++){
		reduce_kernel<<<1,1>>>(d_x,d_out,N);
	}
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));

	CUDA_CHECK(cudaMemcpy(h_out,d_out,sizeof(float),cudaMemcpyDeviceToHost));

	//计算时间
	float total_ms = 0.0f;
	CUDA_CHECK(cudaEventElapsedTime(&total_ms,start,stop));
	float ms = total_ms / REPEAT;

	//输出结果
	double gb = 1.0 * (double)bytes / 1e9;
	printf("N = %d launch = <<<1,1>>> REPEAT = %d\n",N,REPEAT);
	printf("total = %.3f ms  per-call = %.4f ms\n",total_ms,ms);
	printf("bandwidth = %.3f GB/s\n",gb / ((double)ms * 1e-3));
	printf("sum = %.9g\n",(double)h_out[0]);

	//主机 float32 顺序累加,和 kernel 结果对比
	//顺序、精度都一致 -> 应该逐位相等
	float h_seq = 0.0f;
	for(int i = 0;i < N;i++){
		h_seq += h_x[i];
	}
	printf("host seq = %.9g\n",(double)h_seq);
	printf("kernel   = %.9g\n",(double)h_out[0]);
	printf("bitwise  = %s\n",h_seq == h_out[0] ? "IDENTICAL" : "DIFFERENT");

	//结果写入文件
	FILE* fp = fopen("sum.bin","wb");
	if(fp == NULL) {
		printf("open sum.bin failed\n");
		return 1;
	}
	fwrite(h_out,sizeof(float),1,fp);
	fclose(fp);
	printf("wrote sum.bin (4 bytes)\n");

	//释放内存
	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));
	CUDA_CHECK(cudaFree(d_x));
	CUDA_CHECK(cudaFree(d_out));
	free(h_x);
	free(h_out);

	printf("ok\n");
	return 0;
}
