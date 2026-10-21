#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cmath>

constexpr int ROWS = 4096;
constexpr int COLS = 4096;
constexpr int BLOCK = 256;

__global__ void row_max_kernel(const float* x,float* m){
	__shared__ float sh[BLOCK];
	const int tid = threadIdx.x;
	const float* row = x + (size_t)blockIdx.x * COLS;

	float local = -INFINITY;
	for(int j = tid;j < COLS;j += BLOCK){
		local = fmaxf(local,row[j]);
	}

	sh[tid] = local;
	__syncthreads();

	for(int s = BLOCK >> 1;s > 0;s >>= 1){
		if(tid < s) sh[tid] = fmaxf(sh[tid],sh[tid + s]);
		__syncthreads();
	}

	if(tid == 0) m[blockIdx.x] = sh[0];
}

__global__ void row_sum_kernel(const float* x,const float* m,float* y){
	__shared__ float srow[COLS];
	__shared__ float sh[BLOCK];

	const int tid = threadIdx.x;
	const float* row = x + (size_t)blockIdx.x * COLS;
	float* y_row = y + (size_t)blockIdx.x * COLS;
	const float mv = m[blockIdx.x];

	float local = 0.0f;
	for(int j = tid;j < COLS;j += BLOCK){
		float e = expf(row[j] - mv);
		srow[j] = e;
		local += e;
	}

	sh[tid] = local;
	__syncthreads();

	for(int s = BLOCK >> 1;s > 0;s >>= 1){
		if(tid < s) sh[tid] += sh[tid + s];
		__syncthreads();
	}

	const float S = sh[0];
	for(int j = tid;j < COLS;j += BLOCK){
		y_row[j] = srow[j] / S;
	}
}

static void bench(const float* d_x,float* d_m,float* d_y){
	const int WARM = 100;
	const int REPS1 = 1000;
	const int REPS2 = 500;
	const int REPS3 = 500;
	const size_t n = (size_t)ROWS * COLS;

	cudaEvent_t start,stop;
	cudaError_t err;

	err = cudaEventCreate(&start);
	if(err != cudaSuccess){printf("event create failed\n"); return;}
        err = cudaEventCreate(&stop);
        if(err != cudaSuccess){printf("event create failed\n"); cudaEventDestroy(start); return;}

	//k1:纯读
	for(int i = 0;i < WARM;i++) row_max_kernel<<<ROWS,BLOCK>>>(d_x,d_m);
	err = cudaGetLastError();
	if(err != cudaSuccess){
		printf("k1 warmup failed: %s\n",cudaGetErrorString(err));
		return;
	}

	cudaEventRecord(start);
	for(int i = 0;i < REPS1;i++) row_max_kernel<<<ROWS,BLOCK>>>(d_x,d_m);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float ms1 = 0.0f;
	cudaEventElapsedTime(&ms1,start,stop);
	ms1 /= REPS1;

	//k2:读1写1
        for(int i = 0;i < WARM;i++) row_sum_kernel<<<ROWS,BLOCK>>>(d_x,d_m,d_y);
        err = cudaGetLastError();
        if(err != cudaSuccess){
                printf("k2 warmup failed: %s\n",cudaGetErrorString(err));
                return;
        }

        cudaEventRecord(start);
        for(int i = 0;i < REPS2;i++) row_sum_kernel<<<ROWS,BLOCK>>>(d_x,d_m,d_y);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms2 = 0.0f;
        cudaEventElapsedTime(&ms2,start,stop);
        ms2 /= REPS2;

	//k1+k2
        for(int i = 0;i < WARM;i++){
		row_max_kernel<<<ROWS,BLOCK>>>(d_x,d_m);
		row_sum_kernel<<<ROWS,BLOCK>>>(d_x,d_m,d_y);
	}
        err = cudaGetLastError();
        if(err != cudaSuccess){
                printf("combined warmup failed: %s\n",cudaGetErrorString(err));
                return;
        }

        cudaEventRecord(start);
        for(int i = 0;i < REPS3;i++){
		row_max_kernel<<<ROWS,BLOCK>>>(d_x,d_m);
		row_sum_kernel<<<ROWS,BLOCK>>>(d_x,d_m,d_y);
	}
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms3 = 0.0f;
        cudaEventElapsedTime(&ms3,start,stop);
        ms3 /= REPS3;

	double gb1 = 1.0 * n * 4 / 1e9;
	double gb2 = 1.0 * n * 8 / 1e9;
	double gb3 = 1.0 * n * 12 / 1e9;

	printf("k1:\nper-call = %.4f ms\nbw = %.1f GB/s\nREPEAT = %d\nwindow = %.1fms\ntotal = %.1f ms\n",
		ms1,gb1 / (ms1 * 1e-3),REPS1,ms1 * REPS1,(WARM + REPS1) * ms1);
	printf("k2:\nper-call = %.4f ms\nbw = %.1f GB/s\nREPEAT = %d\nwindow = %.1fms\ntotal = %.1f ms\n",
		ms2,gb2 / (ms2 * 1e-3),REPS2,ms2 * REPS2,(WARM + REPS2) * ms2);
	printf("k1+k2:\nper-call = %.4f ms\nbw = %.1f GB/s\nREPEAT = %d\nwindow = %.1fms\ntotal = %.1f ms\n",
		ms3,gb3 / (ms3 * 1e-3),REPS3,ms3 * REPS3,(WARM + REPS3) * ms3);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}


int main(int argc,char** argv){
	const size_t n = (size_t)ROWS * COLS;
	const size_t bytes = n * sizeof(float);

	float* h_x = (float*)malloc(bytes);
	if(h_x == NULL){
		printf("malloc failed\n");
		return 1;
	}

	//生成数据
	for(int r = 0;r < ROWS;r++){
		for(int c = 0;c < COLS;c++){
			size_t idx = (size_t)r * COLS + c;
			if(r == 0){
				h_x[idx] = -120.0f - 0.001f * (float)c;
			} else {
				h_x[idx] = (float)((c * 37 + r * 101) % 300) - 150.0f;
			}
		}
	}

	printf("n = %zu bytes = %zu\n",n,bytes);
	for(int c = 0;c < 4;c++){
		printf("h_x[0][%d] = %.3f\n",c,h_x[c]);
	}
	printf("h_x[1][0...3] = %.3f %.3f %.3f %.3f\n",
		h_x[(size_t)COLS + 0],h_x[(size_t)COLS + 1],
		h_x[(size_t)COLS + 2],h_x[(size_t)COLS + 3]);

	FILE* fp = fopen("x.bin","wb");
	if(fp == NULL){
		printf("打开x.bin失败\n");
		free(h_x);
		return 1;
	}
	size_t wrote = fwrite(h_x,sizeof(float),n,fp);
	fclose(fp);
	printf("wrote x.bin (%zu bytes),fwrite returned %zu\n",bytes,wrote);

	float* d_x = nullptr;
	float* d_m = nullptr;
	float* d_y = nullptr;
	cudaError_t err;

	err = cudaMalloc(&d_x,bytes);
	if(err != cudaSuccess){
		printf("cudaMalloc d_x failed: %s\n",cudaGetErrorString(err));
		free(h_x);
		return 1;
	}

	err = cudaMalloc(&d_m,(size_t)ROWS * sizeof(float));
	if(err != cudaSuccess){
		printf("cudaMalloc d_m failed: %s\n",cudaGetErrorString(err));
		cudaFree(d_x);
		free(h_x);
		return 1;
	}

	err = cudaMalloc(&d_y,bytes);
	if(err != cudaSuccess){
		printf("cudaMalloc d_y failed: %s\n",cudaGetErrorString(err));
		cudaFree(d_m);
		cudaFree(d_x);
		free(h_x);
		return 1;
	}

	err = cudaMemcpy(d_x,h_x,bytes,cudaMemcpyHostToDevice);
	if(err != cudaSuccess){
		printf("copy x to device failed: %s\n",cudaGetErrorString(err));
		cudaFree(d_m);
		cudaFree(d_x);
		cudaFree(d_y);
		free(h_x);
		return 1;
	}

	row_max_kernel<<<ROWS,BLOCK>>>(d_x,d_m);

	err = cudaGetLastError();

	if(err != cudaSuccess){
		printf("kernel launch failed: %s\n",cudaGetErrorString(err));
		cudaFree(d_y);
		cudaFree(d_m);
		cudaFree(d_x);
		free(h_x);
		return 1;
	}

	row_sum_kernel<<<ROWS,BLOCK>>>(d_x,d_m,d_y);

	err = cudaGetLastError();
	if(err != cudaSuccess){
		printf("kernel k2 launch failed: %s\n",cudaGetErrorString(err));
		cudaFree(d_y);
		cudaFree(d_m);
		cudaFree(d_x);
		free(h_x);
		return 1;
	}

	float* h_m = (float*)malloc((size_t)ROWS * sizeof(float));
	if(h_m == NULL){
		printf("malloc h_m failed\n");
		cudaFree(d_y);
		cudaFree(d_m);
		cudaFree(d_x);
		free(h_x);
		return 1;
	}

	float* h_y = (float*)malloc(bytes);
	if(h_y == NULL){
		printf("malloc h_y failed\n");
		free(h_m);
		cudaFree(d_y);
		cudaFree(d_m);
		cudaFree(d_x);
		free(h_x);
		return 1;
	}

	err = cudaMemcpy(h_m,d_m,(size_t)ROWS * sizeof(float),cudaMemcpyDeviceToHost);
	if(err != cudaSuccess){
		printf("copy m failed: %s\n",cudaGetErrorString(err));
		free(h_m);
		free(h_y);
		cudaFree(d_y);
		cudaFree(d_m);
		cudaFree(d_x);
		free(h_x);
		return 1;
	}

	err = cudaMemcpy(h_y,d_y,bytes,cudaMemcpyDeviceToHost);
	if(err != cudaSuccess){
		printf("copy y failed: %s\n",cudaGetErrorString(err));
		free(h_y);
		free(h_m);
		cudaFree(d_y);
		cudaFree(d_m);
		cudaFree(d_x);
		free(h_x);
		return 1;
	}

	FILE* fm = fopen("m.bin","wb");
	if(fm == NULL){
		printf("打开m.bin失败\n");
		free(h_m);
		cudaFree(d_m);
		cudaFree(d_x);
		free(h_y);
		cudaFree(d_y);
		free(h_x);
		return 1;
	}
	size_t wrote_m = fwrite(h_m,sizeof(float),(size_t)ROWS,fm);
	fclose(fm);
	printf("wrote m.bin (%zu floats),fwrite returned %zu\n",(size_t)ROWS,wrote_m);

	FILE* fy = fopen("y.bin","wb");
	if(fy == NULL){
		printf("打开y.bin失败\n");
		free(h_y);
		free(h_m);
		cudaFree(d_y);
		cudaFree(d_m);
		cudaFree(d_x);
		free(h_x);
		return 1;
	}
	size_t wrote_y = fwrite(h_y,sizeof(float),n,fy);
	fclose(fy);
	printf("wrote y.bin (%zu bytes),fwrite returned %zu\n",bytes,wrote_y);

	printf("m[0]=%.6f m[1]=%.6f m[2]=%.6f\n",h_m[0],h_m[1],h_m[2]);
	printf("y[0][0]=%.9g y[0][%d]=%.9g ratio=%.3f\n",
		h_y[0],COLS - 1,h_y[COLS-1],h_y[0] / h_y[COLS - 1]);
	printf("y[1][0]=%.9g y[1][1]=%.9g\n",h_y[COLS],h_y[COLS + 1]);

	if(argc > 1 && argv[1][0] == 'b') bench(d_x,d_m,d_y);

	free(h_y);
	free(h_m);
	cudaFree(d_m);
	cudaFree(d_x);
	cudaFree(d_y);
	free(h_x);
	printf("ok\n");
	return 0;
}
