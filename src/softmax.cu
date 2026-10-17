#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

constexpr int ROWS = 4096;
constexpr int COLS = 4096;
constexpr int BLOCK = 256;

__global__ void row_max_kernel(const float* x,float* m){

}

int main(){
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

	free(h_x);
	printf("ok\n");
	return 0;
}
