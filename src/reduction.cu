#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <string>

constexpr int N = (1 << 24)+1;
constexpr int REPEAT = 1000;
constexpr int BLOCK = 256;
constexpr int GRID = 1024;

#define CUDA_CHECK(call)\
	do{\
		cudaError_t err = (call);\
		if(err != cudaSuccess) {\
			throw std::runtime_error(\
				std::string("CUDA error at ") + __FILE__ + ":"\
				+std::to_string(__LINE__) + "->"\
				+cudaGetErrorString(err));\
		}\
	} while(0)

//--------设备内存:unique_ptr + 自定义 deleter ---------
struct CudaFreeDeleter{
	void operator()(void* p) const noexcept{
		if(p) cudaFree(p);
	}
};

template<typename T>
std::unique_ptr<T[],CudaFreeDeleter> make_dev_buf(size_t count) {
	T* raw = nullptr;
	CUDA_CHECK(cudaMalloc(&raw,count * sizeof(T)));
	return std::unique_ptr<T[],CudaFreeDeleter>(raw);
}

//---------事件:手写 RAII --------
class CudaEventGuard {
public:
	CudaEventGuard() {CUDA_CHECK(cudaEventCreate(&e_));}
	~CudaEventGuard() { if(e_) cudaEventDestroy(e_);};

	CudaEventGuard(const CudaEventGuard&) = delete;
	CudaEventGuard& operator=(const CudaEventGuard&) = delete;
	CudaEventGuard(CudaEventGuard&&) = delete;
	CudaEventGuard& operator=(CudaEventGuard&&) = delete;

	cudaEvent_t get() const {return e_;}
private:
	cudaEvent_t e_ = nullptr;
};

//GPU计算加法
//--------v0:单线程版本--------
__global__ void reduce_v0_kernel(const float* x,float* out,int n){
	float sum = 0.0f;
	for(int i = 0;i < n;i++){
		sum += x[i];
	}
	out[0] = sum;
}

//---------v1:多block + 块内shared树形规约-------
__global__ void reduce_kernel(const float* x,float* partial,int n){
	__shared__ float sh[BLOCK];
	int tid = threadIdx.x;
	int gtid = blockIdx.x * blockDim.x + tid;

	float local = 0.0f;
	for(int i = gtid;i < n;i += gridDim.x * blockDim.x){
		local += x[i];
	}

	sh[tid] = local;
	__syncthreads();

	for(int s = BLOCK >> 1;s > 0;s >>= 1){
		if(tid < s) sh[tid] += sh[tid + s];
		__syncthreads();
	}

	if(tid == 0) partial[blockIdx.x] = sh[0];
}

__global__ void reduce_final(const float* partial,float* out,int m){
	__shared__ float sh[BLOCK];
	int tid = threadIdx.x;

	float local = 0.0f;
	for(int i = tid;i < m;i += blockDim.x){
		local += partial[i];
	}

	sh[tid] = local;
	__syncthreads();

	for(int s = BLOCK >> 1;s > 0;s >>= 1){
		if(tid < s) sh[tid] += sh[tid + s];
		__syncthreads();
	}

	if(tid == 0) out[0] = sh[0];
}

//------------v2--------------
__global__ void reduce_kernel_v2(const float* x,float* partial,int n){
	__shared__ float sh[BLOCK / 32];
	int tid = threadIdx.x;
	int gtid = blockIdx.x * blockDim.x + tid;
	int lane = tid & 31;
	int wid = tid >> 5;

	float local = 0.0f;
	for(int i = gtid;i < n;i+= gridDim.x * blockDim.x){
		local += x[i];
	}

	for(int off = 16;off > 0;off >>= 1){
		local += __shfl_down_sync(0xffffffffu,local,off);
	}

	if(lane == 0) sh[wid] = local;
	__syncthreads();

	if(wid == 0){
		local = (lane < BLOCK / 32) ? sh[lane] : 0.0f;
		for(int off = 16;off > 0;off >>= 1){
			local += __shfl_down_sync(0xffffffffu,local,off);
		}
		if(lane == 0) partial[blockIdx.x] = local;
	}
}

__global__ void reduce_final_v2(const float* partial,float* out,int m){
        __shared__ float sh[BLOCK / 32];
        int tid = threadIdx.x;
        int lane = tid & 31;
        int wid = tid >> 5;

        float local = 0.0f;
        for(int i = tid;i < m;i += blockDim.x){
                local += partial[i];
        }

        for(int off = 16;off > 0;off >>= 1){
                local += __shfl_down_sync(0xffffffffu,local,off);
        }

        if(lane == 0) sh[wid] = local;
        __syncthreads();

        if(wid == 0){
                local = (lane < BLOCK / 32) ? sh[lane] : 0.0f;
                for(int off = 16;off > 0;off >>= 1){
                        local += __shfl_down_sync(0xffffffffu,local,off);
                }
                if(lane == 0) out[0] = local;
        }
}

void launch(int ver,const float* d_x,float* d_partial,float* d_out,int n){
	if(ver == 0){
		reduce_v0_kernel<<<1,1>>>(d_x,d_out,n);
	} else if(ver == 2){
		reduce_kernel_v2<<<GRID,BLOCK>>>(d_x,d_partial,n);
		reduce_final_v2<<<1,BLOCK>>>(d_partial,d_out,GRID);
	} else if(ver == 3){
		reduce_kernel<<<GRID,BLOCK>>>(d_x,d_partial,n);
	} else {
		reduce_kernel<<<GRID,BLOCK>>>(d_x,d_partial,n);
		reduce_final<<<1,BLOCK>>>(d_partial,d_out,GRID);
	}
}

int run(int argc,char** argv){
	int ver = (argc > 1) ? atoi(argv[1]) : 1;
	int reps = (argc > 2) ? atoi(argv[2]) : REPEAT;
	if(reps <= 0){
		printf("reps must be > 0,got %d\n",reps);
		return 1;
	}
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
	auto d_x = make_dev_buf<float>(N);
	auto d_out = make_dev_buf<float>(1);
	auto d_partial = make_dev_buf<float>(GRID);

	//拷贝数据 host->device
	CUDA_CHECK(cudaMemcpy(d_x.get(),h_x,bytes,cudaMemcpyHostToDevice));

	//创建计时标记start，stop
	CudaEventGuard start,stop;

	//预热，让GPU进入稳定状态，避免冷启动开销影响时间
	launch(ver,d_x.get(),d_partial.get(),d_out.get(),N);
	CUDA_CHECK(cudaGetLastError());

	//重复跑
	CUDA_CHECK(cudaEventRecord(start.get()));
	for(int r = 0;r < reps;r++){
		launch(ver,d_x.get(),d_partial.get(),d_out.get(),N);
	}
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaEventRecord(stop.get()));
	CUDA_CHECK(cudaEventSynchronize(stop.get()));

	CUDA_CHECK(cudaMemcpy(h_out,d_out.get(),sizeof(float),cudaMemcpyDeviceToHost));

	//计算时间
	float total_ms = 0.0f;
	CUDA_CHECK(cudaEventElapsedTime(&total_ms,start.get(),stop.get()));
	float ms = total_ms / reps;

	//输出结果
	double gb = 1.0 * (double)bytes / 1e9;
	printf("N = %d ver = v%d REPEAT = %d\n",N,ver,reps);
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
	free(h_x);
	free(h_out);

	printf("ok\n");
	return 0;
}

int main(int argc,char** argv){
	try{
		return run(argc,argv);
	} catch (const std::exception& e) {
		printf("FATAL: %s\n",e.what());
		return 1;
	}
}
