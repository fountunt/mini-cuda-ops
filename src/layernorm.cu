#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

#define COLS 4096
#define ROWS 4096
#define BLOCK 256
#define EPS 1e-5f

//CUDA API错误检查宏
#define CUDA_CHECK(call)\
    do {\
        cudaError_t err_ = (call);\
        if(err_ != cudaSuccess){\
            char msg_[512];\
            snprintf(msg_,sizeof(msg_),"CUDA error at %s:%d -> %s",\
                    __FILE__,__LINE__,cudaGetErrorString(err_));\
            throw std::runtime_error(msg_);\
        }\
} while(0)

//自定义删除器 让std::unique_ptr用cudaFree
struct CudaDeleter{
    void operator()(float *p) const {cudaFree(p);}
};
using DeviceFloats = std::unique_ptr<float,CudaDeleter>;

//分配bytes个float给设备内存 返回cudaFree的RAII句柄
static DeviceFloats make_device(size_t bytes){
    float *p = nullptr;
    CUDA_CHECK(cudaMalloc(&p,bytes));
    return DeviceFloats(p);
}

//写数据函数
static void write_bin(const char* path,const void* data,size_t bytes){
    FILE *fp = fopen(path,"wb");
    if(fp == nullptr) {
        throw std::runtime_error(std::string("open failed: ") + path);
    }
    size_t w = fwrite(data,1,bytes,fp);
    fclose(fp);
    if(w != bytes) {
        throw std::runtime_error(std::string("short write: ") + path);
    }
}

__global__ void layernorm_kernel(const float* __restrict__ x,
                                 const float* __restrict__ gamma,
                                 const float* __restrict__ beta,
                                 float* __restrict__ y)
{
    __shared__ float srow[COLS];
    __shared__ float sh[BLOCK];

    const int tid = threadIdx.x;
    const int r = blockIdx.x;
    const float *xr = x + (size_t)r * COLS;
    float *yr = y + (size_t)r * COLS;

    for(int j = tid;j < COLS;j += BLOCK){
        srow[j] = xr[j];
    }
    __syncthreads();

    float local = 0.0f;
    for(int j = tid;j < COLS;j += BLOCK){
        local += srow[j];
    }

    sh[tid] = local;
    __syncthreads();
    for(int s = BLOCK / 2;s > 0;s >>= 1){
        if(tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }
    const float mu = sh[0] / (float)COLS;

    __syncthreads();

    float local2 = 0.0f;
    for(int j = tid;j < COLS;j += BLOCK){
        const float d = srow[j] - mu;
        local2 += d * d;
    }
    sh[tid] = local2;
    __syncthreads();

    for(int s = BLOCK / 2;s > 0;s >>= 1){
        if(tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }
    const float inv = rsqrtf(sh[0] / (float)COLS + EPS);

    for(int j = tid;j < COLS;j += BLOCK){
        yr[j] = (srow[j] - mu) * inv * gamma[j] + beta[j];
    }
}

void bench(const float *d_x,const float *d_g,
            const float *d_b,float *d_y){
    const int WARM = 100;
    const int REPS = 500;

    for(int i = 0;i < WARM;i++){
        layernorm_kernel<<<ROWS,BLOCK>>>(d_x,d_g,d_b,d_y);
    }
    CUDA_CHECK(cudaGetLastError());//主动挂起kernel启动错误

    //Create 和 Record配套记录一段时间
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for(int i = 0;i < REPS;i++){
        layernorm_kernel<<<ROWS,BLOCK>>>(d_x,d_g,d_b,d_y);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms,start,stop));

    const double per_call = ms / REPS;
    const double gb = 2.0 * (double)ROWS * (double)COLS * sizeof(float);

    printf("layernorm:\n");
    printf("per-call = %.4f ms\n", per_call);
    printf("bw = %.1f GB/s\n", gb / (per_call * 1e-3));
    printf("REPEAT = %d\n", REPS);
    printf("window = %.1fms\n", per_call * REPS);
    printf("total = %.1f ms\n", per_call * (WARM + REPS));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main(int argc,char** argv){
    try{
        const size_t n = (size_t)ROWS * COLS;
        const size_t bytes = n * sizeof(float);
        const size_t gbytes = (size_t)COLS * sizeof(float);

        std::vector<float> h_x(n),h_y(n),h_g(COLS),h_b(COLS);

        //生成测试数据
        for(int r = 0;r < ROWS;r++){
            for(int c = 0;c < COLS;c++){
                const size_t idx = (size_t)r * COLS + c;
                if(r == 0){
                    h_x[idx] = -120.0f - 0.001f * (float)c;
                } else {
                    h_x[idx] = (float)((c * 37 + r * 101) % 300) - 150.0f;
                }
            }
        }
        for(int c = 0;c < COLS;c++){
            h_g[c] = 1.0f + 0.001f * (float)(c % 100);
            h_b[c] = 0.01f * (float)(c % 50);
        }

        //记录测试结果 用于和py程序结果对比
        write_bin("ln_x.bin",h_x.data(),bytes);
        write_bin("ln_g.bin",h_g.data(),gbytes);
        write_bin("ln_b.bin",h_b.data(),gbytes);

        //分配内存
        DeviceFloats d_x = make_device(bytes);
        DeviceFloats d_y = make_device(bytes);
        DeviceFloats d_g = make_device(gbytes);
        DeviceFloats d_b = make_device(gbytes);

        //将数据复制到GPU
        CUDA_CHECK(cudaMemcpy(d_x.get(),h_x.data(),bytes,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_g.get(),h_g.data(),gbytes,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b.get(),h_b.data(),gbytes,cudaMemcpyHostToDevice));

        layernorm_kernel<<<ROWS,BLOCK>>>(d_x.get(),d_g.get(),d_b.get(),d_y.get());
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(h_y.data(),d_y.get(),bytes,cudaMemcpyDeviceToHost));
        write_bin("ln_y.bin",h_y.data(),bytes);

        //人工观察
        printf("y[0][0]=%.6f y[0][1]=%.6f\n",h_y[0],h_y[1]);
        printf("y[1][0]=%.6f y[1][1]=%.6f\n",h_y[COLS],h_y[COLS+1]);

        if(argc > 1 && argv[1][0] == 'b'){
            bench(d_x.get(),d_g.get(),d_b.get(),d_y.get());
        }
    }
    catch(const std::exception& e){
        fprintf(stderr,"%s\n",e.what());
        return 1;
    }
    return 0;
}