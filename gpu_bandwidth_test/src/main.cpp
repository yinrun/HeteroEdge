#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

// 从文件读取 Kernel 源代码
static char* read_kernel_source(const char* filename, size_t* source_size) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("错误: 无法打开 kernel 文件: %s\n", filename);
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    char* source = (char*)malloc(file_size + 1);
    if (!source) {
        fclose(file);
        return NULL;
    }

    size_t read_size = fread(source, 1, file_size, file);
    source[read_size] = '\0';
    fclose(file);

    if (source_size) {
        *source_size = read_size;
    }

    return source;
}

// 获取高精度时间（秒）
static double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// 查询设备信息
static void print_device_info(cl_device_id device) {
    size_t name_size;
    clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &name_size);
    char* device_name = (char*)malloc(name_size);
    clGetDeviceInfo(device, CL_DEVICE_NAME, name_size, device_name, NULL);
    printf("设备名称: %s\n", device_name);
    free(device_name);

    cl_ulong global_mem_size;
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_mem_size, NULL);
    printf("全局内存大小: %.2f GB\n", global_mem_size / (1024.0 * 1024.0 * 1024.0));

    size_t max_work_group_size;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
    printf("最大工作组大小: %zu\n", max_work_group_size);

    cl_uint max_compute_units;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &max_compute_units, NULL);
    printf("计算单元数: %u\n", max_compute_units);

    // 查询内存带宽相关信息（如果支持）
    cl_ulong max_mem_alloc_size;
    clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &max_mem_alloc_size, NULL);
    printf("最大内存分配大小: %.2f GB\n", max_mem_alloc_size / (1024.0 * 1024.0 * 1024.0));
}

// 测试带宽
static double test_bandwidth(cl_command_queue queue, cl_kernel kernel, cl_mem src_buffer,
                            cl_mem dst_buffer, size_t data_size, size_t num_elements,
                            int num_iterations, const char* kernel_name, cl_device_id device) {
    size_t global_work_size = num_elements;

    // 查询设备的最佳工作组大小
    size_t max_work_group_size;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);

    // 选择合适的工作组大小（通常是 256 或 512，但不超过设备限制）
    size_t local_work_size = 256;
    if (local_work_size > max_work_group_size) {
        local_work_size = max_work_group_size;
    }

    // 确保 global_work_size 是 local_work_size 的倍数
    if (global_work_size % local_work_size != 0) {
        global_work_size = ((global_work_size / local_work_size) + 1) * local_work_size;
    }

    // 预热
    for (int i = 0; i < 3; i++) {
        clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    }
    clFinish(queue);

    // 实际测试
    double start_time = get_time();
    for (int i = 0; i < num_iterations; i++) {
        clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    }
    clFinish(queue);
    double end_time = get_time();

    double elapsed = end_time - start_time;
    double total_data = (double)data_size * num_iterations * 2;  // 读+写
    double bandwidth_gb_s = (total_data / (1024.0 * 1024.0 * 1024.0)) / elapsed;

    printf("  %s: %.2f GB/s (%.2f MB/s, %d 次迭代, %.3f 秒)\n",
           kernel_name, bandwidth_gb_s, bandwidth_gb_s * 1024.0, num_iterations, elapsed);

    return bandwidth_gb_s;
}

int main(int argc, char* argv[]) {
    printf("=== OpenCL GPU 带宽测试 ===\n\n");

    // 配置参数
    size_t data_size_mb = (argc > 1) ? atoi(argv[1]) : 1024;  // 默认 1GB
    size_t data_size = data_size_mb * 1024 * 1024;
    int num_iterations = (argc > 2) ? atoi(argv[2]) : 10;    // 默认 10 次迭代

    printf("测试配置:\n");
    printf("  数据大小: %zu MB\n", data_size_mb);
    printf("  迭代次数: %d\n", num_iterations);
    printf("\n");

    cl_int err;
    cl_platform_id platform = NULL;
    cl_device_id device = NULL;
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    cl_program program = NULL;
    cl_mem src_buffer = NULL;
    cl_mem dst_buffer = NULL;

    // 1. 获取平台和设备
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        printf("错误: 无法获取 OpenCL 平台\n");
        return 1;
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS || device == NULL) {
        printf("错误: 无法获取 GPU 设备\n");
        return 1;
    }

    printf("=== 设备信息 ===\n");
    print_device_info(device);
    printf("\n");

    // 2. 创建上下文和命令队列
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("错误: 无法创建上下文\n");
        return 1;
    }

    queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("错误: 无法创建命令队列\n");
        clReleaseContext(context);
        return 1;
    }

    // 3. 读取并编译 Kernel
    size_t kernel_source_size = 0;
    char* kernel_source = read_kernel_source("kernels/vector_copy.cl", &kernel_source_size);
    if (!kernel_source) {
        printf("错误: 无法读取 kernel 文件\n");
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_source_size, &err);
    free(kernel_source);
    if (err != CL_SUCCESS) {
        printf("错误: 无法创建程序\n");
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("错误: 程序编译失败\n");
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("编译日志:\n%s\n", log);
        free(log);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    // 4. 创建缓冲区
    src_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("错误: 无法创建源缓冲区\n");
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    dst_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("错误: 无法创建目标缓冲区\n");
        clReleaseMemObject(src_buffer);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    // 初始化源缓冲区数据
    void* host_data = malloc(data_size);
    memset(host_data, 0xAA, data_size);  // 填充测试数据
    clEnqueueWriteBuffer(queue, src_buffer, CL_TRUE, 0, data_size, host_data, 0, NULL, NULL);
    free(host_data);

    printf("=== 带宽测试结果 ===\n");

    // 5. 测试 kernel
    cl_kernel kernel = clCreateKernel(program, "vector_copy_float8", &err);
    if (err != CL_SUCCESS) {
        printf("错误: 无法创建 kernel\n");
        clReleaseMemObject(dst_buffer);
        clReleaseMemObject(src_buffer);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    size_t num_vecs = data_size / (sizeof(float) * 8);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &dst_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &src_buffer);
    clSetKernelArg(kernel, 2, sizeof(int), &num_vecs);

    double bandwidth = test_bandwidth(queue, kernel, src_buffer, dst_buffer,
                                     data_size, num_vecs, num_iterations, "vector_copy_float8", device);

    printf("\n=== 总结 ===\n");
    printf("实际带宽: %.2f GB/s (%.2f MB/s)\n", bandwidth, bandwidth * 1024.0);
    printf("\n注: 理论内存带宽 84.8 GB/s (LPDDR5X-5300, 4ch x 16bit)\n");
    printf("带宽利用率: %.1f%%\n", bandwidth / 84.8 * 100.0);
    printf("实际带宽受内存控制器效率、缓存、系统负载等因素影响\n");

    clReleaseKernel(kernel);

    // 清理资源
    clReleaseMemObject(dst_buffer);
    clReleaseMemObject(src_buffer);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
