// 简单的带宽测试 kernel
// 使用 float8 向量化类型最大化内存带宽
// 每个 work-item 处理 32 字节（8个float），最大化内存吞吐量
// 参考 llama.cpp 的优化思路：向量化、连续内存访问、最小化计算
__kernel void vector_copy_float8(__global float8* dst, __global const float8* src, int num_vecs) {
    int id = get_global_id(0);
    if (id < num_vecs) {
        dst[id] = src[id];  // 简单的向量复制，最大化内存带宽
    }
}
