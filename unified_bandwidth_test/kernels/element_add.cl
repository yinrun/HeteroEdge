// Element-wise add kernel for bandwidth testing.
// Uses uchar16 (16 bytes/work-item) to match NPU's UFIXED_POINT_8 data type.
__kernel void element_add_uchar16(
    __global uchar16* C,
    __global const uchar16* A,
    __global const uchar16* B,
    int num_vecs) {
    int id = get_global_id(0);
    if (id < num_vecs) {
        C[id] = A[id] + B[id];
    }
}
