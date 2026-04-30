extern "C" __global__ void wiener_dummy(float* out, const float* in, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        // Dummy operation to validate kernel running
        out[idx] = in[idx] * 0.1f;
    }
}
