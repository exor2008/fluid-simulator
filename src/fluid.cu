extern "C" __global__ void divergence(float *div, const float *u, const float *v, int rows, int cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > 0 && y > 0 && x < cols - 1 && y < rows - 1)
    {
        int idx = y * cols + x;
        div[idx] = u[y * cols + x + 1] - u[idx] + v[(y + 1) * cols + x + 1] - v[idx];
    }
}
