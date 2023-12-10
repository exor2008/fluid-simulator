extern "C" __global__ void divergence(float *div, const float *u, const float *v, int rows, int cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > 0 && y > 0 && x < cols - 1 && y < rows - 1)
    {
        int idx = y * cols + x;
        float du_dx = u[y * cols + x + 1] - u[idx];
        float dv_dy = v[(y + 1) * cols + x + 1] - v[idx];
        div[idx] = du_dx + dv_dy;
    }
}

extern "C" __global__ void pressure(float *pressure_a, float *pressure_b, const float *div, int rows, int cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && y > 0 && x < cols - 1 && y < rows - 1)
    {
        int idx = y * cols + x;

        float sum = pressure_b[(y + 1) * cols + x] + pressure_b[(y - 1) * cols + x] + pressure_b[y * cols + x + 1] + pressure_b[y * cols + x - 1];
        pressure_a[idx] = (sum - div[idx]) / 4;
    }
}

extern "C" __global__ void incompress(float *u, float *v, const float *pressure, int rows, int cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && y > 0 && x < cols - 1 && y < rows - 1)
    {
        int idx = y * cols + x;

        float dp_dx = pressure[y * cols + x + 1] - pressure[idx];
        float dp_dy = pressure[(y + 1) * cols + x] - pressure[idx];

        u[idx] -= dp_dx;
        v[idx] -= dp_dy;
    }
}