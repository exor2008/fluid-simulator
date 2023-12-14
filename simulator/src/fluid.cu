__constant__ int U_FEILD = 0;
__constant__ int V_FEILD = 1;
__constant__ int S_FEILD = 2;

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

extern "C" __device__ float avg_u(const float *u, int rows, int cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && y > 0 && x < cols - 1 && y < rows - 1)
    {
        // ***
        // *++
        // *++
        return (u[(y - 1) * cols + x] + u[y * cols + x] + u[(y - 1) * cols + x + 1] + u[y * cols + x + 1]) * 0.25;
    }
    else
    {
        return 0;
    }
}

extern "C" __device__ float avg_v(const float *v, int rows, int cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && y > 0 && x < cols - 1 && y < rows - 1)
    {
        // ++*
        // ++*
        // ***
        return (v[y * cols + x - 1] + v[y * cols + x] + v[(y + 1) * cols + x - 1] + v[(y + 1) * cols + x]) * 0.25;
    }
    else
    {
        return 0;
    }
}

extern "C" __device__ float sample_field(const float *u, const float *v, const float *smoke, float x_shift, float y_shift, const int field_type, float h, int rows, int cols)
{
    float h1 = 1.0 / h;
    float h2 = 0.5 * h;

    x_shift = max(min(x_shift, cols * h), h);
    y_shift = max(min(y_shift, rows * h), h);

    float dx = 0.0;
    float dy = 0.0;
    const float *field;

    if (field_type == 0)
    {
        field = u;
        dy = h2;
    }
    else if (field_type == 1)
    {
        field = v;
        dx = h2;
    }
    else
    {
        field = smoke;
        dx = h2;
        dy = h2;
    }

    float col = float(cols - 1);
    float row = float(rows - 1);

    float x0 = min(floor((x_shift - dx) * h1), col);
    float tx = ((x_shift - dx) - x0 * h) * h1;
    float x1 = min(x0 + 1, col);

    float y0 = min(floor((y_shift - dy) * h1), row);
    float ty = ((y_shift - dy) - y0 * h) * h1;
    float y1 = min(y0 + 1, row);

    float sx = 1.0 - tx;
    float sy = 1.0 - ty;

    int x_0 = int(x0);
    int x_1 = int(x1);
    int y_0 = int(y0);
    int y_1 = int(y1);

    return (
        sx * sy * field[y_0 * cols + x_0] + tx * sy * field[y_0 * cols + x_1] + tx * ty * field[y_1 * cols + x_1] + sx * ty * field[y_0 * cols + x_1]);
}

extern "C" __global__ void advect_velocity(const float *u, const float *v, float *new_u, float *new_v, const float *smoke, float dt, float h, int rows, int cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && y > 0 && x < cols - 1 && y < rows - 1)
    {
        float h2 = 0.5 * h;

        float x_shift = x * h;
        float y_shift = y * h + h2;

        float u_scalar = u[y * cols + x];
        float v_scalar = avg_v(v, rows, cols);
        x_shift -= dt * u_scalar;
        y_shift -= dt * v_scalar;

        new_u[y * cols + x] = sample_field(u, v, smoke, x_shift, y_shift, U_FEILD, h, rows, cols);

        x_shift = x * h + h2;
        y_shift = y * h;

        u_scalar = avg_u(u, rows, cols);
        v_scalar = v[y * cols + x];
        x_shift -= dt * u_scalar;
        y_shift -= dt * v_scalar;

        new_v[y * cols + x] = sample_field(u, v, smoke, x_shift, y_shift, V_FEILD, h, rows, cols);
    }
}

extern "C" __global__ void advect_smoke(const float *smoke, float *new_smoke, const float *u, const float *v, float dt, float h, int rows, int cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && y > 0 && x < cols - 1 && y < rows - 1)
    {
        float h2 = 0.5 * h;

        float u_scalar = (u[y * cols + x] + u[y * cols + x + 1]) * 0.5;
        float v_scalar = (v[y * cols + x] + v[(y + 1) * cols + x]) * 0.5;
        float x_shift = x * h + h2 - dt * u_scalar;
        float y_shift = y * h + h2 - dt * v_scalar;

        float val = sample_field(u, v, smoke, x_shift, y_shift, S_FEILD, h, rows, cols);
        new_smoke[y * cols + x] = val;
    }
}
