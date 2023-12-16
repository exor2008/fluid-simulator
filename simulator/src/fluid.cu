__constant__ int U_FEILD = 0;
__constant__ int V_FEILD = 1;
__constant__ int W_FEILD = 2;
__constant__ int S_FEILD = 3;

extern "C" __global__ void divergence(float *div, const float *u, const float *v, const float *w, int rows, int cols, int depths)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x > 0 && y > 0 && z > 0 && x < cols - 1 && y < rows - 1 && z < depths - 1)
    {
        int idx = (y + depths * z) * cols + x;

        float du_dx = u[(y + depths * z) * cols + x + 1] - u[idx];
        float dv_dy = v[(y + 1 + depths * z) * cols + x] - v[idx];
        float dw_dz = w[(y + depths * (z + 1)) * cols + x] - w[idx];
        div[idx] = du_dx + dv_dy + dw_dz;
    }
}

extern "C" __global__ void pressure(float *pressure_a, float *pressure_b, const float *div, int rows, int cols, int depths)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x > 0 && y > 0 && z > 0 && x < cols - 1 && y < rows - 1 && z < depths - 1)
    {
        int idx = (y + depths * z) * cols + x;

        float sum = (pressure_b[(y + depths * (z + 1)) * cols + x] +
                     pressure_b[(y + depths * (z - 1)) * cols + x] +
                     pressure_b[(y + 1 + depths * z) * cols + x] +
                     pressure_b[(y - 1 + depths * z) * cols + x] +
                     pressure_b[(y + depths * z) * cols + x + 1] +
                     pressure_b[(y + depths * z) * cols + x - 1]);
        pressure_a[idx] = (sum - div[idx]) / 6;
    }
}

extern "C" __global__ void incompress(float *u, float *v, float *w, const float *pressure, int rows, int cols, int depths)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x > 0 && y > 0 && z > 0 && x < cols - 1 && y < rows - 1 && z < depths - 1)
    {
        int idx = (y + depths * z) * cols + x;

        float dp_dx = pressure[(y + depths * z) * cols + x + 1] - pressure[idx];
        float dp_dy = pressure[(y + 1 + depths * z) * cols + x] - pressure[idx];
        float dp_dz = pressure[(y + depths * (z + 1)) * cols + x] - pressure[idx];

        u[idx] -= dp_dx;
        v[idx] -= dp_dy;
        w[idx] -= dp_dz;
    }
}

extern "C" __device__ float avg_u(const float *u, int rows, int cols, int depths)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x > 0 && y > 0 && z > 0 && x < cols - 1 && y < rows - 1 && z < depths - 1)
    {
        return (u[(y + depths * z) * cols + x] +
                u[(y + depths * z) * cols + x + 1] +
                u[(y - 1 + depths * z) * cols + x] +
                u[(y - 1 + depths * z) * cols + x + 1] +
                u[(y + depths * (z - 1)) * cols + x] +
                u[(y + depths * (z - 1)) * cols + x + 1] +
                u[(y - 1 + depths * (z - 1)) * cols + x] +
                u[(y - 1 + depths * (z - 1)) * cols + x + 1]) *
               0.125;
    }
    else
    {
        return 0;
    }
}

extern "C" __device__ float avg_v(const float *v, int rows, int cols, int depths)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x > 0 && y > 0 && z > 0 && x < cols - 1 && y < rows - 1 && z < depths - 1)
    {
        return (v[(y + depths * z) * cols + x] +
                v[(y + depths * z) * cols + x - 1] +
                v[(y + 1 + depths * z) * cols + x] +
                v[(y + 1 + depths * z) * cols + x - 1] +
                v[(y + depths * (z - 1)) * cols + x] +
                v[(y + depths * (z - 1)) * cols + x - 1] +
                v[(y + 1 + depths * (z - 1)) * cols + x] +
                v[(y + 1 + depths * (z - 1)) * cols + x - 1]) *
               0.125;
    }
    else
    {
        return 0;
    }
}

extern "C" __device__ float avg_w(const float *w, int rows, int cols, int depths)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x > 0 && y > 0 && z > 0 && x < cols - 1 && y < rows - 1 && z < depths - 1)
    {
        return (w[(y + depths * z) * cols + x] +
                w[(y + depths * z) * cols + x - 1] +
                w[(y - 1 + depths * z) * cols + x] +
                w[(y - 1 + depths * z) * cols + x - 1] +
                w[(y + depths * (z + 1)) * cols + x] +
                w[(y + depths * (z + 1)) * cols + x - 1] +
                w[(y - 1 + depths * (z + 1)) * cols + x] +
                w[(y - 1 + depths * (z + 1)) * cols + x - 1]) *
               0.125;
    }
    else
    {
        return 0;
    }
}

extern "C" __device__ float sample_field(const float *u, const float *v, const float *w, const float *smoke, float x_shift, float y_shift, float z_shift, const int field_type, float h, int rows, int cols, int depths)
{
    float h1 = 1.0 / h;
    float h2 = 0.5 * h;

    x_shift = max(min(x_shift, cols * h), h);
    y_shift = max(min(y_shift, rows * h), h);
    z_shift = max(min(z_shift, depths * h), h);

    float dx = 0.0;
    float dy = 0.0;
    float dz = 0.0;
    const float *field;

    if (field_type == U_FEILD)
    {
        field = u;
        dy = h2;
        dz = h2;
    }
    else if (field_type == V_FEILD)
    {
        field = v;
        dx = h2;
        dz = h2;
    }
    else if (field_type == W_FEILD)
    {
        field = w;
        dx = h2;
        dy = h2;
    }
    else
    {
        field = smoke;
        dx = h2;
        dy = h2;
        dz = h2;
    }

    float col = float(cols - 1);
    float row = float(rows - 1);
    float depth = float(depths - 1);

    float x0 = min(floor((x_shift - dx) * h1), col);
    float tx = ((x_shift - dx) - x0 * h) * h1;
    float x1 = min(x0 + 1, col);

    float y0 = min(floor((y_shift - dy) * h1), row);
    float ty = ((y_shift - dy) - y0 * h) * h1;
    float y1 = min(y0 + 1, row);

    float z0 = min(floor((z_shift - dz) * h1), depth);
    float tz = ((z_shift - dz) - z0 * h) * h1;
    float z1 = min(z0 + 1, depth);

    float sx = 1.0 - tx;
    float sy = 1.0 - ty;
    float sz = 1.0 - tz;

    int x_0 = int(x0);
    int x_1 = int(x1);
    int y_0 = int(y0);
    int y_1 = int(y1);
    int z_0 = int(z0);
    int z_1 = int(z1);

    return (
        sy * sz * sx * field[(y_0 + depths * z_0) * cols + x_0] +
        sy * sz * tx * field[(y_0 + depths * z_0) * cols + x_1] +
        ty * sz * sx * field[(y_1 + depths * z_0) * cols + x_0] +
        ty * sz * tx * field[(y_1 + depths * z_0) * cols + x_1] +
        sy * tz * sx * field[(y_0 + depths * z_1) * cols + x_0] +
        sy * tz * tx * field[(y_0 + depths * z_1) * cols + x_1] +
        ty * tz * sx * field[(y_1 + depths * z_1) * cols + x_0] +
        ty * tz * tx * field[(y_1 + depths * z_1) * cols + x_1]);
}

extern "C" __global__ void advect_velocity(const float *u, const float *v, const float *w, float *new_u, float *new_v, float *new_w, const float *smoke, float dt, float h, int rows, int cols, int depths)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float h2 = 0.5 * h;

    if (x > 0 && y > 0 && z > 0 && x < cols - 1 && y < rows - 1 && z < depths - 1)
    {
        int idx = (y + depths * z) * cols + x;

        float avg_u_scalar = avg_u(u, rows, cols, depths);
        float avg_v_scalar = avg_v(v, rows, cols, depths);
        float avg_w_scalar = avg_w(w, rows, cols, depths);

        // U
        float x_shift = x * h;
        float y_shift = y * h + h2;
        float z_shift = z * h + h2;

        float u_scalar = u[idx];
        float v_scalar = avg_v_scalar;
        float w_scalar = avg_w_scalar;

        x_shift -= dt * u_scalar;
        y_shift -= dt * v_scalar;
        z_shift -= dt * w_scalar;

        new_u[idx] = sample_field(u, v, w, smoke, x_shift, y_shift, z_shift, U_FEILD, h, rows, cols, depths);

        // V
        x_shift = x * h + h2;
        y_shift = y * h;
        z_shift = z * h + h2;

        u_scalar = avg_u_scalar;
        v_scalar = v[idx];
        w_scalar = avg_w_scalar;

        x_shift -= dt * u_scalar;
        y_shift -= dt * v_scalar;
        z_shift -= dt * w_scalar;

        new_v[idx] = sample_field(u, v, w, smoke, x_shift, y_shift, z_shift, V_FEILD, h, rows, cols, depths);

        // W
        x_shift = x * h + h2;
        y_shift = y * h + h2;
        z_shift = z * h;

        u_scalar = avg_u_scalar;
        v_scalar = avg_v_scalar;
        w_scalar = w[idx];

        x_shift -= dt * u_scalar;
        y_shift -= dt * v_scalar;
        z_shift -= dt * w_scalar;

        new_w[idx] = sample_field(u, v, w, smoke, x_shift, y_shift, z_shift, W_FEILD, h, rows, cols, depths);
    }
}

extern "C" __global__ void advect_smoke(const float *smoke, float *new_smoke, const float *u, const float *v, const float *w, float dt, float h, int rows, int cols, int depths)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x > 0 && y > 0 && z > 0 && x < cols - 1 && y < rows - 1 && z < depths - 1)
    {
        float h2 = 0.5 * h;

        int idx = (y + depths * z) * cols + x;

        float u_scalar = (u[idx] + u[(y + depths * z) * cols + x + 1]) * 0.5;
        float v_scalar = (v[idx] + v[(y + 1 + depths * z) * cols + x]) * 0.5;
        float w_scalar = (w[idx] + w[(y + depths * (z + 1)) * cols + x]) * 0.5;

        float x_shift = x * h + h2 - dt * u_scalar;
        float y_shift = y * h + h2 - dt * v_scalar;
        float z_shift = z * h + h2 - dt * w_scalar;

        float val = sample_field(u, v, w, smoke, x_shift, y_shift, z_shift, S_FEILD, h, rows, cols, depths);
        new_smoke[idx] = val;
    }
}

extern "C" __global__ void constant(float *u, float *w, float *smoke, int rows, int cols, int depths)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x == 20 && z == 10 && y > 10 && y < cols - 10)
    {
        w[(y + depths * z) * cols + x] = 2.0;
        smoke[(y + depths * z) * cols + x] = 1.0;
    }
    else if (x == 40 && z == 10 && y > 10 && y < cols - 10)
    {
        w[(y + depths * z) * cols + x] = 2.0;
        smoke[(y + depths * z) * cols + x] = 1.0;
    }
    else if (x == 60 && z == 10 && y > 10 && y < cols - 10)
    {
        w[(y + depths * z) * cols + x] = 2.0;
        smoke[(y + depths * z) * cols + x] = 1.0;
    }
    else if (x == 80 && z == 10 && y > 10 && y < cols - 10)
    {
        w[(y + depths * z) * cols + x] = 2.0;
        smoke[(y + depths * z) * cols + x] = 1.0;
    }

    if (x > 40 && x <= 60 && y > 40 && y <= 60 && z > 10 && z < 20)
    {
        w[(y + depths * z) * cols + x] = 4.0;
        smoke[(y + depths * z) * cols + x] = 1.0;
    }
}