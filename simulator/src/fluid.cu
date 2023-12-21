__constant__ int U_FEILD = 0;
__constant__ int V_FEILD = 1;
__constant__ int W_FEILD = 2;
__constant__ int S_FEILD = 3;
__constant__ float FRICTION = 1.0;
__constant__ float H = 1.0 / 110.0;

extern "C" __global__ void divergence(
    float *div,
    const float *u,
    const float *v,
    const float *w,
    const bool *block,
    int x_size,
    int y_size,
    int z_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x > 0 && y > 0 && z > 0 && x < x_size - 1 && y < y_size - 1 && z < z_size - 1)
    {
        int idx = (y + y_size * z) * x_size + x;

        if (!block[idx])
        {
            float du_dx = u[(y + y_size * z) * x_size + x + 1] - u[idx];
            float dv_dy = v[(y + 1 + y_size * z) * x_size + x] - v[idx];
            float dw_dz = w[(y + y_size * (z + 1)) * x_size + x] - w[idx];
            div[idx] = du_dx + dv_dy + dw_dz;
        }
    }
}

extern "C" __global__ void pressure(
    float *pressure_a,
    float *pressure_b,
    const float *div,
    const bool *block,
    int x_size,
    int y_size,
    int z_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x > 0 && y > 0 && z > 0 && x < x_size - 1 && y < y_size - 1 && z < z_size - 1)
    {
        int idx_z_r = (y + y_size * (z + 1)) * x_size + x;
        int idx_z_l = (y + y_size * (z - 1)) * x_size + x;
        int idx_y_r = (y + 1 + y_size * z) * x_size + x;
        int idx_y_l = (y - 1 + y_size * z) * x_size + x;
        int idx_x_r = (y + y_size * z) * x_size + x + 1;
        int idx_x_l = (y + y_size * z) * x_size + x - 1;

        int idx = (y + y_size * z) * x_size + x;
        if (!block[idx])
        {
            float pressure_z_r = block[idx_z_r] ? pressure_b[idx] : pressure_b[idx_z_r];
            float pressure_z_l = block[idx_z_l] ? pressure_b[idx] : pressure_b[idx_z_l];
            float pressure_y_r = block[idx_y_r] ? pressure_b[idx] : pressure_b[idx_y_r];
            float pressure_y_l = block[idx_y_l] ? pressure_b[idx] : pressure_b[idx_y_l];
            float pressure_x_r = block[idx_x_r] ? pressure_b[idx] : pressure_b[idx_x_r];
            float pressure_x_l = block[idx_x_l] ? pressure_b[idx] : pressure_b[idx_x_l];

            float sum = pressure_z_r +
                        pressure_z_l +
                        pressure_y_r +
                        pressure_y_l +
                        pressure_x_r +
                        pressure_x_l;

            pressure_a[idx] = (sum - div[idx]) / 6;
        }
    }
}

extern "C" __global__ void incompress(
    float *u,
    float *v,
    float *w,
    const float *pressure,
    const bool *block,
    int x_size,
    int y_size,
    int z_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x > 0 && y > 0 && z > 0 && x < x_size - 1 && y < y_size - 1 && z < z_size - 1)
    {
        int idx = (y + y_size * z) * x_size + x;
        if (!block[idx])
        {

            int idx_x = (y + y_size * z) * x_size + x + 1;
            int idx_y = (y + 1 + y_size * z) * x_size + x;
            int idx_z = (y + y_size * (z + 1)) * x_size + x;

            float pressure_x = block[idx_x] ? pressure[idx] : pressure[idx_x];
            float pressure_y = block[idx_y] ? pressure[idx] : pressure[idx_y];
            float pressure_z = block[idx_z] ? pressure[idx] : pressure[idx_z];

            float dp_dx = pressure_x - pressure[idx];
            float dp_dy = pressure_y - pressure[idx];
            float dp_dz = pressure_z - pressure[idx];

            u[idx] -= dp_dx;
            v[idx] -= dp_dy;
            w[idx] -= dp_dz;
        }
    }
}

extern "C" __device__ float avg_u(const float *u, int x_size, int y_size, int z_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x > 0 && y > 0 && z > 0 && x < x_size - 1 && y < y_size - 1 && z < z_size - 1)
    {
        return (u[(y + y_size * z) * x_size + x] +
                u[(y + y_size * z) * x_size + x + 1] +
                u[(y - 1 + y_size * z) * x_size + x] +
                u[(y - 1 + y_size * z) * x_size + x + 1] +
                u[(y + y_size * (z - 1)) * x_size + x] +
                u[(y + y_size * (z - 1)) * x_size + x + 1] +
                u[(y - 1 + y_size * (z - 1)) * x_size + x] +
                u[(y - 1 + y_size * (z - 1)) * x_size + x + 1]) *
               0.125;
    }
    else
    {
        return 0;
    }
}

extern "C" __device__ float avg_v(const float *v, int x_size, int y_size, int z_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x > 0 && y > 0 && z > 0 && x < x_size - 1 && y < y_size - 1 && z < z_size - 1)
    {
        return (v[(y + y_size * z) * x_size + x] +
                v[(y + y_size * z) * x_size + x - 1] +
                v[(y + 1 + y_size * z) * x_size + x] +
                v[(y + 1 + y_size * z) * x_size + x - 1] +
                v[(y + y_size * (z - 1)) * x_size + x] +
                v[(y + y_size * (z - 1)) * x_size + x - 1] +
                v[(y + 1 + y_size * (z - 1)) * x_size + x] +
                v[(y + 1 + y_size * (z - 1)) * x_size + x - 1]) *
               0.125;
    }
    else
    {
        return 0;
    }
}

extern "C" __device__ float avg_w(const float *w, int x_size, int y_size, int z_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x > 0 && y > 0 && z > 0 && x < x_size - 1 && y < y_size - 1 && z < z_size - 1)
    {
        return (w[(y + y_size * z) * x_size + x] +
                w[(y + y_size * z) * x_size + x - 1] +
                w[(y - 1 + y_size * z) * x_size + x] +
                w[(y - 1 + y_size * z) * x_size + x - 1] +
                w[(y + y_size * (z + 1)) * x_size + x] +
                w[(y + y_size * (z + 1)) * x_size + x - 1] +
                w[(y - 1 + y_size * (z + 1)) * x_size + x] +
                w[(y - 1 + y_size * (z + 1)) * x_size + x - 1]) *
               0.125;
    }
    else
    {
        return 0;
    }
}

extern "C" __device__ float sample_field(
    const float *u,
    const float *v, const float *w,
    const float *smoke,
    float x_shift,
    float y_shift,
    float z_shift,
    const int field_type,
    float h,
    int x_size,
    int y_size,
    int z_size)
{
    float h1 = 1.0 / h;
    float h2 = 0.5 * h;

    x_shift = max(min(x_shift, x_size * h), h);
    y_shift = max(min(y_shift, y_size * h), h);
    z_shift = max(min(z_shift, z_size * h), h);

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

    float xs = float(x_size - 1);
    float ys = float(y_size - 1);
    float zs = float(z_size - 1);

    float x0 = min(floor((x_shift - dx) * h1), xs);
    float tx = ((x_shift - dx) - x0 * h) * h1;
    float x1 = min(x0 + 1, xs);

    float y0 = min(floor((y_shift - dy) * h1), ys);
    float ty = ((y_shift - dy) - y0 * h) * h1;
    float y1 = min(y0 + 1, ys);

    float z0 = min(floor((z_shift - dz) * h1), zs);
    float tz = ((z_shift - dz) - z0 * h) * h1;
    float z1 = min(z0 + 1, zs);

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
        sy * sz * sx * field[(y_0 + y_size * z_0) * x_size + x_0] +
        sy * sz * tx * field[(y_0 + y_size * z_0) * x_size + x_1] +
        ty * sz * sx * field[(y_1 + y_size * z_0) * x_size + x_0] +
        ty * sz * tx * field[(y_1 + y_size * z_0) * x_size + x_1] +
        sy * tz * sx * field[(y_0 + y_size * z_1) * x_size + x_0] +
        sy * tz * tx * field[(y_0 + y_size * z_1) * x_size + x_1] +
        ty * tz * sx * field[(y_1 + y_size * z_1) * x_size + x_0] +
        ty * tz * tx * field[(y_1 + y_size * z_1) * x_size + x_1]);
}

extern "C" __global__ void advect_velocity(
    const float *u,
    const float *v,
    const float *w,
    float *new_u,
    float *new_v,
    float *new_w,
    const float *smoke,
    const bool *block,
    float dt,
    int x_size,
    int y_size,
    int z_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float h2 = 0.5 * H;

    if (x > 0 && y > 0 && z > 0 && x < x_size - 1 && y < y_size - 1 && z < z_size - 1)
    {
        int idx = (y + y_size * z) * x_size + x;

        if (!block[idx])
        {
            float avg_u_scalar = avg_u(u, x_size, y_size, z_size);
            float avg_v_scalar = avg_v(v, x_size, y_size, z_size);
            float avg_w_scalar = avg_w(w, x_size, y_size, z_size);

            // U
            float x_shift = x * H;
            float y_shift = y * H + h2;
            float z_shift = z * H + h2;

            float u_scalar = u[idx];
            float v_scalar = avg_v_scalar;
            float w_scalar = avg_w_scalar;

            x_shift -= dt * u_scalar;
            y_shift -= dt * v_scalar;
            z_shift -= dt * w_scalar;

            new_u[idx] = sample_field(u, v, w, smoke, x_shift, y_shift, z_shift, U_FEILD, H, x_size, y_size, z_size);

            // V
            x_shift = x * H + h2;
            y_shift = y * H;
            z_shift = z * H + h2;

            u_scalar = avg_u_scalar;
            v_scalar = v[idx];
            w_scalar = avg_w_scalar;

            x_shift -= dt * u_scalar;
            y_shift -= dt * v_scalar;
            z_shift -= dt * w_scalar;

            new_v[idx] = sample_field(u, v, w, smoke, x_shift, y_shift, z_shift, V_FEILD, H, x_size, y_size, z_size);

            // W
            x_shift = x * H + h2;
            y_shift = y * H + h2;
            z_shift = z * H;

            u_scalar = avg_u_scalar;
            v_scalar = avg_v_scalar;
            w_scalar = w[idx];

            x_shift -= dt * u_scalar;
            y_shift -= dt * v_scalar;
            z_shift -= dt * w_scalar;

            new_w[idx] = sample_field(u, v, w, smoke, x_shift, y_shift, z_shift, W_FEILD, H, x_size, y_size, z_size);
        }
    }
}

extern "C" __global__ void advect_smoke(
    const float *smoke,
    float *new_smoke,
    const float *u,
    const float *v,
    const float *w,
    const bool *block,
    float dt,
    int x_size,
    int y_size,
    int z_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x > 0 && y > 0 && z > 0 && x < x_size - 1 && y < y_size - 1 && z < z_size - 1)
    {
        float h2 = 0.5 * H;

        int idx = (y + y_size * z) * x_size + x;

        if (!block[idx])
        {
            float u_scalar = (u[idx] + u[(y + y_size * z) * x_size + x + 1]) * 0.5;
            float v_scalar = (v[idx] + v[(y + 1 + y_size * z) * x_size + x]) * 0.5;
            float w_scalar = (w[idx] + w[(y + y_size * (z + 1)) * x_size + x]) * 0.5;

            float x_shift = x * H + h2 - dt * u_scalar;
            float y_shift = y * H + h2 - dt * v_scalar;
            float z_shift = z * H + h2 - dt * w_scalar;

            float val = sample_field(u, v, w, smoke, x_shift, y_shift, z_shift, S_FEILD, H, x_size, y_size, z_size);
            new_smoke[idx] = val;
        }
    }
}

extern "C" __global__ void calc_borders(
    float *u,
    float *v,
    float *w,
    const float *normal_u,
    const float *normal_v,
    const float *normal_w,
    const bool *block,
    int x_size,
    int y_size,
    int z_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    float flipped_x;
    float flipped_y;
    float flipped_z;

    if (x > 0 && y > 0 && z > 0 && x < x_size - 1 && y < y_size - 1 && z < z_size - 1)
    {
        int idx = (y + y_size * z) * x_size + x;

        if (!block[idx])
        {
            if (normal_u[idx] != 0 || normal_v[idx] != 0 || normal_w[idx] != 0)
            {
                float dot = u[idx] * normal_u[idx] + v[idx] * normal_v[idx] + w[idx] * normal_w[idx];

                u[idx] = u[idx] - 2 * dot * normal_u[idx];
                v[idx] = v[idx] - 2 * dot * normal_v[idx];
                w[idx] = w[idx] - 2 * dot * normal_w[idx];
            }
        }
    }
}

extern "C" __global__ void constant(
    float *u,
    float *v,
    float *w,
    float *smoke,
    const bool *block,
    int x_size,
    int y_size,
    int z_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= 40 && x < 50 && y >= 45 && y < y_size - 45 && z >= 35 && z < z_size - 35)
    {
        int idx = (y + y_size * z) * x_size + x;

        if (!block[idx])
        {
            smoke[idx] = 1.0;
        }
    }

    if (x >= 2 && x <= 4 && y > 0 && z > 0 && y < y_size - 1 && z < z_size - 1)
    {
        int idx = (y + y_size * z) * x_size + x;
        u[idx] = 3.0;
    }

    if (x < 2 && y > 0 && z > 0 && y < y_size - 1 && z < z_size - 1)
    {
        int idx = (y + y_size * z) * x_size + x;
        u[idx] = 0;
    }
    if (y < 2 && x > 0 && z > 0 && x < x_size - 1 && z < z_size - 1)
    {
        int idx = (y + y_size * z) * x_size + x;
        v[idx] = 0;
    }
    if (z < 2 && y > 0 && x > 0 && y < y_size - 1 && x < x_size - 1)
    {
        int idx = (y + y_size * z) * x_size + x;
        w[idx] = 0;
    }
}