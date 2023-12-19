#![feature(iter_array_chunks)]

pub use cudarc::{driver::*, nvrtc::compile_ptx};
use std::{mem::swap, sync::Arc};
pub mod gltf_reader;
pub mod raster;

const ITERATIONS: usize = 40;
const H: f32 = 1.0 / 100.0;

pub struct Fluid {
    u_dev: CudaSlice<f32>,
    v_dev: CudaSlice<f32>,
    w_dev: CudaSlice<f32>,
    new_u_dev: CudaSlice<f32>,
    new_v_dev: CudaSlice<f32>,
    new_w_dev: CudaSlice<f32>,
    smoke_dev: CudaSlice<f32>,
    new_smoke_dev: CudaSlice<f32>,
    div_dev: CudaSlice<f32>,
    pressure_a_dev: CudaSlice<f32>,
    pressure_b_dev: CudaSlice<f32>,
    normal_u_dev: CudaSlice<f32>,
    normal_v_dev: CudaSlice<f32>,
    normal_w_dev: CudaSlice<f32>,
    x_size: usize,
    y_size: usize,
    z_size: usize,
}

impl Fluid {
    pub fn new(
        dev: Arc<CudaDevice>,
        x_size: usize,
        y_size: usize,
        z_size: usize,
    ) -> Result<Self, DriverError> {
        let size = x_size * y_size * z_size;

        let u_host = vec![0f32; size];
        let v_host = vec![0f32; size];
        let w_host = vec![0f32; size];
        let new_u_host = vec![0f32; size];
        let new_v_host = vec![0f32; size];
        let new_w_host = vec![0f32; size];
        let smoke_host = vec![0f32; size];
        let new_smoke_host = vec![0f32; size];
        let div_host = vec![0f32; size];
        let pressure_a_host = vec![0f32; size];
        let pressure_b_host = vec![0f32; size];
        let normal_u_host = vec![0f32; size];
        let normal_v_host = vec![0f32; size];
        let normal_w_host = vec![0f32; size];

        let u_dev = dev.htod_copy(u_host)?;
        let v_dev = dev.htod_copy(v_host)?;
        let w_dev = dev.htod_copy(w_host)?;
        let new_u_dev = dev.htod_copy(new_u_host)?;
        let new_v_dev = dev.htod_copy(new_v_host)?;
        let new_w_dev = dev.htod_copy(new_w_host)?;
        let smoke_dev = dev.htod_copy(smoke_host)?;
        let new_smoke_dev = dev.htod_copy(new_smoke_host)?;
        let div_dev = dev.htod_copy(div_host)?;
        let pressure_a_dev = dev.htod_copy(pressure_a_host)?;
        let pressure_b_dev = dev.htod_copy(pressure_b_host)?;
        let normal_u_dev = dev.htod_copy(normal_u_host)?;
        let normal_v_dev = dev.htod_copy(normal_v_host)?;
        let normal_w_dev = dev.htod_copy(normal_w_host)?;

        let fluid = Fluid {
            u_dev,
            v_dev,
            w_dev,
            new_u_dev,
            new_v_dev,
            new_w_dev,
            smoke_dev,
            new_smoke_dev,
            div_dev,
            pressure_a_dev,
            pressure_b_dev,
            normal_u_dev,
            normal_v_dev,
            normal_w_dev,
            x_size,
            y_size,
            z_size,
        };

        Ok(fluid)
    }

    pub fn step(
        &mut self,
        dev: Arc<CudaDevice>,
        cfg: LaunchConfig,
        dt: f32,
    ) -> Result<(), DriverError> {
        unsafe {
            // Constant power
            let constant = dev.get_func("fluid", "constant").unwrap();
            constant.launch(
                cfg,
                (
                    &mut self.u_dev,
                    &mut self.w_dev,
                    &mut self.smoke_dev,
                    self.x_size,
                    self.y_size,
                    self.z_size,
                ),
            )?;
            dev.synchronize()?;

            // Divergence
            let divergence = dev.get_func("fluid", "divergence").unwrap();
            divergence.launch(
                cfg,
                (
                    &mut self.div_dev,
                    &self.u_dev,
                    &self.v_dev,
                    &self.w_dev,
                    self.x_size,
                    self.y_size,
                    self.z_size,
                ),
            )?;
            dev.synchronize()?;

            // Pressure Jacobi
            for _ in 0..ITERATIONS {
                let pressure = dev.get_func("fluid", "pressure").unwrap();

                pressure.launch(
                    cfg,
                    (
                        &mut self.pressure_a_dev,
                        &mut self.pressure_b_dev,
                        &self.div_dev,
                        self.x_size,
                        self.y_size,
                        self.z_size,
                    ),
                )?;
                dev.synchronize()?;

                swap(&mut self.pressure_a_dev, &mut self.pressure_b_dev);
            }

            // Solve Incompressibility
            let incompress = dev.get_func("fluid", "incompress").unwrap();
            incompress.launch(
                cfg,
                (
                    &mut self.u_dev,
                    &mut self.v_dev,
                    &mut self.w_dev,
                    &self.pressure_a_dev,
                    self.x_size,
                    self.y_size,
                    self.z_size,
                ),
            )?;
            dev.synchronize()?;

            // Advect velocity
            let advect_velocity = dev.get_func("fluid", "advect_velocity").unwrap();
            advect_velocity.launch(
                cfg,
                (
                    &self.u_dev,
                    &self.v_dev,
                    &self.w_dev,
                    &mut self.new_u_dev,
                    &mut self.new_v_dev,
                    &mut self.new_w_dev,
                    &self.smoke_dev,
                    dt,
                    H,
                    self.x_size,
                    self.y_size,
                    self.z_size,
                ),
            )?;
            swap(&mut self.u_dev, &mut self.new_u_dev);
            swap(&mut self.v_dev, &mut self.new_v_dev);
            swap(&mut self.w_dev, &mut self.new_w_dev);
            dev.synchronize()?;

            // Advect smoke
            let advect_smoke = dev.get_func("fluid", "advect_smoke").unwrap();
            advect_smoke.launch(
                cfg,
                (
                    &self.smoke_dev,
                    &mut self.new_smoke_dev,
                    &self.u_dev,
                    &self.v_dev,
                    &self.w_dev,
                    dt,
                    H,
                    self.x_size,
                    self.y_size,
                    self.z_size,
                ),
            )?;
            swap(&mut self.smoke_dev, &mut self.new_smoke_dev);
        };
        Ok(())
    }

    pub fn smoke(&self, dev: Arc<CudaDevice>) -> Result<Vec<f32>, DriverError> {
        let result = dev.sync_reclaim(self.smoke_dev.clone())?;
        // let mut result = dev.sync_reclaim(self.normal_u_dev.clone())?;

        // for r in result.iter_mut() {
        //     *r += 1.0;
        //     *r /= 2.0;
        // }

        Ok(result)
    }
}

pub fn get_device(ordinal: usize) -> Result<Arc<CudaDevice>, DriverError> {
    CudaDevice::new(ordinal)
}

pub fn get_fluid(
    dev: Arc<CudaDevice>,
    x_size: usize,
    y_size: usize,
    z_size: usize,
) -> Result<Fluid, DriverError> {
    const PTX_SRC: &str = include_str!("fluid.cu");

    let ptx = compile_ptx(PTX_SRC).unwrap();

    dev.load_ptx(
        ptx,
        "fluid",
        &[
            "divergence",
            "pressure",
            "incompress",
            "advect_velocity",
            "advect_smoke",
            "constant",
        ],
    )?;

    // Fluid::new(dev.clone(), x_size, y_size, z_size)

    let bin = include_bytes!("D:/code/fluid-simulator/scene.glb");
    let (doc, buffers, _) = gltf::import_slice(bin).unwrap();
    Fluid::from_gltf(dev.clone(), x_size, y_size, z_size, doc, buffers)
}
