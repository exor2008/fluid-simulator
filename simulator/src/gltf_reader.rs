use crate::raster::Triangle;
use crate::Fluid;
use cudarc::driver::CudaDevice;
use cudarc::driver::DriverError;
use gltf::{self, buffer::Data, Document};
use std::sync::Arc;

impl Fluid {
    pub fn from_gltf(
        dev: Arc<CudaDevice>,
        x_size: usize,
        y_size: usize,
        z_size: usize,
        doc: Document,
        buffers: Vec<Data>,
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
        let mut normal_u_host = vec![0f32; size];
        let mut normal_v_host = vec![0f32; size];
        let mut normal_w_host = vec![0f32; size];
        let mut block_host = vec![false; size];

        // normal_u_host[(0 + 10 * 80) * 140 + 100] = -1.0;

        for node in doc.nodes() {
            // for mesh in doc.meshes() {
            let (origin, _rotation, _scale) = node.transform().decomposed();

            if let Some(mesh) = node.mesh() {
                for primitive in mesh.primitives() {
                    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
                    let positions: Vec<[f32; 3]> = reader.read_positions().unwrap().collect();
                    let normals: Vec<[f32; 3]> = reader.read_normals().unwrap().collect();

                    // Iterate over triangles
                    for [ind1, ind2, ind3] in
                        reader.read_indices().unwrap().into_u32().array_chunks()
                    {
                        let ind1 = ind1 as usize;
                        let ind2 = ind2 as usize;
                        let ind3 = ind3 as usize;

                        let pos1 = Fluid::translate(positions[ind1], origin);
                        let pos2 = Fluid::translate(positions[ind2], origin);
                        let pos3 = Fluid::translate(positions[ind3], origin);

                        let norm = normals[ind1];
                        let norm = normalize(norm);

                        let triangle = Triangle::new(pos1, pos2, pos3);
                        let points = triangle.on_grid();

                        for [x, y, z] in points {
                            let idx = (y + y_size * z) * x_size + x;
                            normal_u_host[idx] = norm[0];
                            normal_v_host[idx] = norm[2];
                            normal_w_host[idx] = norm[1];
                        }
                    }
                }
            }
        }

        let mut block;

        for z in 0..z_size {
            for y in 0..y_size {
                block = false;
                for x in 0..x_size {
                    let idx = (y + y_size * z) * x_size + x;
                    let normal = normal_u_host[idx] + normal_v_host[idx] + normal_w_host[idx];
                    match normal == 0.0 {
                        true => block_host[idx] = block,
                        false => block = !block,
                    }
                }
            }
        }

        for z in 0..z_size {
            for x in 0..x_size {
                block = false;
                for y in 0..y_size {
                    let idx = (y + y_size * z) * x_size + x;
                    let normal = normal_u_host[idx] + normal_v_host[idx] + normal_w_host[idx];
                    match normal == 0.0 {
                        true => block_host[idx] = block,
                        false => block = !block,
                    }
                }
            }
        }

        for x in 0..x_size {
            for y in 0..y_size {
                block = false;
                for z in 0..z_size {
                    let idx = (y + y_size * z) * x_size + x;
                    let normal = normal_u_host[idx] + normal_v_host[idx] + normal_w_host[idx];
                    match normal == 0.0 {
                        true => block_host[idx] = block,
                        false => block = !block,
                    }
                }
            }
        }

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
        let block_dev = dev.htod_copy(block_host)?;

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
            block_dev,
            x_size,
            y_size,
            z_size,
        };

        Ok(fluid)
    }

    fn translate(mut pos: [f32; 3], origin: [f32; 3]) -> [f32; 3] {
        pos[0] += origin[0];
        pos[1] += origin[1];
        pos[2] += origin[2];

        pos[0] *= 10.0;
        pos[1] *= 10.0;
        pos[2] *= 10.0;

        pos
    }
}

fn normalize(vector: [f32; 3]) -> [f32; 3] {
    let x: f32 = vector[0];
    let y: f32 = vector[1];
    let z: f32 = vector[2];

    let mag = (x * x + y * y + z * z).sqrt();

    if mag == 0.0 {
        return [0.0; 3];
    } else {
        return [x / mag, y / mag, z / mag];
    }
}
