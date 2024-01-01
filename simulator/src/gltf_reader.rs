use crate::raster::Triangle;
use crate::Fluid;
use cudarc::driver::CudaDevice;
use cudarc::driver::DriverError;
use gltf::mesh::Reader;
use gltf::Buffer;
use gltf::{self, buffer::Data, Document};
use nalgebra::Quaternion;
use nalgebra::UnitQuaternion;
use nalgebra::Vector3;
use std::sync::Arc;

const CELLS_PER_UNIT: f32 = 10.0;

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
        let mut u_init_host = vec![0f32; size];
        let mut v_init_host = vec![0f32; size];
        let mut w_init_host = vec![0f32; size];
        let new_u_host = vec![0f32; size];
        let new_v_host = vec![0f32; size];
        let new_w_host = vec![0f32; size];
        let smoke_host = vec![0f32; size];
        let mut smoke_init_host = vec![0f32; size];
        let new_smoke_host = vec![0f32; size];
        let div_host = vec![0f32; size];
        let pressure_a_host = vec![0f32; size];
        let pressure_b_host = vec![0f32; size];
        let out_host = vec![0f32; size];
        let mut normal_u_host = vec![0f32; size];
        let mut normal_v_host = vec![0f32; size];
        let mut normal_w_host = vec![0f32; size];
        let mut block_host = vec![false; size];

        for node in doc.nodes() {
            if let Some(mesh) = node.mesh() {
                for primitive in mesh.primitives() {
                    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
                    let (inner, outer, normals) =
                        Fluid::gltf_to_grid::<'_, '_, Buffer<'_>>(reader, x_size, y_size, z_size);

                    if let Some(name) = node.name() {
                        match name {
                            // Speed
                            name if name.starts_with("Speed") => {
                                let number =
                                    name.strip_prefix("Speed").expect("Incorrect speed naming");
                                let angle_node = doc
                                    .nodes()
                                    .find(|node| node.name().unwrap() == format!("Angle{}", number))
                                    .expect("Missing angle for Speed");
                                let (_origin, rotation, scale) =
                                    angle_node.transform().decomposed();
                                let quaternion = Quaternion::from(rotation);
                                let quaternion = UnitQuaternion::from_quaternion(quaternion);
                                let v = Vector3::new(0.0, 0.0, scale[2]);
                                let rotated = quaternion.transform_vector(&v);

                                for idx in inner {
                                    u_init_host[idx] = rotated.x;
                                    v_init_host[idx] = rotated.y;
                                    w_init_host[idx] = rotated.z;
                                }
                            }

                            // Smoke
                            name if name.starts_with("Smoke") => {
                                let number =
                                    name.strip_prefix("Smoke").expect("Incorrect smoke naming");
                                let angle_node = doc
                                    .nodes()
                                    .find(|node| node.name().unwrap() == format!("Angle{}", number))
                                    .expect("Missing angle for Smoke");
                                let (_origin, _rotation, scale) =
                                    angle_node.transform().decomposed();

                                for idx in inner {
                                    smoke_init_host[idx] = scale[0];
                                }
                            }

                            // Regular geometry
                            _name => {
                                for (idx, norm) in outer.iter().zip(normals) {
                                    normal_u_host[*idx] = norm[0];
                                    normal_v_host[*idx] = norm[2];
                                    normal_w_host[*idx] = norm[1];
                                }
                                for idx in inner {
                                    block_host[idx] = true;
                                }
                            }
                        }
                    }
                }
            }
        }

        let u_dev = dev.htod_copy(u_host)?;
        let v_dev = dev.htod_copy(v_host)?;
        let w_dev = dev.htod_copy(w_host)?;
        let u_init_dev = dev.htod_copy(u_init_host)?;
        let v_init_dev = dev.htod_copy(v_init_host)?;
        let w_init_dev = dev.htod_copy(w_init_host)?;
        let new_u_dev = dev.htod_copy(new_u_host)?;
        let new_v_dev = dev.htod_copy(new_v_host)?;
        let new_w_dev = dev.htod_copy(new_w_host)?;
        let smoke_dev = dev.htod_copy(smoke_host)?;
        let new_smoke_dev = dev.htod_copy(new_smoke_host)?;
        let smoke_init_dev = dev.htod_copy(smoke_init_host)?;
        let div_dev = dev.htod_copy(div_host)?;
        let pressure_a_dev = dev.htod_copy(pressure_a_host)?;
        let pressure_b_dev = dev.htod_copy(pressure_b_host)?;
        let normal_u_dev = dev.htod_copy(normal_u_host)?;
        let normal_v_dev = dev.htod_copy(normal_v_host)?;
        let normal_w_dev = dev.htod_copy(normal_w_host)?;
        let out_dev = dev.htod_copy(out_host)?;
        let block_dev = dev.htod_copy(block_host)?;

        let fluid = Fluid {
            u_dev,
            v_dev,
            w_dev,
            u_init_dev,
            v_init_dev,
            w_init_dev,
            new_u_dev,
            new_v_dev,
            new_w_dev,
            smoke_dev,
            new_smoke_dev,
            smoke_init_dev,
            div_dev,
            pressure_a_dev,
            pressure_b_dev,
            normal_u_dev,
            normal_v_dev,
            normal_w_dev,
            block_dev,
            out_dev,
            x_size,
            y_size,
            z_size,
        };

        Ok(fluid)
    }

    fn gltf_to_grid<'a, 's, F>(
        reader: Reader<'a, 's, impl Fn(Buffer<'a>) -> Option<&'s [u8]> + Clone>,
        x_size: usize,
        y_size: usize,
        z_size: usize,
    ) -> (Vec<usize>, Vec<usize>, Vec<[f32; 3]>) {
        let positions: Vec<[f32; 3]> = reader.read_positions().unwrap().collect();
        let normals: Vec<[f32; 3]> = reader.read_normals().unwrap().collect();
        let mut outer: Vec<usize> = vec![];
        let mut inner: Vec<usize> = vec![];
        let mut normals_out: Vec<[f32; 3]> = vec![];

        // Iterate over triangles
        for [ind1, ind2, ind3] in reader.read_indices().unwrap().into_u32().array_chunks() {
            let ind1 = ind1 as usize;
            let ind2 = ind2 as usize;
            let ind3 = ind3 as usize;

            let pos1 = Fluid::to_grid_coords(positions[ind1]);
            let pos2 = Fluid::to_grid_coords(positions[ind2]);
            let pos3 = Fluid::to_grid_coords(positions[ind3]);

            if !Fluid::in_bounds(pos1, x_size, y_size, z_size)
                || !Fluid::in_bounds(pos2, x_size, y_size, z_size)
                || !Fluid::in_bounds(pos3, x_size, y_size, z_size)
            {
                continue;
            }

            let norm = normals[ind3];
            let norm = normalize(norm);

            let triangle = Triangle::new(pos1, pos2, pos3);
            let points = triangle.on_grid();

            for [x, y, z] in points {
                let idx = (y + y_size * z) * x_size + x;
                normals_out.push(norm);
                outer.push(idx);

                let x_shift = (norm[0].signum() * norm[0].abs().ceil()) as usize;
                let y_shift = (norm[2].signum() * norm[2].abs().ceil()) as usize;
                let z_shift = (norm[1].signum() * norm[1].abs().ceil()) as usize;

                let x_block = x + x_shift;
                let y_block = y + y_shift;
                let z_block = z + z_shift;

                let idx_block = (y_block + y_size * z_block) * x_size + x_block;
                inner.push(idx_block);
            }
        }
        (inner, outer, normals_out)
    }

    fn in_bounds(coord: [i32; 3], x_size: usize, y_size: usize, z_size: usize) -> bool {
        let [x, y, z] = coord;
        if x < 0 || y < 0 || z < 0 {
            false
        } else {
            x < x_size as i32 - 1 && y < y_size as i32 - 1 && z < z_size as i32 - 1
        }
    }

    fn to_grid_coords(mut pos: [f32; 3]) -> [i32; 3] {
        pos[0] *= CELLS_PER_UNIT;
        pos[1] *= CELLS_PER_UNIT;
        pos[2] *= CELLS_PER_UNIT;

        [pos[0] as i32, pos[1] as i32, pos[2] as i32]
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
