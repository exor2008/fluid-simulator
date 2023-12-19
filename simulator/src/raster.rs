use std::cmp::{max, min};

struct Vec3 {
    x: i32,
    y: i32,
    z: i32,
}

impl From<[f32; 3]> for Vec3 {
    fn from(value: [f32; 3]) -> Self {
        Vec3 {
            x: value[0] as i32,
            y: value[2] as i32,
            z: value[1] as i32,
        }
    }
}

pub struct Triangle {
    vertices: [Vec3; 3],
}

impl Triangle {
    pub fn new(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> Self {
        Triangle {
            vertices: [Vec3::from(a), Vec3::from(b), Vec3::from(c)],
        }
    }

    fn barycentric_coords(p: &Vec3, a: &Vec3, b: &Vec3, c: &Vec3) -> (f32, f32, f32) {
        let v0 = [b.x - a.x, b.y - a.y, b.z - a.z];
        let v1 = [c.x - a.x, c.y - a.y, c.z - a.z];
        let v2 = [p.x - a.x, p.y - a.y, p.z - a.z];

        let dot00 = v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2];
        let dot01 = v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2];
        let dot11 = v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2];
        let dot20 = v2[0] * v0[0] + v2[1] * v0[1] + v2[2] * v0[2];
        let dot21 = v2[0] * v1[0] + v2[1] * v1[1] + v2[2] * v1[2];

        let inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01) as f32;

        let v = (dot11 * dot20 - dot01 * dot21) as f32 * inv_denom;
        let w = (dot00 * dot21 - dot01 * dot20) as f32 * inv_denom;
        let u = 1.0 - v - w;

        (u, v, w)
    }

    pub fn on_grid(&self) -> Vec<[usize; 3]> {
        let mut points: Vec<[usize; 3]> = Vec::new();

        let mut min_bound = Vec3 {
            x: i32::MAX,
            y: i32::MAX,
            z: i32::MAX,
        };
        let mut max_bound = Vec3 {
            x: i32::MIN,
            y: i32::MIN,
            z: i32::MIN,
        };

        // Find bounding box of the triangle
        for vertex in self.vertices.iter() {
            min_bound.x = min(min_bound.x, vertex.x);
            min_bound.y = min(min_bound.y, vertex.y);
            min_bound.z = min(min_bound.z, vertex.z);
            max_bound.x = max(max_bound.x, vertex.x);
            max_bound.y = max(max_bound.y, vertex.y);
            max_bound.z = max(max_bound.z, vertex.z);
        }

        // Rasterize the triangle within the bounding box
        for z in min_bound.z..=max_bound.z {
            for y in min_bound.y..=max_bound.y {
                for x in min_bound.x..=max_bound.x {
                    let point = Vec3 { x, y, z };
                    let (u, v, w) = Triangle::barycentric_coords(
                        &point,
                        &self.vertices[0],
                        &self.vertices[1],
                        &self.vertices[2],
                    );

                    if u >= 0.0 && v >= 0.0 && w >= 0.0 {
                        points.push([x as usize, y as usize, z as usize])
                    }
                }
            }
        }

        points
    }
}
