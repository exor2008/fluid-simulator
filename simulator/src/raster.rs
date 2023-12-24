use nalgebra::Vector3;
use std::cmp::{max, min};

pub struct Triangle {
    vertices: [Vector3<i32>; 3],
}

impl Triangle {
    pub fn new(a: [i32; 3], b: [i32; 3], c: [i32; 3]) -> Self {
        Triangle {
            vertices: [Vector3::from(a), Vector3::from(b), Vector3::from(c)],
        }
    }

    pub fn on_grid(&self) -> Vec<[usize; 3]> {
        let mut points: Vec<[usize; 3]> = Vec::new();

        let mut min_bound: Vector3<i32> = Vector3::new(i32::MAX, i32::MAX, i32::MAX);
        let mut max_bound: Vector3<i32> = Vector3::new(i32::MIN, i32::MIN, i32::MIN);

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
                    let point: Vector3<i32> = Vector3::new(x, y, z);
                    if in_triangle(self.vertices[0], self.vertices[1], self.vertices[2], point) {
                        points.push([x as usize, y as usize, z as usize])
                    }
                }
            }
        }

        points
    }
}

fn same_side(p1: Vector3<i32>, p2: Vector3<i32>, a: Vector3<i32>, b: Vector3<i32>) -> bool {
    let diff = b - a;

    let cp1 = diff.cross(&(p1 - a));
    let cp2 = diff.cross(&(p2 - a));

    cp1.dot(&cp2) >= 0
}

fn in_triangle(a: Vector3<i32>, b: Vector3<i32>, c: Vector3<i32>, p: Vector3<i32>) -> bool {
    if same_side(p, a, b, c) && same_side(p, b, a, c) && same_side(p, c, a, b) {
        let vcl = (a - b).cross(&(a - c));
        (a - p).dot(&vcl).abs() <= 300
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_in_triangle() {
        // Vector3::cr
        let p1 = Vector3::new(5, 5, 0);
        let p2 = Vector3::new(0, 0, 0);
        let p3 = Vector3::new(0, 1, 0);
        let p4 = Vector3::new(15, 0, 0);
        let p5 = Vector3::new(15, 0, 10);
        let p6 = Vector3::new(5, 0, 10);

        let a = Vector3::new(0, 0, 0);
        let b = Vector3::new(10, 0, 0);
        let c = Vector3::new(0, 10, 0);

        assert!(in_triangle(a, b, c, p1));
        assert!(in_triangle(a, b, c, p2));
        assert!(in_triangle(a, b, c, p3));
        assert!(!in_triangle(a, b, c, p4));
        assert!(!in_triangle(a, b, c, p5));
        assert!(!in_triangle(a, b, c, p6));
    }
}
