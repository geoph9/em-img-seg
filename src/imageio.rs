extern crate image;
use image::{GenericImageView, Pixel};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use self::image::DynamicImage;

#[pyclass]
pub struct MyImage {
    img: DynamicImage,
    img2d: Vec<Vec<u8>>,
}

#[pymethods]
impl MyImage {
    #[new]
    fn new(path: String) -> Self {
        let img = image::open(path).unwrap();
        MyImage { img, img2d: image_to_data()}
    }

    fn image_to_data(&mut self, _py: Python<'_>) -> Vec<Vec<u8>> {
        // let n = (img.dimensions().0 * img.dimensions().1) as usize;
        // let rgb1 = img.to_rgb8().to_vec()[0..n].to_vec();
        // let rgb2 = img.to_rgb8().to_vec()[n..(2*n)].to_vec();
        // let rgb3 = img.to_rgb8().to_vec()[(2*n)..].to_vec();
        // vec![rgb1, rgb2, rgb3]

        let (height, width) = self.img.dimensions();
        let n = (height*width) as usize;
        let x = vec![vec![0_u8; 3]; n];  // NxD
        for w in 0..width {
            for h in 0..height {
                let tmp = h + (w-1)*height;
                let tmp2 = self.img.get_pixel(h, w).to_rgb();
                x[tmp][0] = tmp2[0];
                x[tmp][1] = tmp2[1];
                x[tmp][2] = tmp2[2];
            }
        }
        x
    }

    // Converts a split image to a normal 3d image (3rd dim is the color).
    fn data_to_image(&mut self, _py: Python<'_>, out_path: String) {
        let (height, width) = self.img.dimensions();
        let mut img_buf = image::ImageBuffer::new(width, height);
        for w in 0..width {
            for h in 0..height {
                let tmp = h + (w-1)*height;
                let pixel = img_buf.get_pixel_mut(w, h);
                // let image = *pixel;
                *pixel = image::Rgb([self.img2d[tmp][0], self.img2d[tmp][1], self.img2d[tmp][2]]);
            }
        }
        img_buf.save(out_path).unwrap();
    }
}




