// extern crate image;
// // extern crate rulinalg;
// extern crate ndarray;
// extern crate ndarray_image;
//
// // use image::{GenericImageView, Pixel};
// use ndarray::{Array2, Dim};
// use ndarray_image::{open_image, Colors};
// // use rulinalg::matrix::Matrix;
//
// // use pyo3::exceptions::PyRuntimeError;
// use pyo3::prelude::*;
//
// #[pyclass]
// pub struct MyImage {
//     // img: DynamicImage,
//     img: ndarray::Array3<u8>,
//     img_arr: Array2<f32> ,
// }
//
// #[pymethods]
// impl MyImage {
//     #[new]
//     fn new(path: String) -> Self {
//         // let img = image::open(path).unwrap();
//         // let img = open_image(path, Colors::Rgb).expect("unable to open input image");
//         let (img, img_arr) = image_to_data(path);
//         // Normalized on [0, 1]
//         MyImage {
//             img,
//             img_arr,
//         }
//     }
//
//     // Converts a split image to a normal 3d image (3rd dim is the color).
//     fn data_to_image(&mut self, _py: Python<'_>, out_path: String) {
//         let sh = self.img.shape();
//         let (height, width) = (sh[0] as u32, sh[1] as u32);
//         let mut img_buf = image::ImageBuffer::new(width, height);
//         for i in 0..self.img_arr.nrows() {
//             let color0 = (self.img_arr[[i, 0]] * 255.0) as u8;
//             let color1 = (self.img_arr[[i, 1]] * 255.0) as u8;
//             let color2 = (self.img_arr[[i, 2]] * 255.0) as u8;
//             let x = i as u32 % width;
//             let y = i as u32 / width;
//             let pixel = img_buf.get_pixel_mut(x, y);
//             *pixel = image::Rgb([color0, color1, color2]);
//         }
//         img_buf.save(out_path).unwrap();
//     }
// }
//
// // Converts a 3d image (with rgb colors) to a 2d image of dimensions Nx3 (3 is the number of colors)
// // Also normalizes the pixels by dividing with 255
// fn image_to_data(path: String) -> (ndarray::Array3<u8>, Array2<f32>) {
//     let img = open_image(path, Colors::Rgb).expect("unable to open input image");
//     let sh = img.shape();
//     let (height, width) = (sh[0] as u32, sh[1] as u32);
//     let n = (width * height) as usize;
//     let new_dim = Dim([n, 3]);
//     let img_vec = img.map(|&e| e as f32 / 255_f32).into_raw_vec();
//     let img_arr = Array2::from_shape_vec(new_dim, img_vec).ok().unwrap();
//     (img, img_arr)
// }
//
//
// #[pymodule]
// pub fn myimage(_py: Python, m: &PyModule) -> PyResult<()> {
//     m.add_class::<MyImage>()?;
//     Ok(())
// }
