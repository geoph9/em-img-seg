extern crate ndarray;
extern crate ndarray_image;

// use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use ndarray::{Array, Axis, Array2, Array1, Dim, s, Zip};
use ndarray_rand::{RandomExt, SamplingStrategy};
use ndarray_image::{open_image, Colors};
use numpy::{
    PyArray2, PyArray1, PyArray3, ToPyArray
};

use std::f32::consts::{PI};
// use ndarray_rand::rand_distr::num_traits::real::Real;
use ndarray_stats::QuantileExt;

#[pymodule]
pub fn em(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<EM>()?;
    Ok(())
}

#[pyclass]
pub struct EM {
    img: ImageIO,
    k: u8,  // number of clusters
    gamma: Array2<f32>,  // dims = NxK
    pi: Array1<f32>,  // dims = Kx1
    mu: Array2<f32>,  // dims = Kx3 (randomly sampled K rows from img_data
    sigma: Array1<f32>,  // dims = Kx1  (diagonal KxK cov matrix)
}

#[pymethods]
impl EM {
    #[new]
    fn new(path: String, k: u8) -> Self {
        let img = ImageIO::new(path);
        let (gamma, pi, mu, sigma) = _init(&img.img_arr, k);
        EM { img, k, gamma, pi, mu, sigma }
    }

    // ================================================================
    // ======================== EM FUNCTIONS ==========================
    // ================================================================

    fn _maximum_likelihood(&mut self, _py: Python) -> PyResult<f32> {// &'py PyArray2<f32> {
        let sh = self.img.img_arr.shape();
        let (n, d) = (sh[0], sh[1]);  // d is 3
        let mut likelihood: Array2<f32> = Array::zeros((n, self.k as usize));
        for k in 0..self.k {  // this loop can be moved out
            for col in 0..d {
                let inner_left = (2.0 * PI * self.sigma[k as usize]).sqrt().ln();
                let val: Array1<f32> = self.img.img_arr.index_axis(Axis(1), col).
                    mapv(|e| e-self.mu[[k as usize, col]]). // subtract mean
                    mapv(|e| e.powi(2)).  // to the power of 2
                    mapv(|e| e / (2_f32*self.sigma[k as usize])).  // divide by sigma
                    mapv(|e| e + inner_left);  // add the left part (single val)
                // let mut slice = likelihood.slice_mut(s![.., k as usize]);
                // slice += val;
                // println!(" K: {:?}, COL: {:?}", k, col);
                Zip::from(&mut likelihood.column_mut(k as usize))
                    .and(&val)
                    .apply(|e, &v| {
                        *e += v;
                    });
            }
            let tmp = self.pi[k as usize].ln();
            let mut slice = likelihood.slice_mut(s![.., k as usize]);
            slice.mapv_inplace(|e| tmp - e);
        }
        let axis_max: Array1<f32> = likelihood.map_axis(Axis(1), |e| *e.max().unwrap());
        let replicated_max: Array2<f32> = Array2::from_shape_fn((n, self.k as usize), |(i, _)| axis_max[i]);
        likelihood = likelihood - replicated_max;
        likelihood.mapv_inplace(|e| e.exp());
        let tmp: Array1<f32> = likelihood.sum_axis(Axis(1)).map(|e| e.ln()) + axis_max;
        Ok(tmp.sum())
        // likelihood.to_pyarray(_py)
    }

    fn _estep(&mut self, _py: Python) {
        // println!("ESTEP");
        let n = self.img.img_arr.shape()[0];
        // NxK array
        let mut s: Array2<f32> = Array::zeros((n, self.k as usize));
        for k in 0..self.k as usize {
            for col in 0..3 {
                let inner_left = (2.0 * PI * self.sigma[k]).sqrt().ln();
                let val: Array1<f32> = self.img.img_arr.index_axis(Axis(1), col).
                    mapv(|e| e-self.mu[[k, col]]). // subtract mean
                    mapv(|e| e.powi(2)).  // to the power of 2
                    mapv(|e| e / (2_f32*self.sigma[k])).  // divide by sigma
                    mapv(|e| e + inner_left);  // add the left part (single val)
                Zip::from(&mut s.column_mut(k))
                    .and(&val)
                    .apply(|e, &v| {
                        *e += v;
                    });
            }
            let tmp = self.pi[k].ln();
            let mut slice = s.slice_mut(s![.., k]);
            slice.mapv_inplace(|e| tmp - e);
        }
        s.mapv_inplace(|e| e.exp());
        let row_sum: Array1<f32> = s.sum_axis(Axis(1));
        self.gamma = Array2::from_shape_fn((n, self.k as usize), |(i, j)| s[[i, j]] / row_sum[i]);
    }

    fn _mstep(&mut self, _py: Python) {
        // println!("MSTEP");
        let n = self.img.img_arr.shape()[0];
        let sum_gamma: Array1<f32> = self.gamma.sum_axis(Axis(0));
        for k in 0..self.k as usize {
            let gamma_col: Array1<f32> = self.gamma.slice(s![.., k]).to_owned();
            for d in 0..3 {
                // Update mu
                self.mu[[k, d]] = gamma_col.t().dot(&self.img.img_arr.slice(s![.., d])) / sum_gamma[k];
            }
            // Update sigma
            let denom: f32 = 3.0 * gamma_col.sum();
            let numerator: Array2<f32> = (self.img.img_arr.clone() - self.mu.slice(s![k, ..])).mapv(|e| e.powi(2));
            self.sigma[k] = (gamma_col * numerator.sum_axis(Axis(1))).sum() / denom;
        }
        self.pi = sum_gamma.mapv(|e| e/n as f32);
    }

    // ================================================================
    // ======================== Python Called =========================
    // ================================================================

    fn fit(&mut self, _py: Python, epochs: u8) {
        let tol = 1e-6;
        for epoch in 0..epochs {
            self._estep(_py);
            self._mstep(_py);
            let likelihood = self._maximum_likelihood(_py);
            println!("New likelihood: {:?}", likelihood);
        }
    }

    // ================================================================
    // ========================== GETTERS =============================
    // ================================================================
    fn get_gamma<'py>(&mut self, _py: Python<'py>) -> &'py PyArray2<f32> {
        let tmp = ndarray::ArrayView2::from(&self.gamma);
        tmp.to_pyarray(_py)
    }

    fn get_pi<'py>(&mut self, _py: Python<'py>) -> &'py PyArray1<f32> {
        let tmp = ndarray::ArrayView1::from(&self.pi);
        tmp.to_pyarray(_py)
    }

    fn get_mu<'py>(&mut self, _py: Python<'py>) -> &'py PyArray2<f32> {
        let tmp = ndarray::ArrayView2::from(&self.mu);
        // tmp.into_pyarray(_py)
        tmp.to_pyarray(_py)
    }

    fn get_sigma<'py>(&mut self, _py: Python<'py>) -> &'py PyArray1<f32> {
        let tmp = ndarray::ArrayView1::from(&self.sigma);
        tmp.to_pyarray(_py)
    }

    fn save_new_image(&mut self, _py: Python, out_path: String) {
       &self.img.data_to_image(out_path);
    }

    fn get_2d_image<'py>(&mut self, _py: Python<'py>) -> &'py PyArray2<f32> {
        let tmp = ndarray::ArrayView2::from(&self.img.img_arr);
        tmp.to_pyarray(_py)
    }

    // ================================================================
    // ============================ OTHER =============================
    // ================================================================

    // Useful for reading the image in python without needing PIL
    fn read_image<'py>(&mut self, _py: Python<'py>, path: String) -> &'py PyArray3<u8> {
        let img = open_image(path, Colors::Rgb).expect("Unable to open input image");
        let tmp = ndarray::ArrayView3::from(&img);
        tmp.to_pyarray(_py)
    }
}

// fn _estep_helper(em: EM, k: usize) -> Array1<f32> {
//     let inner_left = (2.0 * PI * em.sigma[k]).sqrt().ln();
//     let val: Array1<f32> = em.img.img_arr.index_axis(Axis(1), col).
//         mapv(|e| e-em.mu[[k, col]]). // subtract mean
//         mapv(|e| e.powi(2)).  // to the power of 2
//         mapv(|e| e / (2_f32*em.sigma[k])).  // divide by sigma
//         mapv(|e| e + inner_left);  // add the left part (single val)
//     val
// }

// returns initialized gamma, pi, mu, sigma
fn _init(img2d: &Array2<f32>, k: u8) -> (Array2<f32>, Array1<f32>, Array2<f32>, Array1<f32>) {
    let n = img2d.len();
    let mut gamma: Array2<f32> = Array::zeros((n as usize, k as usize));
    let mut pi: Array1<f32> = Array::from_elem((k as usize,), 1_f32/k as f32);
    // Sample k rows from the image data without replacement (a row cannot be repeated)
    let mut mu = img2d.sample_axis(Axis(0), k as usize, SamplingStrategy::WithoutReplacement);
    // Sigma is a diagonal array which we will represent as a 1d array
    let img_flattened_var: f32 = (img2d.var_axis(Axis(1), 0.0)).var_axis(Axis(0), 0.0).into_scalar();
    let mut sigma: Array1<f32> = Array::from_elem((k as usize,), 10f32 * img_flattened_var);
    (gamma, pi, mu, sigma)
}

pub struct ImageIO {
    img: ndarray::Array3<u8>,
    img_arr: Array2<f32> ,
}

impl ImageIO {
    fn new(path: String) -> Self {
        // let img = image::open(path).unwrap();
        // let img = open_image(path, Colors::Rgb).expect("unable to open input image");
        let (img, img_arr) = Self::image_to_data(path);
        // Normalized on [0, 1]
        ImageIO {
            img,
            img_arr,
        }
    }

    // Converts a split image to a normal 3d image (3rd dim is the color).
    fn data_to_image(&mut self, out_path: String) {
        let sh = self.img.shape();
        let (height, width) = (sh[0] as u32, sh[1] as u32);
        let mut img_buf = image::ImageBuffer::new(width, height);
        for i in 0..self.img_arr.nrows() {
            let color0 = (self.img_arr[[i, 0]] * 255.0) as u8;
            let color1 = (self.img_arr[[i, 1]] * 255.0) as u8;
            let color2 = (self.img_arr[[i, 2]] * 255.0) as u8;
            let x = i as u32 % width;
            let y = i as u32 / width;
            let pixel = img_buf.get_pixel_mut(x, y);
            *pixel = image::Rgb([color0, color1, color2]);
        }
        img_buf.save(out_path).unwrap();
    }

    // Converts a 3d image (with rgb colors) to a 2d image of dimensions Nx3 (3 is the number of colors)
    // Also normalizes the pixels by dividing with 255
    fn image_to_data(path: String) -> (ndarray::Array3<u8>, Array2<f32>) {
        let img = open_image(path, Colors::Rgb).expect("unable to open input image");
        let sh = img.shape();
        let (height, width) = (sh[0] as u32, sh[1] as u32);
        let n = (width * height) as usize;
        let new_dim = Dim([n, 3]);
        let img_vec = img.map(|&e| e as f32 / 255_f32).into_raw_vec();
        let img_arr = Array2::from_shape_vec(new_dim, img_vec).ok().unwrap();
        (img, img_arr)
    }
}
