use fehler::{throw, throws};
use nalgebra::{
    allocator::Allocator, DefaultAllocator, Dim, DimName, MatrixMN, MatrixN, RealField, VectorN,
};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum KfError {
    #[error("Matrix inversion failed")]
    InverseError,
}

pub struct Filter<T, DimZ, DimX>
where
    T: RealField,
    DimZ: Dim + DimName,
    DimX: Dim + DimName,
    DefaultAllocator: Allocator<T, DimX>
        + Allocator<T, DimX, DimX>
        + Allocator<T, DimZ, DimX>
        + Allocator<T, DimZ, DimZ>,
{
    ///state estimate vector
    x: VectorN<T, DimX>,
    ///state covariance matrix
    p: MatrixN<T, DimX>,
    ///system model
    f: MatrixN<T, DimX>,
    ///measurement noise covariance matrix
    r: MatrixN<T, DimZ>,
    ///measurement function
    h: MatrixMN<T, DimZ, DimX>,
    ///process noise covariance matrix
    q: MatrixN<T, DimX>,
}

impl<T, DimZ, DimX> Filter<T, DimZ, DimX>
where
    T: RealField,
    DimZ: Dim + DimName,
    DimX: Dim + DimName,
    DefaultAllocator: Allocator<T, DimX>
        + Allocator<T, DimZ>
        + Allocator<T, DimX, DimX>
        + Allocator<T, DimX, DimZ>
        + Allocator<T, DimZ, DimX>
        + Allocator<T, DimZ, DimZ>,
{
    pub fn new(
        x: VectorN<T, DimX>,
        p: MatrixN<T, DimX>,
        f: MatrixN<T, DimX>,
        r: MatrixN<T, DimZ>,
        h: MatrixMN<T, DimZ, DimX>,
        q: MatrixN<T, DimX>,
    ) -> Filter<T, DimZ, DimX> {
        Filter { x, p, f, r, h, q }
    }

    #[throws(KfError)]
    pub fn run(&mut self, z: VectorN<T, DimZ>) -> (VectorN<T, DimX>, MatrixN<T, DimX>, VectorN<T, DimZ>, MatrixN<T, DimZ>, MatrixN<T, DimZ>) {
        //predict
        let x = &self.f * &self.x;
        let p = &self.f * &self.p * &self.f.transpose() + &self.q;
        //update
        let s = &self.h * &p * &self.h.transpose() + &self.r;
        let s_inverse = match s.clone().try_inverse() {
            Some(m) => m,
            None => throw!(KfError::InverseError),
        };
        let k = &p * &self.h.transpose() * &s_inverse;
        let y = z - &self.h * &x;
        self.x = x + &k * &y;
        self.p = &p - k * &self.h * &p;
        (self.x.clone(), self.p.clone(), y, s, s_inverse)
    }

    pub fn set_p(&mut self, p: MatrixN<T, DimX>){
        self.p = p;
    }

    pub fn set_x(&mut self, x: VectorN<T, DimX>){
        self.x = x;
    }
}