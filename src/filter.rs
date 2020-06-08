
use nalgebra::{VectorN, DimName, MatrixMN, MatrixN, RealField, Dim, allocator::Allocator, DefaultAllocator};


pub struct Filter<T, DimZ, DimX>
where
    T: RealField,
    DimZ: Dim + DimName,
    DimX: Dim + DimName,
    DefaultAllocator: Allocator<T, DimX, DimX>
    + Allocator<T, DimX>
    + Allocator<T, DimZ, DimZ>
    + Allocator<T, DimZ, DimX>
    + Allocator<T, DimX, DimZ>
    + Allocator<T, DimZ>
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
    q: MatrixN<T, DimX>
}

impl <T, DimZ, DimX> Filter<T, DimZ, DimX>
where
    T: RealField,
    DimZ: Dim + DimName,
    DimX: Dim + DimName,
    DefaultAllocator: Allocator<T, DimX, DimX>
        + Allocator<T, DimX>
        + Allocator<T, DimZ, DimZ>
        + Allocator<T, DimZ, DimX>
        + Allocator<T, DimX, DimZ>
        + Allocator<T, DimZ>
{
    pub fn new(x: VectorN<T, DimX>, p: MatrixN<T, DimX>, f: MatrixN<T, DimX>, r: MatrixN<T, DimZ>, h: MatrixMN<T, DimZ, DimX>, q: MatrixN<T, DimX>) ->  Filter<T, DimZ, DimX>{
        Filter{x, p, f, r, h, q}
    }

    pub fn run(&mut self, z: VectorN<T, DimZ>) -> (VectorN<T, DimX>, MatrixN<T, DimX>){
        //predict
        let x = &self.f * &self.x;
        let p = &self.f * &self.p * &self.f.transpose() + &self.q;
        //update
        let s = &self.h * &p * &self.h.transpose() + &self.r;
        let k = &p * &self.h.transpose() * s.try_inverse().unwrap();
        let y = z - &self.h * &x;
        self.x = x + &k * y;
        self.p = &p - k * &self.h * &p;
        (self.x.clone(), self.p.clone())
    }
}
pub struct Filter2<T, DimZ, DimX, DimU>
where
    T: RealField,
    DimZ: Dim + DimName,
    DimX: Dim + DimName,
    DimU: Dim + DimName,
    DefaultAllocator: Allocator<T, DimX, DimX>
    + Allocator<T, DimX, DimU>
    + Allocator<T, DimX>
    + Allocator<T, DimZ, DimZ>
    + Allocator<T, DimZ, DimX>
    + Allocator<T, DimX, DimZ>
    + Allocator<T, DimX, DimU>
    + Allocator<T, DimZ>
{
    ///state estimate vector
    x: VectorN<T, DimX>,
    ///state covariance matrix
    p: MatrixN<T, DimX>,
    ///system model
    f: MatrixN<T, DimX>,
    ///control function
    b: MatrixMN<T, DimX, DimU>,
    ///measurement noise covariance matrix
    r: MatrixN<T, DimZ>,
    ///measurement function
    h: MatrixMN<T, DimZ, DimX>,
    ///process noise covariance matrix
    q: MatrixN<T, DimX>
}

impl <T, DimZ, DimX, DimU> Filter2<T, DimZ, DimX, DimU>
where
    T: RealField,
    DimZ: Dim + DimName,
    DimX: Dim + DimName,
    DimU: Dim + DimName,
    DefaultAllocator: Allocator<T, DimX, DimX>
        + Allocator<T, DimX, DimU>
        + Allocator<T, DimX>
        + Allocator<T, DimZ, DimZ>
        + Allocator<T, DimZ, DimX>
        + Allocator<T, DimX, DimZ>
        + Allocator<T, DimX, DimU>
        + Allocator<T, DimU>
        + Allocator<T, DimZ>
{
    pub fn new(x: VectorN<T, DimX>, p: MatrixN<T, DimX>, f: MatrixN<T, DimX>, b: MatrixMN<T, DimX, DimU>, r: MatrixN<T, DimZ>, h: MatrixMN<T, DimZ, DimX>, q: MatrixN<T, DimX>) ->  Filter2<T, DimZ, DimX, DimU>{
        Filter2{x, p, f, b, r, h, q}
    }

    pub fn run(&mut self, z: VectorN<T, DimZ>, u: VectorN<T, DimU>) -> (VectorN<T, DimX>, MatrixN<T, DimX>){
        //predict
        let x = &self.f * &self.x + &self.b * u;
        let p = &self.f * &self.p * &self.f.transpose() + &self.q;
        //update
        let s = &self.h * &p * &self.h.transpose() + &self.r;
        let k = &p * &self.h.transpose() * s.try_inverse().unwrap();
        let y = z - &self.h * &x;
        self.x = x + &k * y;
        self.p = &p - k * &self.h * &p;
        (self.x.clone(), self.p.clone())
    }
}