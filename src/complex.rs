use std::{ops::{Add, Mul, Neg, Sub}, f64::consts::PI};

#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Complex {
    pub real: f64,
    pub imaginary: f64,
}

impl Complex {
    pub fn new(real: f64, imaginary: f64) -> Self {
        Self {
            real,
            imaginary
        }
    }

    pub fn from_polar(r: f64, theta: f64) -> Self {
        let c: f64 = theta.cos();
        Self::new(r * c, r * (1.0 - c * c).sqrt() *
            {if theta > 0.0 {
                if theta < PI {
                    1.0
                } else {
                    -1.0
                }
             } else {
                if theta > -PI {
                    -1.0
                } else {
                    1.0
                }
             }})
    }

    pub fn conjugate(&mut self) {
        self.imaginary = -self.imaginary;
    }
}

impl From<f64> for Complex {
    fn from(x: f64) -> Self {
        Self::new(x, 0.0)
    }
}

impl Neg for Complex {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.real, -self.imaginary)
    }
}

impl Add for Complex {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self::new(self.real + other.real, self.imaginary + other.imaginary)
    }
}

impl Sub for Complex {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self::new(self.real - other.real, self.imaginary - other.imaginary)
    }
}

impl Mul for Complex {

    type Output = Self;
    fn mul(self, other: Self) -> Self {
        let real = self.real * other.real - self.imaginary * other.imaginary;
        let imaginary = self.imaginary * other.real + self.real * other.imaginary;
        Self::new(real, imaginary)
    }
}

impl Mul<Complex> for f64 {
    type Output = Complex;
    fn mul(self, rhs: Complex) -> Complex {
        Complex::new(rhs.real * self, rhs.imaginary * self)
    }
}
