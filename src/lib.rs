use std::arch::x86_64::{__m128d, _mm_set_pd1, _mm_mul_pd, _mm_load_pd, _mm_shuffle_pd, _mm_sub_pd, _mm_add_pd, _mm_store_pd};

use complex::Complex;

mod complex;
pub mod fft;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn but4() {
        let v1 = vec![7.0, 1.0, 1.0];
        let v2 = vec![2.0, 4.0];
        assert_eq!(fft::pmul(&v1, &v2), [14.0, 30.0, 6.0, 4.0]);
    }

    #[test]
    fn but5() {
        let v1 = vec![7.0, 1.0, 1.0];
        let v2 = vec![2.0, 4.0, 3.0];
        assert_eq!(fft::pmul(&v1, &v2), [14.000000000000004, 30.0, 27.0, 7.0, 3.0]);
    }

    #[test]
    fn large() {
        let v = vec![1.0; 100];
        fft::pmul(&v, &v);
        assert_eq!(0.0, 1.0);
    }

    #[test]
    fn multiply () {
        let a = mult(Complex::new(1.0, 2.0), Complex::new(-0.5, 3.0));
        assert!(a == Complex::new(-6.5, 2.0));
    }

}

fn mult(a: Complex, b: Complex) -> Complex {
    let mut temp: Complex = Default::default();
    unsafe {
        let x = _mm_load_pd((&a) as *const Complex as *const f64);
        let mut yr: __m128d = _mm_set_pd1(b.real);
        let mut yi: __m128d = _mm_set_pd1(b.imaginary);

        yr = _mm_mul_pd(x, yr);
        let mut n1: __m128d = _mm_shuffle_pd::<1>(x, x);
        println!("{:?}", n1);
        yi = _mm_mul_pd(n1, yi);
        n1 = _mm_sub_pd(yr, yi);
        yr = _mm_add_pd(yr, yi);
        n1 = _mm_shuffle_pd::<2>(n1, yr);
        _mm_store_pd(&mut temp as *mut Complex as *mut f64, n1);
    }
    temp
}
