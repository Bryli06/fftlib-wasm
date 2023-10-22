use core::arch::wasm32::*;
use wasm_bindgen_test::*;
use fftlib::{complex::Complex, fft};

#[wasm_bindgen_test]
fn but4() {
    let v1 = vec![7.0, 1.0, 1.0];
    let v2 = vec![2.0, 4.0];
    assert_eq!(fft::pmul(&v1, &v2), [14.0, 30.0, 6.0, 4.0]);
}
/*
#[wasm_bindgen_test]
fn but2() {
    let v1 = vec![7.0, 1.0];
    let v2 = vec![2.0];
    assert_eq!(fft::pmul(&v1, &v2), [14.0, 30.0, 6.0, 4.0]);
}

#[wasm_bindgen_test]
fn but3() {
    let v1 = vec![7.0, 1.0];
    let v2 = vec![2.0, 4.0];
    assert_eq!(fft::pmul(&v1, &v2), [14.0, 30.0, 6.0, 4.0]);
}
*/
#[wasm_bindgen_test]
fn but5() {
    let v1 = vec![7.0, 1.0, 1.0];
    let v2 = vec![2.0, 4.0, 3.0];
    assert_eq!(fft::pmul(&v1, &v2), [14.000000000000004, 30.0, 27.0, 7.0, 3.0]);
}

#[wasm_bindgen_test]
fn large() {
    let v = vec![1.0; 100];
    fft::pmul(&v, &v);
    println!("{:?}", v);
    assert_eq!(0.0, 1.0);
}

#[wasm_bindgen_test]
fn multiply () {
    let a = mult(Complex::new(1.0, 2.0), Complex::new(-0.5, 3.0));
    assert!(a == Complex::new(-6.5, 2.0));
}

#[wasm_bindgen_test]
fn full(){
    let v: Vec<f64> = (0..10).map(|x| x as f64).collect();
    assert_eq!(fft::pmul(&v, &v), [0.0]);
}


fn mult(a: Complex, b: Complex) -> Complex {
    let mut temp: Complex = Default::default();
    unsafe {
        let x = v128_load((&a) as *const Complex as *const v128);
        let mut yr: v128 = f64x2_splat(b.real);
        let mut yi: v128 = f64x2_splat(b.imaginary);

        yr = f64x2_mul(x, yr);
        let mut n1: v128 = i64x2_shuffle::<1,2>(x, x);
        yi = f64x2_mul(n1, yi);
        n1 = f64x2_sub(yr, yi);
        yr = f64x2_add(yr, yi);
        n1 = i64x2_shuffle::<0,3>(n1, yr);
        v128_store(&mut temp as *mut Complex as *mut v128, n1);
    }
    temp
}
