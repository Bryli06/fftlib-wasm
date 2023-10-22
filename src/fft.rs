use std::f64::consts::PI;
use core::arch::wasm32::*;

use crate::complex::Complex;

pub struct FFT {
    nfft: usize,
    inverse: bool,
    twiddlers: Vec::<Complex>,
    stage_radix: Vec::<usize>,
    stage_remainder: Vec::<usize>,
}

impl FFT {
    pub fn new(nfft: usize, inverse: bool) -> Self {
        let mut n = nfft;
        let mut p = 4;
        let mut stage_radix = Vec::new();
        let mut stage_remainder = Vec::new();
        loop {
            while n % p > 0 {
                match p {
                    4 => p = 2,
                    2 => p = 3,
                    _ => p += 2,
                }
                if p * p > n {
                    p = n
                }
            }
            n /= p;
            stage_radix.push(p);
            stage_remainder.push(n);
            if n <= 1 {
                break;
            }
        };
        Self {
            nfft,
            inverse,
            twiddlers: {
                let step = if inverse { -2.0 } else { 2.0 } * PI / nfft as f64;
                (0..nfft).map(|i| Complex::from_polar(1.0, step * i as f64))
                    .collect()
            },
            stage_radix,
            stage_remainder,
        }
    }

    pub fn invert(&mut self) {
        self.inverse = !self.inverse;

        for i in &mut self.twiddlers {
            i.conjugate();
        }
    }

    pub fn fft(&self, v: &[f64]) -> Vec<Complex> {
        let complex_v = v.iter()
                         .cloned()
                         .map(Complex::from)
                         .collect::<Vec<_>>();
        let mut out = vec![Default::default(); self.nfft];
        self.transform(&complex_v, &mut out, 0, 1, 1);
        out
    }

    pub fn ifft(&self, v: &[Complex]) -> Vec<Complex> {
        let mut out = vec![Default::default(); self.nfft];
        self.transform(v, &mut out, 0, 1, 1);
        out
    }

    pub fn transform(&self, v_in: &[Complex], out: &mut [Complex], stage: usize, fstride: usize, in_stride: usize) {
        let p = self.stage_radix[stage];
        let m = self.stage_remainder[stage];
        let it = fstride * in_stride;

        if m == 1 {
            for i in 0..m*p {
                out[i] = v_in[i * it];
            }
        }
        else {
            for (i, j) in (0..m*p).step_by(m).enumerate() {
                self.transform(&v_in[i * it..], &mut out[j..], stage+1, fstride * p, in_stride);
            }
        }

        match p {
            2 => self.kf_butterfly2_simd(out, fstride, m),
            3 => self.kf_butterfly3_simd(out, fstride, m),
            4 => self.kf_butterfly4_simd(out, fstride, m),
            5 => self.kf_butterfly5_simd(out, fstride, m),
            _ => self.kf_butterfly_generic(out, fstride, m, p),
        }
    }

    fn kf_butterfly2(&self, v: &mut [Complex], fstride: usize, m: usize) {
        for i in 0..m {
            let t: Complex = v[i + m] * self.twiddlers[i * fstride];
            v[i + m] = v[i] - t;
            v[i] = v[i] + t;
        }
    }

    fn kf_butterfly3(&self, v: &mut [Complex], fstride: usize, m: usize) {
        let m2 = 2 * m;
        let mut temp: [Complex; 5] = [Default::default(); 5];
        let epi3 = self.twiddlers[m * fstride];

        for i in 0..m {
            temp[1] = v[i+m] * self.twiddlers[fstride * i];
            temp[2] = v[i+m2] * self.twiddlers[2 * fstride * i];

            temp[3] = temp[1] + temp[2];
            temp[0] = temp[1] - temp[2];

            v[m+i] = v[i] - 0.5 * temp[3];
            temp[0] = epi3.imaginary * temp[0];

            v[i] = v[i] + temp[3];

            v[i+m2] = Complex::new(v[i+m].real + temp[0].imaginary,
                                      v[i+m].imaginary - temp[0].real);

            v[i+m] = v[i+m] + Complex::new(-temp[0].imaginary, temp[0].real);
        }
    }

    fn kf_butterfly4(&self, v: &mut [Complex], fstride: usize, m: usize) {
        let mut temp: [Complex; 7] = [Default::default(); 7];
        let scale = if self.inverse { 1.0 } else { -1.0 };

        for i in 0..m {
            temp[0] = v[i + m]  * self.twiddlers[i * fstride];
            temp[1] = v[i + 2*m] * self.twiddlers[i * fstride * 2];
            temp[2] = v[i + 3*m] * self.twiddlers[i * fstride * 3];
            temp[5] = v[i] - temp[1];

            v[i] = v[i] + temp[1];
            temp[3] = temp[0] + temp[2];
            temp[4] = temp[0] - temp[2];
            temp[4] = Complex::new(temp[4].imaginary * scale,
                                   -temp[4].real * scale);

            v[i + 2*m] = v[i] - temp[3];
            v[i      ] = v[i] + temp[3];
            v[i + m  ] = temp[5] + temp[4];
            v[i + 3*m] = temp[5] - temp[4];
        }
    }

    fn kf_butterfly5(&self, v: &mut [Complex], fstride: usize, m: usize) {
        let ya = self.twiddlers[fstride * m];
        let yb = self.twiddlers[fstride * 2 * m];
        let mut temp: [Complex; 13] = [Default::default(); 13];

        for i in 0..m {
            temp[0] = v[i];

            temp[1] = v[i +   m] * self.twiddlers[  i*fstride];
            temp[2] = v[i + 2*m] * self.twiddlers[2*i*fstride];
            temp[3] = v[i + 3*m] * self.twiddlers[3*i*fstride];
            temp[4] = v[i + 4*m] * self.twiddlers[4*i*fstride];

            temp[7 ] = temp[1] + temp[4];
            temp[10] = temp[1] - temp[4];
            temp[8 ] = temp[2] + temp[3];
            temp[9 ] = temp[2] - temp[3];

            v[i] = v[i] + temp[7] + temp[8];

            temp[5] = temp[0] + Complex::new(
                temp[7].real * ya.real + temp[8].real * yb.real,
                temp[7].imaginary * ya.real + temp[8].imaginary * yb.real);

            temp[6] = Complex::new(
                temp[10].imaginary * ya.imaginary + temp[9].imaginary * yb.imaginary,
                -temp[10].real * ya.imaginary - temp[9].real * yb.imaginary);

            v[i +   m] = temp[5] - temp[6];
            v[i + 4*m] = temp[5] + temp[6];

            temp[11] = temp[0] + Complex::new(
                temp[7].real * yb.real + temp[8].real * ya.real,
                temp[7].imaginary * yb.real + temp[8].imaginary * ya.real);

            temp[12] = Complex::new(
                -temp[10].imaginary * yb.imaginary + temp[9].imaginary * ya.imaginary,
                temp[10].real * yb.imaginary - temp[9].real * ya.imaginary);

            v[i + 2*m] = temp[11] + temp[12];
            v[i + 3*m] = temp[11] - temp[12];
        }
    }

    fn kf_butterfly2_simd(&self, v: &mut [Complex], fstride: usize, m: usize) {
        for i in 0..m {
            unsafe {
                let mut x = v128_load((&self.twiddlers[i * fstride]) as *const Complex as *const v128);
                let mut yr: v128 = f64x2_splat(v[i + m].real);
                let mut yi: v128 = f64x2_splat(v[i + m].imaginary);

                yr = f64x2_mul(x, yr);
                let mut n1: v128 = i64x2_shuffle::<1,2>(x, x);
                yi = f64x2_mul(n1, yi);
                n1 = f64x2_sub(yr, yi);
                yr = f64x2_add(yr, yi);
                n1 = i64x2_shuffle::<0,3>(n1, yr);

                x = v128_load(&v[i] as *const Complex as *const v128);
                v128_store(&mut v[i + m] as *mut Complex as *mut v128, f64x2_sub(x, n1));
                v128_store(&mut v[i    ] as *mut Complex as *mut v128, f64x2_add(x, n1));
            }
        }
    }

    fn kf_butterfly3_simd(&self, v: &mut [Complex], fstride: usize, m: usize) {
        unsafe {
            let m2 = 2 * m;
            let mut temp: [v128; 6] = [f64x2_splat(0.0); 6];
            let epi3: v128 = f64x2_splat(self.twiddlers[m * fstride].imaginary);

            for i in 0..m {
                temp[1] = mult(v[i+m ], self.twiddlers[    fstride * i]);
                temp[2] = mult(v[i+m2], self.twiddlers[2 * fstride * i]);
                temp[4] = v128_load(&v[i] as *const Complex as *const v128);

                temp[3] = f64x2_add(temp[1], temp[2]);
                temp[0] = f64x2_sub(temp[1], temp[2]);

                let x = f64x2_sub(temp[4], f64x2_mul(f64x2_splat(0.5), temp[3]));

                temp[0] = f64x2_mul(temp[0], epi3);
                temp[0] = i64x2_shuffle::<1,2>(temp[0], temp[0]);
                temp[0] = v128_xor(temp[0], f64x2(0.0, -0.0));

                v128_store(&mut v[i] as *mut Complex as *mut v128, f64x2_add(temp[4], temp[3]));

                v128_store(&mut v[i+m2] as *mut Complex as *mut v128, f64x2_add(x, temp[0]));
                v128_store(&mut v[i+m] as *mut Complex as *mut v128, f64x2_sub(x, temp[0]));
            }
        }
    }

    fn kf_butterfly4_simd(&self, v: &mut [Complex], fstride: usize, m: usize) {
        unsafe {
            let mut temp: [v128; 7] = [f64x2(0.0, 0.0); 7];
            let scale = if self.inverse { f64x2(0.0, -0.0) } else { f64x2(-0.0, 0.0) };
            for i in 0..m {
                temp[0] = mult(v[i + m], self.twiddlers[i * fstride]);
                temp[1] = mult(v[i + 2*m], self.twiddlers[i * fstride * 2]);
                temp[2] = mult(v[i + 3*m], self.twiddlers[i * fstride * 3]);
                temp[6] = v128_load(&v[i] as *const Complex as *const v128);
                temp[5] = f64x2_sub(temp[6], temp[1]);

                temp[6] = f64x2_add(temp[6], temp[1]);
                temp[3] = f64x2_add(temp[0], temp[2]);

                temp[4] = f64x2_sub(temp[0], temp[2]);
                temp[4] = i64x2_shuffle::<1,2>(temp[4], temp[4]);
                temp[4] = v128_xor(temp[4], scale);

                v128_store(&mut v[i + 2*m] as *mut Complex as *mut v128, f64x2_sub(temp[6], temp[3]));
                v128_store(&mut v[i      ] as *mut Complex as *mut v128, f64x2_add(temp[6], temp[3]));
                v128_store(&mut v[i +   m] as *mut Complex as *mut v128, f64x2_add(temp[5], temp[4]));
                v128_store(&mut v[i + 3*m] as *mut Complex as *mut v128, f64x2_sub(temp[5], temp[4]));
            }
        }
    }

    fn kf_butterfly5_simd(&self, v: &mut [Complex], fstride: usize, m: usize) {
        unsafe {
            let yar = f64x2_splat(self.twiddlers[fstride * m].real);
            let yai = f64x2(-self.twiddlers[fstride * m].imaginary, self.twiddlers[fstride * m].imaginary);
            let ybr = f64x2_splat(self.twiddlers[fstride * 2 * m].real);
            let ybi = f64x2(self.twiddlers[fstride * 2 * m].imaginary, -self.twiddlers[fstride * 2 * m].imaginary);
            let mut temp: [v128; 13] = [f64x2(0.0, 0.0); 13];

            for i in 0..m {
                temp[0] = v128_load(&v[i] as *const Complex as *const v128);

                temp[1] = mult(v[i +   m], self.twiddlers[  i*fstride]);
                temp[2] = mult(v[i + 2*m], self.twiddlers[2*i*fstride]);
                temp[3] = mult(v[i + 3*m], self.twiddlers[3*i*fstride]);
                temp[4] = mult(v[i + 4*m], self.twiddlers[4*i*fstride]);

                temp[7 ] = f64x2_add(temp[1], temp[4]);
                temp[10] = f64x2_sub(temp[1], temp[4]);
                temp[8 ] = f64x2_add(temp[2], temp[3]);
                temp[9 ] = f64x2_sub(temp[2], temp[3]);

                v128_store(&mut v[i] as *mut Complex as *mut v128, f64x2_add(temp[0], f64x2_add(temp[7], temp[8])));

                temp[5] = f64x2_add(temp[0], f64x2_add(
                        f64x2_mul(temp[7], yar), f64x2_mul(temp[8], ybr)));

                temp[6] = f64x2_sub(f64x2_mul(temp[10], yai), f64x2_mul(temp[9], ybi));
                temp[6] = i64x2_shuffle::<1,2>(temp[6], temp[6]);

                v128_store(&mut v[i +   m] as *mut Complex as *mut v128, f64x2_sub(temp[5], temp[6]));
                v128_store(&mut v[i + 4*m] as *mut Complex as *mut v128, f64x2_add(temp[5], temp[6]));

                temp[11] = f64x2_add(temp[0], f64x2_add(
                        f64x2_mul(temp[7], ybr), f64x2_mul(temp[8], yar)));

                temp[12] = f64x2_add(f64x2_mul(temp[9], yai), f64x2_mul(temp[10], ybi));
                temp[12] = i64x2_shuffle::<1,2>(temp[12], temp[12]);

                v128_store(&mut v[i + 2*m] as *mut Complex as *mut v128, f64x2_add(temp[11], temp[12]));
                v128_store(&mut v[i + 3*m] as *mut Complex as *mut v128, f64x2_sub(temp[11], temp[12]));
            }
        }
    }

    fn kf_butterfly_generic(&self, v: &mut [Complex], fstride: usize, m: usize, p: usize) {
        let mut temp = vec![Default::default(); p];

        for i in 0..m {
            let mut k = i;
            for j in 0..p {
                temp[j] = v[k];
                k += m;
            }

            k = i;

            for _ in 0..p {
                let mut twidx = 0;
                v[k] = temp[0];
                for j in 1..p {
                    twidx += fstride * k;
                    if twidx >= self.nfft {
                        twidx -= self.nfft;
                    }
                    v[k] = v[k] + temp[j] * self.twiddlers[twidx];
                }
                k += m;
            }

        }
    }
}

pub fn pmul(a: &[f64], b: &[f64]) -> Vec<f64> { // scaled by factor of len
    let len = find_ugly(a.len() + b.len() - 1);
    let mut fft = FFT::new(len, false);
    let mut x = vec![Default::default(); len];
    fft.transform(&a.iter().cloned()
               .chain(std::iter::repeat(0.0))
               .take(len).map(Complex::from).collect::<Vec<_>>(),
               &mut x, 0, 1, 1);
    let mut y = vec![Default::default(); len];
    fft.transform(&b.iter().cloned()
               .chain(std::iter::repeat(0.0))
               .take(len).map(Complex::from).collect::<Vec<_>>(),
               &mut y, 0, 1, 1);
    let v3 = x.into_iter().zip(y.into_iter())
        .map(|(a,b)| a * b).collect::<Vec<_>>();
    fft.invert();
    fft.ifft(&v3).into_iter().map(|i| i.real / len as f64).collect()
}


fn find_ugly(k: usize) -> usize {
    let items = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48, 50, 54, 60, 64, 72, 75, 80, 81, 90, 96, 100, 108, 120, 125, 128, 135, 144, 150, 160, 162, 180, 192, 200, 216, 225, 240, 243, 250, 256, 270, 288, 300, 320, 324, 360, 375, 384, 400, 405, 432, 450, 480, 486, 500, 512, 540, 576, 600, 625, 640, 648, 675, 720, 729, 750, 768, 800, 810, 864, 900, 960, 972, 1000, 1024, 1080, 1125, 1152, 1200, 1215, 1250, 1280, 1296, 1350, 1440, 1458, 1500, 1536, 1600, 1620, 1728, 1800, 1875, 1920, 1944, 2000, 2025, 2048, 2160, 2187, 2250, 2304, 2400, 2430, 2500, 2560, 2592, 2700, 2880, 2916, 3000, 3072, 3125, 3200, 3240, 3375, 3456, 3600, 3645, 3750, 3840, 3888, 4000, 4050, 4096, 4320, 4374, 4500, 4608, 4800, 4860, 5000, 5120, 5184];

    let mut low: usize = 0;
    let mut high: usize = items.len() - 1;

    while low <= high {
        let middle = (high + low) / 2;
        if let Some(current) = items.get(middle) {
            if *current == k {
                return items[middle];
            }
            if *current > k {
                if middle == 0 {
                    return 1;
                }
                high = middle - 1
            }
            if *current < k {
                low = middle + 1
            }
        }
    }
    items[low]
}

fn mult(a: Complex, b: Complex) -> v128 {
    unsafe {
        let x = v128_load((&a) as *const Complex as *const v128);
        let mut yr: v128 = f64x2_splat(b.real);
        let mut yi: v128 = f64x2_splat(b.imaginary);

        yr = f64x2_mul(x, yr);
        let mut n1: v128 = i64x2_shuffle::<1,2>(x, x);
        yi = f64x2_mul(n1, yi);
        n1 = f64x2_sub(yr, yi);
        yr = f64x2_add(yr, yi);
        i64x2_shuffle::<0,3>(n1, yr)
    }
}
