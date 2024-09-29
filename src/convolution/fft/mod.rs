use ark_poly::{DenseMultilinearExtension, EvaluationDomain, GeneralEvaluationDomain};
use ark_sumcheck::ml_sumcheck::protocol::{ListOfProductsOfPolynomials, verifier::SubClaim};
use ark_sumcheck::ml_sumcheck::{MLSumcheck, Proof};
use ark_sumcheck::rng::{Blake2b512Rng, FeedableRNG};
use ark_ff::FftField;
use ark_std::rc::Rc;
use std::iter::successors;


pub struct FftPublicInfo <'a, F: FftField> {
    /// domain
    pub domain: &'a GeneralEvaluationDomain<F>,
    /// domain size
    pub size: u64,
    /// log domain size
    pub log_size: u32,
    /// roots of unity
    pub omega_vec: Vec<F>,
    /// inverse roots of unity
    pub inv_omega_vec: Vec<F>,
    /// fft random challenge
    pub u: Vec<F>,
    /// fft after extension
    pub omega_ext: Vec<F>,
    /// ifft after extension
    pub inv_omega_ext: Vec<F>,
}

impl <'a, F: FftField> FftPublicInfo<'a, F> {
    pub fn new(domain: &'a GeneralEvaluationDomain<F>) -> Self {
        match domain {
            GeneralEvaluationDomain::Radix2(radix2_domain) => {
                
                Self {
                    domain,
                    size: radix2_domain.size,
                    log_size: radix2_domain.log_size_of_group,
                    omega_vec: successors(Some(F::one()), 
                                          |&prev| Some(prev * radix2_domain.group_gen))
                                        .take(radix2_domain.size as usize)
                                        .collect(),
                    inv_omega_vec: successors(Some(F::one()), 
                                              |&prev| Some(prev * radix2_domain.group_gen_inv))
                                            .take(radix2_domain.size as usize)
                                            .collect(),
                    u: Vec::new(),
                    omega_ext: Vec::new(),
                    inv_omega_ext: Vec::new(),
                }
            }
            GeneralEvaluationDomain::MixedRadix(mixedradix_domain) => {
                Self {
                    domain,
                    size: mixedradix_domain.size,
                    log_size: mixedradix_domain.log_size_of_group,
                    omega_vec: successors(Some(F::one()), 
                                          |&prev| Some(prev * mixedradix_domain.group_gen))
                                        .take(mixedradix_domain.size as usize)
                                        .collect(),
                    inv_omega_vec: successors(Some(F::one()), 
                                              |&prev| Some(prev * mixedradix_domain.group_gen_inv))
                                            .take(mixedradix_domain.size as usize)
                                            .collect(),
                    u: Vec::new(),
                    omega_ext: Vec::new(),
                    inv_omega_ext: Vec::new(),
                }
            }
        }
    }

    pub fn omega_init(&mut self) {
        let size = self.size.try_into().expect("size can't convert to usize type!");
        let log_size = self.log_size.try_into().expect("log_size can't convert to usize type!");
        assert_eq!(size, self.omega_vec.len(), "Illegal omega vector length!");
        assert_eq!(log_size, self.u.len(), "Illegal random point u length!");

        let mut omega_ext = vec![F::one(); size];
        let mut mask = 0;

        for i in 0..log_size {
            for j in (0..1<<(i + 1)).rev() {   
                omega_ext[j] = omega_ext[j & mask] * 
                        (
                            F::one() - self.u[i] + 
                            self.u[i] * self.omega_vec[j * (size >> (i + 1))] 
                        )
            }
            mask = (mask << 1) + 1;
        }

        self.omega_ext = omega_ext;
    }

    pub fn inv_omega_init(&mut self) {
        let size = self.size.try_into().expect("size can't convert to usize type!");
        let log_size = self.log_size.try_into().expect("log_size can't convert to usize type!");
        assert_eq!(size, self.inv_omega_vec.len(), "Illegal inv_omega vector length!");
        assert_eq!(log_size, self.u.len(), "Illegal random point u length!");

        let mut inv_omega_ext = vec![F::one(); size];
        let mut mask = 0;
        for i in 0..log_size {
            for j in (0..1<<(i + 1)).rev() {   
                inv_omega_ext[j] = inv_omega_ext[j & mask] * 
                        (
                            F::one() - self.u[i] + 
                            self.u[i] * self.inv_omega_vec[j * (size >> (i + 1))] 
                        )
            }
            mask = (mask << 1) + 1;
        }

        self.inv_omega_ext = inv_omega_ext;
    }
}

pub struct Convpolynomial<F: FftField> {
    pub poly: Vec<F>,
    pub size: usize,
    pub log_size: usize,
    pub poly_after_fft_ifft: Vec<F>,
}

impl <F: FftField> Convpolynomial<F> {
    pub fn new(points: Vec<F>) -> Self {
        Self {
            poly: points.clone(),
            size: points.len(),
            log_size: points.len().trailing_zeros() as usize,
            poly_after_fft_ifft: Vec::new(),
        }
    }

    pub fn fft_evaluation(&mut self, public_info: &FftPublicInfo<F>) {
        assert_eq!(self.poly_after_fft_ifft.len(), 0, "Can't perform fft, sequence is not empty!");
        self.poly_after_fft_ifft = public_info.domain.fft(&self.poly);
    }

    pub fn ifft_evaluation(&mut self, public_info: &FftPublicInfo<F>) {
        assert_eq!(self.poly_after_fft_ifft.len(), 0, "Can't perform ifft, sequence is not empty!");
        self.poly_after_fft_ifft = public_info.domain.ifft(&self.poly);
    }

    pub fn fft_load_poly(&self, public_info: &FftPublicInfo<F>) -> ListOfProductsOfPolynomials<F> {
        let mut poly = ListOfProductsOfPolynomials::new(self.log_size);

        let multi_omega_ext = DenseMultilinearExtension::from_evaluations_vec(self.log_size, public_info.omega_ext.clone());
        let multi_conv_poly = DenseMultilinearExtension::from_evaluations_vec(self.log_size, self.poly.clone());
        let product = vec![Rc::new(multi_omega_ext), Rc::new(multi_conv_poly)];
        poly.add_product(product, F::one());

        poly
    }

    pub fn ifft_load_poly(&self, public_info: &FftPublicInfo<F>) -> ListOfProductsOfPolynomials<F> {
        let mut poly = ListOfProductsOfPolynomials::new(self.log_size);

        let multi_inv_omega_ext = DenseMultilinearExtension::from_evaluations_vec(self.log_size, public_info.inv_omega_ext.clone());
        let multi_conv_poly = DenseMultilinearExtension::from_evaluations_vec(self.log_size, self.poly.clone());
        let product = vec![Rc::new(multi_inv_omega_ext), Rc::new(multi_conv_poly)];
        poly.add_product(product, F::from(self.size as u32).inverse().unwrap());

        poly
    }

}

impl <F: FftField> Convpolynomial<F> {
    pub fn fft_ifft_sumcheck_prove(poly: &ListOfProductsOfPolynomials<F>) -> Proof<F> {
        let mut fs_rng = Blake2b512Rng::setup();
        let proof = MLSumcheck::prove_as_subprotocol(&mut fs_rng, &poly)
                                                                    .map(|x| x.0)
                                                                    .expect("fail to fft sumcheck prove!");
        proof
    }

    pub fn fft_ifft_sumcheck_verify(proof: &Proof<F>, poly: &ListOfProductsOfPolynomials<F>) -> SubClaim<F>{
        let mut fs_rng = Blake2b512Rng::setup();
        let claimed_sum = MLSumcheck::extract_sum(&proof);
        let fft_claim = MLSumcheck::verify_as_subprotocol(&mut fs_rng, &poly.info(), claimed_sum, &proof)
                                                                                    .expect("fail to fft sumcheck verify!");
        fft_claim
    }
}


pub fn led_sort <F: FftField> (sequence: &Vec<F>) -> Vec<F> {
    assert_eq!(sequence.len().is_power_of_two(), true, "Invalid sequence length!");

    let bitwidth = sequence.len().trailing_zeros() as usize;
    let mut sequence_bitrev = Vec::new();

    for idx in 0..(1 << bitwidth) as usize {
        let index: Vec<u32> = format!("{:0width$b}", idx, width=bitwidth)
                                    .chars()
                                    .map(|x| x.to_digit(2).unwrap())
                                    .collect();

        let mut index_rev = index.clone();
        index_rev.reverse();

        let idx_rev = index_rev.iter()
                                      .fold(0, 
                                        |acc, &bit| (acc << 1) | bit as usize
                                      );
        
        sequence_bitrev.push(sequence[idx_rev]);
    }

    sequence_bitrev
}




#[cfg(test)]
mod fft_tests {
    use ark_test_curves::bls12_381::Fr;
    use ark_poly::{Polynomial, DenseMultilinearExtension, EvaluationDomain, GeneralEvaluationDomain};
    use ark_ff::UniformRand;
    use ark_sumcheck::ml_sumcheck::MLSumcheck;
    use ark_std::test_rng;
    use crate::convolution::fft::{FftPublicInfo, Convpolynomial, led_sort};

    #[test]
    fn omega_init_test() {
        let mut rng = test_rng();
        let num_coeffs = 4;

        let domain = GeneralEvaluationDomain::<Fr>::new(num_coeffs).unwrap();
        let points = vec![Fr::from(11), Fr::from(12), Fr::from(13), Fr::from(14)];
        let mut poly = Convpolynomial::new(points);
        let u = vec![Fr::rand(&mut rng), Fr::rand(&mut rng)];
        let mut fft_public_info = FftPublicInfo::new(&domain);
        fft_public_info.u = u.clone();
        fft_public_info.omega_init();
 
        dbg!(&fft_public_info.omega_ext);

        poly.fft_evaluation(&fft_public_info);
        let p_fft_u = DenseMultilinearExtension::from_evaluations_vec(
                                                                        num_coeffs.trailing_zeros() as usize, 
                                                                        led_sort(&poly.poly_after_fft_ifft)
                                                                    ).evaluate(&u);
        dbg!(p_fft_u);

        let poly_p = poly.fft_load_poly(&fft_public_info);
        let proof = Convpolynomial::fft_ifft_sumcheck_prove(&poly_p);
        Convpolynomial::fft_ifft_sumcheck_verify(&proof, &poly_p);

        assert_eq!(MLSumcheck::extract_sum(&proof), p_fft_u, "fft sumcheck failed!");
    }

    #[test]
    fn inv_omega_init_test() {
        let mut rng = test_rng();
        let num_coeffs = 4;

        let domain = GeneralEvaluationDomain::<Fr>::new(num_coeffs).unwrap();
        let points = vec![Fr::from(11), Fr::from(12), Fr::from(13), Fr::from(14)];
        let mut poly = Convpolynomial::new(points);
        let u = vec![Fr::rand(&mut rng), Fr::rand(&mut rng)];
        let mut ifft_public_info = FftPublicInfo::new(&domain);
        ifft_public_info.u = u.clone();
        ifft_public_info.inv_omega_init();
 
        dbg!(&ifft_public_info.inv_omega_ext);

        poly.ifft_evaluation(&ifft_public_info);
        let p_ifft_u = DenseMultilinearExtension::from_evaluations_vec(
                                                                        num_coeffs.trailing_zeros() as usize, 
                                                                        led_sort(&poly.poly_after_fft_ifft)
                                                                    ).evaluate(&u);
        dbg!(p_ifft_u);

        let poly_p = poly.ifft_load_poly(&ifft_public_info);
        let proof = Convpolynomial::fft_ifft_sumcheck_prove(&poly_p);
        Convpolynomial::fft_ifft_sumcheck_verify(&proof, &poly_p);

        assert_eq!(MLSumcheck::extract_sum(&proof), p_ifft_u, "ifft sumcheck failed!");
    }
}