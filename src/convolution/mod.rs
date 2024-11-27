use ark_std::marker::PhantomData;
use ark_ff::FftField;
use ark_poly::{DenseMultilinearExtension, SparseMultilinearExtension, GeneralEvaluationDomain, Polynomial};
use ark_sumcheck::ml_sumcheck::protocol::verifier::SubClaim;
use ark_sumcheck::Error;
use ark_sumcheck::ml_sumcheck::{data_structures::PolynomialInfo, MLSumcheck, Proof};
use crate::convolution::gkr::GKRConvProof;
use crate::convolution::fft::{Convpolynomial, FftPublicInfo, led_sort};

pub mod fft;
pub mod gkr;
pub mod prover;
pub mod verifier;
pub mod test;

pub struct IPForConvolution <F: FftField> {
    _marker: PhantomData<F>,
}

pub struct ConvProof <F: FftField> {
    pub x_poly_info: PolynomialInfo,
    pub x_proof: Proof<F>,
    pub w_poly_info: PolynomialInfo,
    pub w_proof: Proof<F>,
    pub y_poly_info: PolynomialInfo,
    pub y_proof: Proof<F>,
    pub hadmard_proof: GKRConvProof<F>,
}

pub struct ConvSubClaim <F: FftField> {
    pub x: SubClaim<F>,
    pub w: SubClaim<F>
}

pub struct MLConvolution <F: FftField> (PhantomData<F>);


pub trait Conv <F: FftField> {
    fn evaluate (
        domain: &GeneralEvaluationDomain<F>, 
        x_poly: &mut Convpolynomial<F>,
        w_poly: &mut Convpolynomial<F>,
    ) -> Convpolynomial<F>;

    fn prove (
        domain: &GeneralEvaluationDomain<F>, 
        g: &Vec<F>,
        x_poly: &mut Convpolynomial<F>,
        w_poly: &mut Convpolynomial<F>,
        y_poly: &mut Convpolynomial<F>,
    ) -> ConvProof<F>;

    fn verify (
        g: &Vec<F>,
        domain: &GeneralEvaluationDomain<F>, 
        proof: &ConvProof<F>,
    ) -> Result<ConvSubClaim<F>, Error>;


}


impl <F: FftField> Conv<F> for MLConvolution<F> {

    fn evaluate (
        domain: &GeneralEvaluationDomain<F>, 
        x_poly: &mut Convpolynomial<F>,
        w_poly: &mut Convpolynomial<F>,
    ) -> Convpolynomial<F> {

        let public_info = FftPublicInfo::new(domain);

        // evaluate x' = fft(x)
        x_poly.fft_evaluation(&public_info);

        // evaluate w' = fft(w)
        w_poly.fft_evaluation(&public_info);

        // evaluate y = ifft(x' * w')
        let y_points: Vec<F> = x_poly.poly_after_fft_ifft.iter()
                                                        .zip(w_poly.poly_after_fft_ifft.iter())
                                                        .map(|(x, w)| *x * *w)
                                                        .collect();
        let mut y_poly = Convpolynomial::new(y_points);
        y_poly.ifft_evaluation(&public_info);

        y_poly
    }

    fn prove (
        domain: &GeneralEvaluationDomain<F>, 
        g: &Vec<F>,
        x_poly: &mut Convpolynomial<F>,
        w_poly: &mut Convpolynomial<F>,
        y_poly: &mut Convpolynomial<F>,
    ) -> ConvProof<F> {
        assert_eq!(x_poly.log_size, w_poly.log_size, "X_poly and W_poly size mismatch!");
        assert_eq!(x_poly.log_size, y_poly.log_size, "X_poly and Y_poly size mismatch!");

        let mut public_info = FftPublicInfo::new(domain);

        // ifft prove    
        assert_ne!(y_poly.poly_after_fft_ifft.len(), 0, "Y_poly hasn't been performed ifft!");
        let (y_poly_info, y_proof, y_randomness) = IPForConvolution::ifft_prove(g, &mut public_info, y_poly);
        
        // hadmard prove
        assert_ne!(x_poly.poly_after_fft_ifft.len(), 0, "X_poly hasn't been performed fft!");
        assert_ne!(w_poly.poly_after_fft_ifft.len(), 0, "W_poly hasn't been performed fft!");
        let size = x_poly.log_size;
        let x_i = &DenseMultilinearExtension::from_evaluations_vec(x_poly.log_size, led_sort(&x_poly.poly_after_fft_ifft));
        let w_i = &DenseMultilinearExtension::from_evaluations_vec(w_poly.log_size, led_sort(&w_poly.poly_after_fft_ifft));
        let gate_idx: Vec<(usize, F)> = (0..y_poly.size).map(|x| (x + (x << size) + (x << (size << 1)), F::from(1))).collect();
        let multi_i = SparseMultilinearExtension::from_evaluations(3 * size, &gate_idx);
        let (hadmard_proof, x_fft_randomness, w_fft_randomness) = IPForConvolution::hadmard_prove(&y_randomness, x_i, w_i, &multi_i);

        //fft prove
        let (x_poly_info, x_proof, _) = IPForConvolution::fft_prove(&x_fft_randomness, &mut public_info, x_poly);
        let (w_poly_info, w_proof, _) = IPForConvolution::fft_prove(&w_fft_randomness, &mut public_info, w_poly);

        ConvProof { 
            x_poly_info, 
            x_proof, 
            w_poly_info, 
            w_proof, 
            y_poly_info, 
            y_proof, 
            hadmard_proof,
        }
    }

    fn verify (
        g: &Vec<F>,
        domain: &GeneralEvaluationDomain<F>, 
        proof: &ConvProof<F>,
    ) -> Result<ConvSubClaim<F>, Error> {
        assert_eq!(proof.x_poly_info.num_variables, proof.w_poly_info.num_variables, "X_poly and W_poly size mismatch!");
        assert_eq!(proof.x_poly_info.num_variables, proof.y_poly_info.num_variables, "X_poly and Y_poly size mismatch!");

        let mut public_info = FftPublicInfo::new(domain);

        // ifft verify 
        let y_subclaim = IPForConvolution::fft_ifft_verify(&proof.y_proof, &proof.y_poly_info);
                
        // generate inv_omega extension
        public_info.u = g.clone();
        public_info.inv_omega_init();

        let inv_omega_r = DenseMultilinearExtension::from_evaluations_vec(
            proof.y_poly_info.num_variables, 
            led_sort(&public_info.inv_omega_ext)
        ).evaluate(&y_subclaim.point);
        
        // ifft -> hadmard check
        let y_poly_size = (1 << proof.y_poly_info.num_variables) as u32;
        let y_expection_r = y_subclaim.expected_evaluation * inv_omega_r.inverse().unwrap() * F::from(y_poly_size);

        // hadmard verify
        let subclaim_gkr = IPForConvolution::hadmard_verify(
                                                            proof.x_poly_info.num_variables, 
                                                            &proof.hadmard_proof, 
                                                            y_expection_r).unwrap();

        // wire prediction (f1) evaluates                                                  
        let gate_idx: Vec<(usize, F)> = (0..y_poly_size as usize).map(
                                                                |x| 
                                                                (x + (x << proof.y_poly_info.num_variables) + (x << (proof.y_poly_info.num_variables << 1)), F::from(1))
                                                                ).collect();
        let multi_i = SparseMultilinearExtension::from_evaluations(3 * proof.y_poly_info.num_variables, &gate_idx);
        let guv: Vec<F> = y_subclaim.point.clone().into_iter()
                            .chain(subclaim_gkr.u.clone().into_iter())
                            .chain(subclaim_gkr.v.clone().into_iter())
                            .collect();
        let multi_i_guv = multi_i.evaluate(&guv);

        // fft verify
        let x_subclaim = IPForConvolution::fft_ifft_verify(&proof.x_proof, &proof.x_poly_info);
        let x_sum_u = MLSumcheck::extract_sum(&proof.x_proof);

        let w_subclaim = IPForConvolution::fft_ifft_verify(&proof.w_proof, &proof.w_poly_info);
        let w_sum_v = MLSumcheck::extract_sum(&proof.w_proof);

        // hadmard -> fft check
        if multi_i_guv * x_sum_u * w_sum_v != subclaim_gkr.expected_evaluation {
            return Err(Error::Reject(Some("Convolution Verification Failed!".into())));
        }

        Ok(ConvSubClaim{
            x: x_subclaim, 
            w: w_subclaim
        })            
    }
}



