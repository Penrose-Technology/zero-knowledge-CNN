use ark_sumcheck::gkr_round_sumcheck::GKRRoundSumcheck;
use ark_sumcheck::ml_sumcheck::{Proof, data_structures::PolynomialInfo};
use ark_sumcheck::rng::{Blake2b512Rng, FeedableRNG};
use ark_poly::{DenseMultilinearExtension, SparseMultilinearExtension};
use ark_ff::FftField;

use crate::convolution::IPForConvolution;
use crate::convolution::fft::{FftPublicInfo, Convpolynomial};
use crate::convolution::gkr::{GKRConv, GKRConvProof};

impl <F: FftField> IPForConvolution<F> { 
    pub fn fft_prove (
            r: &Vec<F>, 
            public_info: &mut FftPublicInfo<F>, 
            poly: &mut Convpolynomial<F>
        ) -> (PolynomialInfo, Proof<F>, Vec<F>) {
        // generate omega extension
        public_info.u = r.clone();
        public_info.omega_init();

        let poly_p = poly.fft_load_poly(&public_info);
        let (proof, randomness) = Convpolynomial::fft_ifft_sumcheck_prove(&poly_p);

        (poly_p.info(), proof, randomness)
    }

    pub fn ifft_prove (
            r: &Vec<F>, 
            public_info: &mut FftPublicInfo<F>, 
            poly: &mut Convpolynomial<F>
        ) -> (PolynomialInfo, Proof<F>, Vec<F>) {
        // generate inv_omega extension
        public_info.u = r.clone();
        public_info.inv_omega_init();

        let poly_p = poly.ifft_load_poly(&public_info);
        let (proof, randomness) = Convpolynomial::fft_ifft_sumcheck_prove(&poly_p);

        (poly_p.info(), proof, randomness)
    }

    pub fn hadmard_prove (
        g: &Vec<F>,  
        x_i: &DenseMultilinearExtension<F>,
        w_i: &DenseMultilinearExtension<F>,
        multi_i: &SparseMultilinearExtension<F>,
    ) -> (GKRConvProof<F>, Vec<F>, Vec<F>) {
        let mut fs_rng = Blake2b512Rng::setup();
        let (proof, u, v) = <GKRRoundSumcheck<F> as GKRConv<F>>::prove(&mut fs_rng, &multi_i, &x_i, &w_i, g);

        (proof, u, v)
    }
}

