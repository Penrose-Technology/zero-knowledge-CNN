use crate::convolution::gkr::{GKRConv, GKRConvProof};
use ark_sumcheck::rng::{Blake2b512Rng, FeedableRNG};
use ark_sumcheck::ml_sumcheck::{protocol::verifier::SubClaim,
                                data_structures::PolynomialInfo,
                                Proof
                            };
use ark_sumcheck::gkr_round_sumcheck::{GKRRoundSumcheck, 
                                       data_structures::GKRRoundSumcheckSubClaim
                                    };
use ark_sumcheck::Error;
use ark_ff::FftField;

use crate::convolution::IPForConvolution;

use super::fft::Convpolynomial;

impl <F: FftField> IPForConvolution<F> {
    pub fn fft_ifft_verify(
        proof: &Proof<F>,
        poly_info: &PolynomialInfo
    ) -> SubClaim<F> {
        let subclaim = Convpolynomial::fft_ifft_sumcheck_verify(proof, poly_info);
        subclaim
    }

    pub fn hadmard_verify(
        size: usize,
        proof: &GKRConvProof<F>,
        claimed_sum: F,
    ) -> Result<GKRRoundSumcheckSubClaim<F>, Error> {
        let mut fs_rng = Blake2b512Rng::setup();
        let subclaim = <GKRRoundSumcheck<F> as GKRConv<F>>::verify(&mut fs_rng, size, proof, claimed_sum);
        subclaim
    }
}