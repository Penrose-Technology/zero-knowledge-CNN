use ark_ff::FftField;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_sumcheck::ml_sumcheck:: MLSumcheck;
use crate::convolution::{Conv, MLConvolution};
use crate::ml_lookup::MLLookupTable;
use crate::conv_relu::data_structure::{GlobalParameters, 
                            VerifierParameters,
                            IPForConvRelu};
use ark_sumcheck::ml_sumcheck::Proof as TruncateProof;
use crate::convolution::ConvProof;
use crate::ml_lookup::protocol::prover::ProverMsg as ReluProof;
use crate::convolution::ConvSubClaim;
use crate::ml_lookup::protocol::verifier::LookupSubClaim;
use ark_sumcheck::ml_sumcheck::protocol::verifier::SubClaim;
use ark_sumcheck::Error;


impl <F: FftField> IPForConvRelu<F> {
    pub fn conv_verify(
        gp: &GlobalParameters<F>,
        vp: &VerifierParameters<F>,
        proof: &ConvProof<F>,
    ) -> Result<ConvSubClaim<F>, Error> {
        let domain = GeneralEvaluationDomain::<F>::new(gp.input_size.0 * gp.input_size.1).unwrap();
        MLConvolution::verify(&vp.conv_info.g, &domain, proof)
    }

    pub fn truncate_verify(
        vp: &VerifierParameters<F>,
        proof: &TruncateProof<F>,
    ) -> Result<SubClaim<F>, Error> {
        MLSumcheck::verify(&vp.truncate_info.poly_info, F::from(0), proof)
                                                
    }

    pub fn relu_verify(
        vp: &VerifierParameters<F>,
        proof: &ReluProof<F>,
    ) -> Result<LookupSubClaim<F>, Error> {
        MLLookupTable::verify(&vp.relu_info.table_info, proof)
    }

}