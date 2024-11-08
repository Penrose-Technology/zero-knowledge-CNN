use ark_ff::FftField;
use ark_sumcheck::Error;
use crate::conv_relu::data_structure::{GlobalParameters, 
                            ProverParameters, InputDate, VerifierParameters,
                            IPForConvRelu, Proof, Subclaim};


pub mod data_structure;
pub mod setup;
pub mod prover;
pub mod verifier;
pub mod test;


impl <F: FftField> IPForConvRelu<F> {
    pub fn setup() -> (
        GlobalParameters<F>, 
        ProverParameters<F>, 
        VerifierParameters<F>,
        InputDate) {
        let gp = Self::load_global_parameters();
        let pp = Self::load_prover_parameters();
        let vp = Self::load_verifier_parameters(&gp);
        let input = Self::load_input();

        (gp, pp, vp, input)
    }

    pub fn prove(
        gp: &GlobalParameters<F>,
        pp: &mut ProverParameters<F>,
        input: &InputDate,
    ) -> Proof<F> {
        let conv_proof = IPForConvRelu::conv_prove(gp, pp, input);
        let truncate_proof = IPForConvRelu::truncate_prove(gp, pp);
        let relu_proof = IPForConvRelu::relu_prove(&gp, pp);

        Proof { 
            conv_proof, 
            truncate_proof, 
            relu_proof,}
    }

    pub fn verify(
        gp: &GlobalParameters<F>,
        vp: &VerifierParameters<F>,
        proof: &Proof<F>
    ) -> Result<Subclaim<F>, Error> {
        let conv_subclaim = IPForConvRelu::conv_verify(gp, vp, &proof.conv_proof)?;
        let truncate_subclaim = IPForConvRelu::truncate_verify(vp, &proof.truncate_proof)?;
        let relu_subclaim = IPForConvRelu::relu_verify(vp, &proof.relu_proof)?;

        Ok(Subclaim { 
            conv_subclaim, 
            truncate_subclaim, 
            relu_subclaim,
        })
    }
}



