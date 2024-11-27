use ark_ff::FftField;
use ark_poly::DenseMultilinearExtension;
use ark_std::marker::PhantomData;
use serde::{Deserialize, Serialize};
use ark_sumcheck::ml_sumcheck::data_structures::PolynomialInfo;
use crate::ml_lookup::data_structures::LookupTableInfo;
use crate::convolution::{ConvProof, ConvSubClaim};
use crate::ml_lookup::protocol::verifier::LookupSubClaim;
use ark_sumcheck::ml_sumcheck::protocol::verifier::SubClaim as SubSubclaim;
use ark_sumcheck::ml_sumcheck::Proof as TruncateProof;
use crate::ml_lookup::protocol::prover::ProverMsg as ReluProof;

pub struct IPForConvRelu<F: FftField> {
    #[doc(hidden)]
    _marker: PhantomData<F>,
}

pub struct  GlobalParameters<F: FftField> {
    pub x_zero_point: u32,
    pub x_scalar: f64,
    pub w_zero_point: u32,
    pub w_scalar: f64,
    pub r_scalar: f64,
    pub bitwidth: u16,
    pub lookup_ceil: u32,
    pub lookup_floor: u32,
    pub int_scalar: usize,
    pub kernel_size: (usize, usize),
    pub input_size: (usize, usize),
    
    pub convolution_check_point: F,
    pub relu_combination_point: F,
    pub num_private_columns: usize,
    pub group_length: usize,
    pub public_table_column: DenseMultilinearExtension<F>,
}

#[derive(Deserialize)]
pub struct InputDate {
    pub x_i: Vec<Vec<u8>>,
    pub w_i: Vec<Vec<u8>>,
}

pub struct OutputData<F> {
    pub r_i_raw: Vec<F>,
    pub r_i: Vec<Vec<u8>>
}

#[derive(Serialize)]
pub struct ProverParameters<F: FftField> {
    pub x_i_fp: Vec<F>,
    pub w_i_fp: Vec<F>,
    pub y_i_fp: Vec<F>,
    pub r_i_fp: Vec<F>,
}

pub struct VerifierParameters<F: FftField> {
    pub conv_info: ConvInfo<F>,
    pub truncate_info: TruncateInfo,
    pub relu_info: ReluInfo<F>,
}

pub struct ConvInfo<F: FftField> {
    pub g: Vec<F>,
}

pub struct TruncateInfo {
    pub poly_info: PolynomialInfo,
}

pub struct ReluInfo<F: FftField> {
    pub table_info: LookupTableInfo<F>,
}

pub struct Proof<F: FftField> {
    pub conv_proof: ConvProof<F>,
    pub truncate_proof: TruncateProof<F>,
    pub relu_proof: ReluProof<F>,
}

pub struct Subclaim<F: FftField> {
    pub conv_subclaim: ConvSubClaim<F>,
    pub truncate_subclaim: SubSubclaim<F>,
    pub relu_subclaim: LookupSubClaim<F>,
}
