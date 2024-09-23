use ark_ff::Field;
use ark_std::marker::PhantomData;
use ark_std::rand::RngCore;

pub mod prover;
pub mod verifier;
use crate::ml_lookup::data_structures::LookupTableInfo;
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};

pub struct IPForLookupTable<F: Field> {
    #[doc(hidden)]
    _marker: PhantomData<F>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug)]
pub struct RandomChallenge<F: Field> {
    pub poly_info: LookupTableInfo<F>,
    pub z: Vec<F>,
    pub beta: F,
    pub lamda: F,
}

impl <F: Field> IPForLookupTable<F> {
    pub fn sample_round<R: RngCore>(rng: &mut R) -> F {
        F::rand(rng)
    }

}
