use ark_ff::Field;
use ark_std::marker::PhantomData;
use ark_std::rand::RngCore;

pub mod prover;
pub mod verifier;
pub use crate::ml_lookup::data_structures::LookupTable;
/// 
pub struct IPForLookupTable<F: Field> {
    #[doc(hidden)]
    _marker: PhantomData<F>,
}

impl <F: Field> IPForLookupTable<F> {
    pub fn sample_round<R: RngCore>(rng: &mut R) -> F {
        F::rand(rng)
    }

}
