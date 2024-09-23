use ark_ff::Field;
use ark_poly::Polynomial;
use data_structures::LookupTableInfo;
use std::marker::PhantomData;
use crate::ml_lookup::data_structures::LookupTable;
use crate::ml_lookup::protocol::{IPForLookupTable, RandomChallenge};
use crate::ml_lookup::protocol::prover::{ProverState, ProverMsg};
use crate::ml_lookup::protocol::verifier::VerifierState;
use ark_sumcheck::rng::{Blake2b512Rng, FeedableRNG};
use ark_sumcheck::Error;
use ark_sumcheck::ml_sumcheck::MLSumcheck;


pub mod protocol;
pub mod data_structures;
#[cfg(test)]
pub mod test;

pub struct MLLookupTable<F: Field>(PhantomData<F>);

impl <F: Field> MLLookupTable<F> {
    pub fn prove(table: &LookupTable<F>) -> Result<ProverMsg<F>, Error> {

        let mut prover_random = RandomChallenge 
            {
                poly_info: table.info(),
                z: Vec::new(),
                beta: F::zero(),
                lamda: F::zero(),
            };

        let mut fs_rng = Blake2b512Rng::setup();
        fs_rng.feed(&prover_random)?;

        // calculate beta
        let beta = IPForLookupTable::sample_round(&mut fs_rng);
        // prover state inilization
        let mut prover_st = ProverState::init();
        prover_random.beta = beta;
        fs_rng.feed(&prover_random)?;
        prover_st.beta = beta;

        // m evaluation
        IPForLookupTable::m_evaluation(&table, &mut prover_st);
        // phi evaluation
        IPForLookupTable::phi_evaluation(&table, &mut prover_st);
        // h evaluation
        IPForLookupTable::h_evaluation(&table, &mut prover_st);
        // lagrange selector evaluation
        for _i in 0..table.num_variables {
            let z_i = IPForLookupTable::sample_round(&mut fs_rng);
            prover_random.z.push(z_i);
            fs_rng.feed(&prover_random)?;
        }
        prover_st.z = prover_random.z.clone();
        IPForLookupTable::lagrange_selector_evaluation(&table, &mut prover_st);
        // q evaluation
        IPForLookupTable::q_evaluation(&table, &mut prover_st);
        // load poly
        let lamda = IPForLookupTable::sample_round(&mut fs_rng);
        prover_random.lamda = lamda;
        fs_rng.feed(&prover_random)?;
        prover_st.lamda = lamda;
        let q_poly = IPForLookupTable::load_ploy(&table, &prover_st);
        
        // sub-sumcheck
        let (sub_proof, r) = MLSumcheck::prove_as_subprotocol(&mut fs_rng, &q_poly)
                                                                    .map(|x| (x.0, x.1.randomness))
                                                                    .expect("fail to sub prove");
        dbg!(&r, prover_st.q.evaluate(&r));
        Ok(
            ProverMsg {
                m_r: prover_st.m[0].evaluate(&r),
                f_r: table.private_columns.iter().map(|f_i| f_i.evaluate(&r)).collect(),
                phi_inv_r: prover_st.phi_inv.iter().map(|phi_inv_i| phi_inv_i.evaluate(&r)).collect(),
                h_r: prover_st.h.iter().map(|h_k| h_k.evaluate(&r)).collect(),
                q_info: q_poly.info(),
                sub_proof,
            }
        )
    }

    pub fn verifeir(table_info: &LookupTableInfo<F>, prover_msg: &ProverMsg<F>) -> Result<bool, Error> {

        let mut verifier_random = RandomChallenge 
            {
                poly_info: table_info.clone(),
                z: Vec::new(),
                beta: F::zero(),
                lamda: F::zero(),
            };

        let mut fs_rng = Blake2b512Rng::setup();
        fs_rng.feed(&verifier_random)?;

        // calculate beta
        let beta = IPForLookupTable::sample_round(&mut fs_rng);
        // verifier state inilization
        let mut verifier_st = VerifierState::init();
        verifier_random.beta = beta;
        fs_rng.feed(&verifier_random)?;
        verifier_st.beta = beta;

        // calculate random point z
        for _i in 0..table_info.num_variables {
            let z_i = IPForLookupTable::sample_round(&mut fs_rng);
            verifier_random.z.push(z_i);
            fs_rng.feed(&verifier_random)?;
        }
        verifier_st.z = verifier_random.z.clone();
        // L(z,*) evaluation
        IPForLookupTable::lagrange_selector_eva(table_info, &mut verifier_st);
        // calculate lamda
        let lamda = IPForLookupTable::sample_round(&mut fs_rng);
        verifier_random.lamda = lamda;
        fs_rng.feed(&verifier_random)?;
        verifier_st.lamda = lamda;

        let subclaim = MLSumcheck::verify_as_subprotocol(&mut fs_rng, &prover_msg.q_info,F::zero(), &prover_msg.sub_proof)
                                                                .expect("fail to sub verify");
        dbg!(subclaim.expected_evaluation);

        // get random point r
        verifier_st.r = subclaim.point;
        dbg!(&verifier_st.beta, &verifier_st.r);
        // h(r) evaluation
        IPForLookupTable::h_r_evaluation(table_info, &mut verifier_st, prover_msg);
        // phi(r) evaluation
        IPForLookupTable::phi_r_evaluation(table_info, &mut verifier_st, prover_msg);
        // m(r) evaluation
        IPForLookupTable::m_r_evaluation(&mut verifier_st, prover_msg);
        // L(z, r) evaluation
        IPForLookupTable::lang_r_evaluation(table_info, &mut verifier_st);
        // q(r) evaluation
        IPForLookupTable::q_r_evaluation(table_info, &mut verifier_st);

        dbg!(verifier_st.q_r, subclaim.expected_evaluation);
        Ok(
            verifier_st.q_r - subclaim.expected_evaluation == F::zero()
        )

    }
    
}

