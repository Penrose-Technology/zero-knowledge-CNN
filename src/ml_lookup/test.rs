use std::future::poll_fn;
use std::iter::Product;

use crate::ml_lookup::protocol:: {prover::ProverState, verifier::VerifierState, IPForLookupTable};
use crate::ml_lookup::data_structures::LookupTable;
use crate::ml_lookup::{MLLookupTable, ProverMsg};
use ark_ff::{Field, UniformRand, Zero};
use ark_poly::DenseMultilinearExtension;
use ark_poly::Polynomial;
use ark_std::test_rng;
use ark_sumcheck::ml_sumcheck::protocol::verifier;
use ark_sumcheck::ml_sumcheck::MLSumcheck;
use ark_test_curves::bls12_381::Fr;
use blake2::digest::consts::True;

fn initialization() -> LookupTable<Fr>{
    let public_column = DenseMultilinearExtension::from_evaluations_vec(3, vec![Fr::from(0), Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(4), Fr::from(5), Fr::from(0), Fr::from(0)]);
    let private_column_1 = DenseMultilinearExtension::from_evaluations_vec(3, vec![Fr::from(2), Fr::from(3), Fr::from(3), Fr::from(1), Fr::from(2), Fr::from(5), Fr::from(4), Fr::from(4)]);
    let private_column_2 = DenseMultilinearExtension::from_evaluations_vec(3, vec![Fr::from(5), Fr::from(2), Fr::from(2), Fr::from(2), Fr::from(0), Fr::from(0), Fr::from(0), Fr::from(0)]);
    let private_column_3 = DenseMultilinearExtension::from_evaluations_vec(3, vec![Fr::from(4), Fr::from(0), Fr::from(3), Fr::from(1), Fr::from(5), Fr::from(0), Fr::from(5), Fr::from(5)]);
    let private_column_4 = DenseMultilinearExtension::from_evaluations_vec(3, vec![Fr::from(1), Fr::from(0), Fr::from(0), Fr::from(0), Fr::from(0), Fr::from(0), Fr::from(0), Fr::from(0)]);
    let private_column_5 = DenseMultilinearExtension::from_evaluations_vec(3, vec![Fr::from(3), Fr::from(5), Fr::from(0), Fr::from(2), Fr::from(4), Fr::from(1), Fr::from(2), Fr::from(1)]);

    
    let mut table = LookupTable::<Fr>::new(3, 3);
    table.add_public_column(public_column);
    table.add_private_column(private_column_1);
    table.add_private_column(private_column_2);
    table.add_private_column(private_column_3);
    table.add_private_column(private_column_4);
    table.add_private_column(private_column_5);

    table
}

#[test]
#[cfg(feature = "honest_prover")]
fn prover_test() {
    let mut rng = test_rng();
    let table = initialization();
    let mut prover_st = ProverState::init();
    
    // test m
    IPForLookupTable::m_evaluation(&table, &mut prover_st);

    let zero_repeat = Fr::from(3).inverse().unwrap() * Fr::from(14);
    let reference = DenseMultilinearExtension::from_evaluations_vec(3, vec![zero_repeat, Fr::from(5), Fr::from(7), Fr::from(4), Fr::from(4), Fr::from(6), zero_repeat, zero_repeat]);
    assert_eq!(prover_st.m[0], reference);

    // test h summation is zero
    prover_st.univariate_randomness = Fr::rand(&mut rng);
    IPForLookupTable::h_evaluation(&table, &mut prover_st);

    let mut sigma_h = DenseMultilinearExtension::zero();
    for i in 0..prover_st.h.len() {
        sigma_h += &prover_st.h[i];
    }

    let mut sum = Fr::zero();
    for i in 0..sigma_h.evaluations.len() {
        sum += &sigma_h.evaluations[i];
        //println!("sum is {}", sum);
    }
    assert!(sum == Fr::zero(), "summation of h is not equal zero!");

    // test q
    prover_st.univariate_randomness = Fr::rand(&mut rng);
    IPForLookupTable::q_evaluation(&table, &mut prover_st);
    let mut sum = Fr::zero();
    for i in 0..(1 << table.num_variables) {
        sum += prover_st.q.evaluations[i];
    }
    assert!(sum == Fr::zero(), "summation of q is not equal zero!");
    assert!(prover_st.q == sigma_h, "q is error!");

    IPForLookupTable::q_sumcheck(&table, &prover_st);
    
}

#[test]
#[cfg(not(feature = "honest_prover"))]
fn prover_test() {
    let mut rng = test_rng();
    let table = initialization();
    let mut prover_st = ProverState::init();
    
    // test m
    IPForLookupTable::m_evaluation(&table, &mut prover_st);

    let zero_repeat = Fr::from(3).inverse().unwrap() * Fr::from(14);
    let reference = DenseMultilinearExtension::from_evaluations_vec(3, vec![zero_repeat, Fr::from(5), Fr::from(7), Fr::from(4), Fr::from(4), Fr::from(6), zero_repeat, zero_repeat]);
    assert_eq!(prover_st.m[0], reference);

    // test h summation is zero
    prover_st.univariate_randomness = Fr::rand(&mut rng);
    IPForLookupTable::h_evaluation(&table, &mut prover_st);

    let mut sigma_h = DenseMultilinearExtension::zero();
    for i in 0..prover_st.h.len() {
        sigma_h += &prover_st.h[i];
    }

    let mut sum = Fr::zero();
    for i in 0..sigma_h.evaluations.len() {
        sum += &sigma_h.evaluations[i];
        //println!("sum is {}", sum);
    }
    assert!(sum == Fr::zero(), "summation of h is not equal zero!");

    // test all h identity is zero
    for k in 0..prover_st.h_identity.len() {
        for i in 0..(1 << table.num_variables) {
            assert!(prover_st.h_identity[k].evaluations[i] == Fr::zero(),
                    "h_identity is not zero!");
        }
    }

    // test lagrange selector
    // should be one at boolean-hypercube, in reverse order.
    for x_0 in 0..=1 as u32 {
        for x_1 in 0..=1 {
            for x_2 in 0..=1 {
                prover_st.multivariate_randomness = vec![Fr::from(x_0), Fr::from(x_1), Fr::from(x_2)];
                let ref_index = vec![x_2, x_1, x_0].into_iter().fold(0, |acc , bit| (acc << 1) | bit);

                IPForLookupTable::lagrange_selector_evaluation(&table, &mut prover_st);
                let mut ref_lang_vec = vec![Fr::from(0); 1 << table.num_variables]; 
                ref_lang_vec[ref_index as usize] = Fr::from(1);

                assert_eq!(prover_st.lang.evaluations, ref_lang_vec);
            }
        }
    }
    prover_st.multivariate_randomness = vec![Fr::rand(&mut rng); 3];
    IPForLookupTable::lagrange_selector_evaluation(&table, &mut prover_st);

    // test q
    prover_st.univariate_randomness = Fr::rand(&mut rng);
    IPForLookupTable::q_evaluation(&table, &mut prover_st);
    let mut sum = Fr::zero();
    for i in 0..(1 << table.num_variables) {
        sum += prover_st.q.evaluations[i];
    }
    assert!(sum == Fr::zero(), "summation of q is not equal zero!");
    assert!(prover_st.q == sigma_h, "q is error!");

    IPForLookupTable::q_sumcheck(&table, &prover_st);
    
}

#[test]
#[cfg(not(feature = "honest_prover"))]
fn verifier_test() {
    let mut rng = test_rng();
    let table = initialization();
    let table_info = table.info();
    let mut verifier_st = VerifierState::init();
    let mut prover_st = ProverState::init();
    
    prover_st.univariate_randomness = Fr::from(10);
    prover_st.multivariate_randomness = vec![Fr::rand(&mut rng); 3];
    IPForLookupTable::m_evaluation(&table, &mut prover_st);
    IPForLookupTable::h_evaluation(&table, &mut prover_st);
    IPForLookupTable::lagrange_selector_evaluation(&table, &mut prover_st);
    IPForLookupTable::q_evaluation(&table, &mut prover_st);
    let q_poly = IPForLookupTable::q_sumcheck(&table, &prover_st);

    let r = vec![Fr::rand(&mut rng); table.num_variables];
    let q_r_ref = prover_st.q.evaluate(&r);
    let prover_msg = ProverMsg {
            m_r: prover_st.m[0].evaluate(&r),
            f_r: table.private_columns.iter().map(|f_i| f_i.evaluate(&r)).collect(),
            h_r: prover_st.h.iter().map(|h_k| h_k.evaluate(&r)).collect(),
            q_info: q_poly.info(),
            sub_proof: Vec::new(),
        };  

    verifier_st.univariate_randomness = prover_st.univariate_randomness;
    verifier_st.multivariate_randomness = r.clone();
    IPForLookupTable::h_r_evaluation(&table_info, &mut verifier_st, &prover_msg);
    IPForLookupTable::phi_r_evaluation(&table_info, &mut verifier_st, &prover_msg);
    IPForLookupTable::m_r_evaluation(&table_info, &mut verifier_st, &prover_msg);
    IPForLookupTable::lagrange_selector_eva(&table_info, &mut verifier_st);
    IPForLookupTable::lang_r_evaluation(&table_info, &mut verifier_st);
    //assert_eq!(verifier_st.lang_r, Fr::from(1));
    IPForLookupTable::q_r_evaluation(&table_info, &mut verifier_st);

    assert_eq!(verifier_st.q_r, q_r_ref);

}

#[test]
fn protocol_test() {
    let mut rng = test_rng();
    let table = initialization();

    let prover_msg = MLLookupTable::prove(&table).expect("test error! can not generate prover message!");

    let table_info = table.info();
    let result = MLLookupTable::verifeir(&table_info, &prover_msg).expect("test error! can not generate verifier message!");
    assert_eq!(result, true);
}
