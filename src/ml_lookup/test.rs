use crate::ml_lookup::protocol:: {prover::ProverState, verifier::VerifierState, IPForLookupTable};
use crate::ml_lookup::data_structures::LookupTable;
use crate::ml_lookup::{MLLookupTable, ProverMsg};
use ark_ff::{Field, UniformRand, Zero};
use ark_poly::DenseMultilinearExtension;
use ark_poly::Polynomial;
use ark_std::test_rng;
use ark_sumcheck::ml_sumcheck::MLSumcheck;
use ark_test_curves::bls12_381::Fr;
use ark_sumcheck::rng::{Blake2b512Rng, FeedableRNG};

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
    assert_eq!(table.num_groups, 1);
    table.add_private_column(private_column_2);
    assert_eq!(table.num_groups, 1);
    table.add_private_column(private_column_3);
    table.add_private_column(private_column_4);
    assert_eq!(table.num_groups, 2);
    table.add_private_column(private_column_5);
    assert_eq!(table.num_groups, 2);

    table
}


#[test]
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
    prover_st.beta = Fr::rand(&mut rng);
    IPForLookupTable::phi_evaluation(&table, &mut prover_st);
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
    assert_eq!(sum, Fr::zero(), "summation of h is not equal zero!");

    // test lagrange selector
    // should be one at boolean-hypercube, in reverse order.
    for x_0 in 0..=1 as u32 {
        for x_1 in 0..=1 {
            for x_2 in 0..=1 {
                prover_st.z = vec![Fr::from(x_0), Fr::from(x_1), Fr::from(x_2)];
                let ref_index = vec![x_2, x_1, x_0].into_iter().fold(0, |acc , bit| (acc << 1) | bit);

                IPForLookupTable::lagrange_selector_evaluation(&table, &mut prover_st);
                let mut ref_lang_vec = vec![Fr::from(0); 1 << table.num_variables]; 
                ref_lang_vec[ref_index as usize] = Fr::from(1);

                assert_eq!(prover_st.lang.evaluations, ref_lang_vec);
            }
        }
    }
    prover_st.z = vec![Fr::rand(&mut rng); 3];
    IPForLookupTable::lagrange_selector_evaluation(&table, &mut prover_st);

    // test q
    prover_st.beta = Fr::rand(&mut rng);
    IPForLookupTable::q_evaluation(&table, &mut prover_st);
    let mut sum = Fr::zero();
    for i in 0..(1 << table.num_variables) {
        sum += prover_st.q.evaluations[i];
    }
    assert_eq!(sum, Fr::zero(), "summation of q is not equal zero!");
    assert_eq!(prover_st.q, sigma_h, "q is error!");
}

#[test]
fn verifier_test() {
    let mut rng = Blake2b512Rng::setup();
    let table = initialization();
    let table_info = table.info();
    let mut verifier_st = VerifierState::init();
    let mut prover_st = ProverState::init();
    
    prover_st.beta = Fr::rand(&mut rng);
    prover_st.lamda = Fr::rand(&mut rng);
    prover_st.z = vec![Fr::rand(&mut rng); 3];
    IPForLookupTable::m_evaluation(&table, &mut prover_st);
    IPForLookupTable::phi_evaluation(&table, &mut prover_st);
    IPForLookupTable::h_evaluation(&table, &mut prover_st);
    IPForLookupTable::lagrange_selector_evaluation(&table, &mut prover_st);
    IPForLookupTable::q_evaluation(&table, &mut prover_st);
    let q_poly = IPForLookupTable::load_ploy(&table, &prover_st);

    let mut fs_rng = Blake2b512Rng::setup();
    let (sub_proof, r) = MLSumcheck::prove_as_subprotocol(&mut fs_rng, &q_poly)
                                                                            .map(|x| (x.0, x.1.randomness))
                                                                            .expect("fail to sub prove");

    let prover_msg = ProverMsg {
            m_r: prover_st.m[0].evaluate(&r),
            f_r: table.private_columns.iter().map(|f_i| f_i.evaluate(&r)).collect(),
            phi_inv_r: prover_st.phi_inv.iter().map(|phi_inv_k| phi_inv_k.evaluate(&r)).collect(),
            h_r: prover_st.h.iter().map(|h_k| h_k.evaluate(&r)).collect(),
            q_info: q_poly.info(),
            sub_proof,
        };  

    let mut fs_rng = Blake2b512Rng::setup();
    let subclaim = MLSumcheck::verify_as_subprotocol(&mut fs_rng, &prover_msg.q_info, Fr::zero(), &prover_msg.sub_proof)
                                                                                                .expect("fail to sub verify");
    verifier_st.beta = prover_st.beta;
    verifier_st.r = r.clone();
    verifier_st.z = prover_st.z.clone();
    verifier_st.lamda = prover_st.lamda;
        
    IPForLookupTable::h_r_evaluation(&table_info, &mut verifier_st, &prover_msg);
    assert_eq!(verifier_st.h_r, prover_msg.h_r);
    IPForLookupTable::phi_r_evaluation(&table_info, &mut verifier_st, &prover_msg);
    IPForLookupTable::m_r_evaluation(&mut verifier_st, &prover_msg);
    IPForLookupTable::lagrange_selector_eva(&table_info, &mut verifier_st);
    IPForLookupTable::lang_r_evaluation(&table_info, &mut verifier_st);
    assert_eq!(prover_st.lang.evaluate(&r), verifier_st.lang_r);

    IPForLookupTable::q_r_evaluation(&table_info, &mut verifier_st);

    assert_eq!(verifier_st.q_r, subclaim.expected_evaluation);

}

#[test]
fn protocol_test() {
    let table = initialization();

    let prover_msg = MLLookupTable::prove(&table).expect("test error! can not generate prover message!");

    let table_info = table.info();
    MLLookupTable::verify(&table_info, &prover_msg).expect("test error! can not generate verifier message!");
}
