use ark_ff::{Field, Zero};
use ark_poly::{DenseMultilinearExtension, Polynomial};
use ark_sumcheck::ml_sumcheck::protocol::verifier;
use ark_std::rc::Rc;
use ark_sumcheck::ml_sumcheck::{Proof, MLSumcheck, data_structures::ListOfProductsOfPolynomials};
use ark_sumcheck::Error;
use ark_sumcheck::rng::Blake2b512Rng;
use crate::ml_lookup::data_structures::LookupTableInfo;
use crate::ml_lookup::protocol::{IPForLookupTable, prover::ProverMsg};

pub struct VerifierState<F: Field> {
    /// sampled univariate randomness
    pub univariate_randomness: F,
    /// sampled multivariate randomness
    pub multivariate_randomness: Vec<F>,
    /// h(r)
    pub h_r: Vec<F>,
    /// phi_i(r)
    pub phi_r: Vec<F>,
    /// m_i(r)
    pub m_r: Vec<F>,
    /// L(z, r)
    pub lang_r: F,
    /// lagrange selector
    pub lang: DenseMultilinearExtension<F>,
    /// q(r)
    pub q_r: F,
}

impl<F: Field> VerifierState<F> {
    pub fn init() -> Self {
        Self {
            univariate_randomness: F::zero(),
            multivariate_randomness: Vec::new(),
            h_r: Vec::new(),
            phi_r: Vec::new(),
            m_r: Vec::new(),
            lang_r: F::zero(),
            lang: DenseMultilinearExtension::zero(),
            q_r: F::zero(),
        }
    }
}

pub struct VerifierMsg<F: Field>  {
    q_r: F,
}

impl<F: Field> IPForLookupTable<F> {
    pub fn h_r_evaluation(table_info: &LookupTableInfo<F>, verifier_st: &mut VerifierState<F>, prover_msg: &ProverMsg<F>) {
        if prover_msg.h_r.len() != table_info.num_groups {
            panic!("illegal h_r length!");
        }

        verifier_st.h_r = prover_msg.h_r.clone();
    }

    pub fn phi_r_evaluation(table_info: &LookupTableInfo<F>, verifier_st: &mut VerifierState<F>, prover_msg: &ProverMsg<F>) {
        let t_r = table_info.public_column.evaluate(&verifier_st.multivariate_randomness);
        let phi_0 = t_r + verifier_st.univariate_randomness;
        verifier_st.phi_r.push(phi_0);

        let phi_i: Vec<F> = prover_msg.f_r.iter().map(|x| *x + verifier_st.univariate_randomness ).collect();
        verifier_st.phi_r.extend(phi_i);
        
    }

    pub fn m_r_evaluation(table_info: &LookupTableInfo<F>, verifier_st: &mut VerifierState<F>, prover_msg: &ProverMsg<F>) {
        verifier_st.m_r.push(prover_msg.m_r);
        let m_i = DenseMultilinearExtension::from_evaluations_vec(table_info.num_variables, 
                                                                     vec![F::from(-1); 1 << table_info.num_variables]
                                                                    ).evaluate(&verifier_st.multivariate_randomness);
        verifier_st.m_r.extend(vec![m_i; table_info.num_private_columns]);                              
    }

    pub fn lagrange_selector_eva(table_info: &LookupTableInfo<F>, verifier_st: &mut VerifierState<F>) {
        if verifier_st.multivariate_randomness.len() != table_info.num_variables {
            panic!("illegal lagrange random challange!")
        }

        let mut lang_vec = vec![F::zero(); 1 << table_info.num_variables];
        for idx in 0..(1 << table_info.num_variables) as u32 {
            let index: Vec<u32> = format!("{:0width$b}", idx, width=table_info.num_variables)
                                            .chars()
                                            .map(|x| x.to_digit(2).unwrap())
                                            .collect();

            let mut index_rev = index.clone();
            index_rev.reverse();

            let index_rev = index_rev.iter()
                                    .fold(0, 
                                            |acc, &bit| (acc << 1) | bit as usize
                                        );
            
            let value: F = index.into_iter()
                                .zip(verifier_st.multivariate_randomness.iter())
                                .map(|(x, z)| F::from(2*x) * *z - F::from(x) - *z + F::one())
                                .collect::<Vec<F>>()
                                .into_iter()
                                .fold(F::one(), 
                                      |prod, i| prod * i
                                    );

            lang_vec[index_rev] = value;
        }

        verifier_st.lang = DenseMultilinearExtension::from_evaluations_vec(table_info.num_variables, lang_vec);   
    }

    pub fn lang_r_evaluation(table_info: &LookupTableInfo<F>, verifier_st: &mut VerifierState<F>) {
        if verifier_st.multivariate_randomness.len() != table_info.num_variables {
            panic!("illegal lagrange evaluation point!")
        }
        
        verifier_st.lang_r = verifier_st.lang.evaluate(&verifier_st.multivariate_randomness);
    }

    pub fn q_r_evaluation(table_info: &LookupTableInfo<F>, verifier_st: &mut VerifierState<F>) {

        let phi: Vec<Vec<F>> = verifier_st.phi_r.chunks(table_info.group_length)
                                                     .map(|chunk| chunk.to_vec())
                                                     .collect();
        dbg!(&phi);
        let phi_prod_k: Vec<F> = phi.iter()
                                     .map(|group| group.iter().fold(F::one(), |prod, i| prod * *i))
                                     .collect();

        let m: Vec<Vec<F>> = verifier_st.m_r.chunks(table_info.group_length)
                                              .map(|chunk| chunk.to_vec())
                                              .collect();
        dbg!(&m);
        let m_phi_k: Vec<F> = m.iter()
                                .zip(phi.iter())
                                .map(|(m_k, phi_k)| 
                                        m_k.iter().zip(phi_k.iter())
                                            .fold(F::zero(), |sum, (m_i, phi_i)| sum + *m_i * phi_i.inverse().unwrap())
                                    ).collect();

        let h_identity: Vec<F> = verifier_st.h_r.iter()
                                                .zip(phi_prod_k.into_iter())
                                                .zip(m_phi_k.into_iter())
                                                .map(|((h_k, prod_k), m_phi_inv_k)|
                                                        (*h_k - m_phi_inv_k) * prod_k
                                                    ).collect();
        dbg!(&h_identity);
        let mut lamda = F::one();
        let mut scalar = Vec::new();
        for _i in 0..h_identity.len() {
            lamda *= verifier_st.univariate_randomness;
            verifier_st.lang_r *= lamda;
            scalar.push(verifier_st.lang_r);
        }

        let q: F = verifier_st.h_r.iter()
                                .zip(h_identity.into_iter())
                                .zip(scalar.into_iter())
                                .map(|((h_k, h_iden_k), scalar_k)| *h_k + scalar_k * h_iden_k)
                                .fold(F::zero(), |sum, q_k| sum + q_k);

        //let q = verifier_st.h_r.iter().fold(F::zero(), |sum, q_k| sum + q_k);

        verifier_st.q_r = q;
    }

}