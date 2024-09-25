use ark_ff::{Field, Zero};
use ark_poly::{DenseMultilinearExtension, Polynomial};
use crate::ml_lookup::data_structures::LookupTableInfo;
use crate::ml_lookup::protocol::{IPForLookupTable, prover::ProverMsg};

pub struct VerifierState<F: Field> {
    /// sampled univariate randomness
    pub beta: F,
    /// sampled multivariate randomness
    pub r: Vec<F>,
    /// lamda, used for h identity check
    pub lamda: F,
    /// z, used for h identity check
    pub z: Vec<F>,
    /// h(r)
    pub h_r: Vec<F>,
    /// phi_i(r)
    pub phi_r: Vec<F>,
    /// phi_i_inv(r)
    pub phi_inv_r: Vec<F>,
    /// m(r)
    pub m_r: F,
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
            beta: F::zero(),
            r: Vec::new(),
            lamda: F::zero(),
            z: Vec::new(),
            h_r: Vec::new(),
            phi_r: Vec::new(),
            phi_inv_r: Vec::new(),
            m_r: F::zero(),
            lang_r: F::zero(),
            lang: DenseMultilinearExtension::zero(),
            q_r: F::zero(),
        }
    }
}


impl<F: Field> IPForLookupTable<F> {

    #[inline]
    pub fn h_r_evaluation(table_info: &LookupTableInfo<F>, verifier_st: &mut VerifierState<F>, prover_msg: &ProverMsg<F>) {
        assert_eq!(prover_msg.h_r.len(), table_info.num_groups, "Illegal h_r length!");

        verifier_st.h_r = prover_msg.h_r.clone();
    }

    pub fn phi_r_evaluation(table_info: &LookupTableInfo<F>, verifier_st: &mut VerifierState<F>, prover_msg: &ProverMsg<F>) {
        assert_eq!(prover_msg.f_r.len(), table_info.num_private_columns, "Illegal f_r length!");

        let t_r = table_info.public_column.evaluate(&verifier_st.r);
        let phi_0 = t_r + verifier_st.beta;
        verifier_st.phi_r.push(phi_0);

        let phi_i: Vec<F> = prover_msg.f_r.iter().map(|x| *x + verifier_st.beta ).collect();
        verifier_st.phi_r.extend(phi_i);

        verifier_st.phi_inv_r = prover_msg.phi_inv_r.clone();
        
    }

    #[inline]
    pub fn m_r_evaluation(verifier_st: &mut VerifierState<F>, prover_msg: &ProverMsg<F>) {
        verifier_st.m_r = prover_msg.m_r;                             
    }

    pub fn lagrange_selector_eva(table_info: &LookupTableInfo<F>, verifier_st: &mut VerifierState<F>) {
        assert_eq!(verifier_st.z.len(), table_info.num_variables, "Illegal lagrange random challange z!");

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
                                .zip(verifier_st.z.iter())
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

    #[inline]
    pub fn lang_r_evaluation(table_info: &LookupTableInfo<F>, verifier_st: &mut VerifierState<F>) {
        assert_eq!(verifier_st.r.len(), table_info.num_variables, "Illegal lagrange evaluation point r!");
        assert_ne!(verifier_st.lang, DenseMultilinearExtension::zero(), "Polynomial lang(z, *) hasn't been evaluated!");

        verifier_st.lang_r = verifier_st.lang.evaluate(&verifier_st.r);
    }

    pub fn q_r_evaluation(table_info: &LookupTableInfo<F>, verifier_st: &mut VerifierState<F>) {
        assert_ne!(verifier_st.phi_r.len(), 0, "Points phi(r) haven't been evaluated!");
        assert_ne!(verifier_st.phi_inv_r.len(), 0, "Points phi_inv(r) haven't been evaluated!");
        assert_ne!(verifier_st.lang_r, F::zero(), "Point lang(r) hasn't been evaluated!");
        assert_ne!(verifier_st.h_r.len(), 0, "Points h(r) haven't been evaluated!");

        let phi: Vec<Vec<F>> = verifier_st.phi_r.chunks(table_info.group_length)
                                                .map(|chunk| chunk.to_vec())
                                                .collect();
        let phi_inv: Vec<Vec<F>> = verifier_st.phi_inv_r.chunks(table_info.group_length)
                                                        .map(|chunk| chunk.to_vec())
                                                        .collect();
        let phi_prod_k_r: Vec<F> = phi.iter()
                                     .map(|group| group.iter().fold(F::one(), |prod, i| prod * *i))
                                     .collect();

        let mut h_identity_r = Vec::with_capacity(table_info.num_groups);
        let mut lamda = verifier_st.lamda;

        // h_1_identity(r)
        let mut sum_phi_i1_inv_r = F::zero();
        let mut prod_phi_i1_r = F::one();
        for i in 1..table_info.group_length {
            sum_phi_i1_inv_r += &phi_inv[0][i];
            prod_phi_i1_r *= &phi[0][i];
        }
        let h_1_plus_sum_phi_i1_inv_r = sum_phi_i1_inv_r + verifier_st.h_r[0];
        let h_iden_1_r = h_1_plus_sum_phi_i1_inv_r * phi_prod_k_r[0] - verifier_st.m_r * prod_phi_i1_r;
        let h_iden_1_r_z = h_iden_1_r * lamda * verifier_st.lang_r;
        h_identity_r.push(h_iden_1_r_z);

        // h_i_identity(r)
        for k in 1..table_info.num_groups {
            let mut sum_phi_i_inv_r = F::zero();
            for i in 0..table_info.group_length {
                sum_phi_i_inv_r += &phi_inv[k][i];
            }
            let h_i_plus_sum_phi_i_inv_r = sum_phi_i_inv_r + verifier_st.h_r[k];
            let h_iden_i_r = h_i_plus_sum_phi_i_inv_r * phi_prod_k_r[k];
            lamda *= verifier_st.lamda; 
            let h_iden_i_r_z = h_iden_i_r *lamda * verifier_st.lang_r;
            h_identity_r.push(h_iden_i_r_z);
        }

        let q_r: F = verifier_st.h_r.iter()
                                .zip(h_identity_r.into_iter())
                                .map(|(h_k, h_iden_k)| *h_k + h_iden_k)
                                .fold(F::zero(), |sum, q_k| sum + q_k);

        verifier_st.q_r = q_r;
    }

}