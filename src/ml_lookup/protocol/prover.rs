use ark_ff::{Field, Zero};
use ark_poly::DenseMultilinearExtension;
use hashbrown::HashMap;
use ark_std::rc::Rc;
use ark_sumcheck::ml_sumcheck::{Proof, 
                                data_structures::ListOfProductsOfPolynomials, 
                                protocol::PolynomialInfo,
                            };
use crate::ml_lookup::{data_structures::LookupTable,
                       IPForLookupTable,
                    };

#[cfg(feature = "honest_prover")]
pub struct ProverState<F: Field> {
    /// sampled univariate randomness given by the verifier
    pub univariate_randomness: F,
    /// sampled multivariate randomness given by the verifier
    pub multivariate_randomness: Vec<F>,
    /// list of m
    /// m_0 = m, m_i = -1
    pub m: Vec<DenseMultilinearExtension<F>>,
    /// list of h
    pub h :Vec<DenseMultilinearExtension<F>>,
    /// q
    pub q: DenseMultilinearExtension<F>,
}

#[cfg(not(feature = "honest_prover"))]
pub struct ProverState<F: Field> {
    /// sampled univariate randomness given by the verifier
    pub univariate_randomness: F,
    /// sampled multivariate randomness given by the verifier
    pub multivariate_randomness: Vec<F>,
    /// lamda, used for h identity check
    pub lamda: F,
    /// z, used for h identity check
    pub z: Vec<F>,
    /// list of m
    /// m_0 = m, m_i = -1
    pub m: Vec<DenseMultilinearExtension<F>>,
    /// lagrange selsctor
    pub lang: DenseMultilinearExtension<F>,
    /// list of phi
    pub phi: Vec<DenseMultilinearExtension<F>>,
    /// list of phi group production
    pub phi_prod: Vec<DenseMultilinearExtension<F>>,
    /// list of h
    pub h :Vec<DenseMultilinearExtension<F>>,
    /// list of identity h
    pub h_identity: Vec<DenseMultilinearExtension<F>>,
    /// q
    pub q: DenseMultilinearExtension<F>,
}

impl<F: Field> ProverState<F> {
    #[cfg(feature = "honest_prover")]
    pub fn init() -> Self {
        Self {
            univariate_randomness: F::zero(),
            multivariate_randomness: Vec::new(),
            m: Vec::new(),
            h: Vec::new(),
            q: DenseMultilinearExtension::zero(),
        }
    }

    #[cfg(not(feature = "honest_prover"))]
    pub fn init() -> Self {
        Self {
            univariate_randomness: F::zero(),
            multivariate_randomness: Vec::new(),
            lamda: F::zero(),
            z: Vec::new(),
            m: Vec::new(),
            lang: DenseMultilinearExtension::zero(),
            phi: Vec::new(),
            phi_prod: Vec::new(),
            h: Vec::new(),
            h_identity: Vec::new(),
            q: DenseMultilinearExtension::zero(),
        }
    }

}

pub struct ProverMsg<F: Field> {
    pub m_r: F,
    pub f_r: Vec<F>,
    pub h_r: Vec<F>,
    pub q_info: PolynomialInfo,
    pub sub_proof: Proof<F>,
}

impl<F: Field> IPForLookupTable<F> {
    pub fn m_evaluation(table: &LookupTable<F>, prover_st: &mut ProverState<F>) {
        if table.max_private_columns < table.private_columns.len() {
            panic!("Exceed max column limitation");
        }
        let mut m = Vec::<DenseMultilinearExtension<F>>::with_capacity(2);

        let mut common_elements: HashMap<F, (F, F)> = table.public_column.evaluations
                                                            .clone()
                                                            .into_iter()
                                                            .map(|key| (key, (F::zero(), F::zero())))
                                                            .collect();

        let public_column_unique: Vec<F> = common_elements.iter().map(|(&k, _)| {k}).collect();

        for item in &public_column_unique {
            let repeat_num = F::from(table.public_column.evaluations.iter().filter(|&x| {*x == *item}).count() as u64);
            if let Some((_, t_repeat_num)) = common_elements.get_mut(item) {
                *t_repeat_num = repeat_num;
            }

            for column in &table.private_columns {
                let count = F::from(column.iter().filter(|&x| {*x == *item}).count() as u64);
                if let Some((value, _)) = common_elements.get_mut(item) {
                    *value += count;                       
                }
            }
        }

        let m_new_poly: Vec<F> = table.public_column.evaluations.iter()
                                                                .map(|&item| {
                                                                        let (numerator, denominator) = *common_elements.get(&item).unwrap();
                                                                        numerator * denominator.inverse().unwrap()
                                                                    })
                                                                .collect();
        let m_0 = DenseMultilinearExtension::from_evaluations_vec(table.num_variables, m_new_poly);
        let m_i = DenseMultilinearExtension::from_evaluations_vec(table.num_variables, vec![F::from(-1); 1 << table.num_variables]);
        m.push(m_0);
        m.push(m_i);

        prover_st.m = m;
    }

    #[cfg(feature = "honest_prover")]
    pub fn h_evaluation(table: &LookupTable<F>, prover_st: &mut ProverState<F>) {
        let num_polys = table.private_columns.len() + 1;
        let num_groups =  (num_polys + table.group_length - 1) / table.group_length;

        for k in 0..num_groups {
            let mut item = DenseMultilinearExtension::zero();
            for i in 0..table.group_length {
                let f_i;
                let m_i;
                if i==0 && k==0 {
                    f_i = table.public_column.clone();        
                    m_i = prover_st.m[0].evaluations.clone();                   
           
                }
                else {
                    f_i = table.private_columns[i + k * table.group_length - 1].clone();
                    m_i = prover_st.m[1].evaluations.clone();                   
                }

                let phi = DenseMultilinearExtension::from_evaluations_vec(
                    table.num_variables, 
                    vec![prover_st.univariate_randomness; 1 << table.num_variables]) + f_i;

                let m_phi_inv_i: Vec<F> = phi.evaluations.iter()
                    .zip(m_i.into_iter())
                    .map(|(point, multiplicand)| point.inverse().unwrap() * multiplicand)
                    .collect();

                let step = DenseMultilinearExtension::from_evaluations_vec(table.num_variables, m_phi_inv_i);
                item += step;
            }
            prover_st.h.push(item);
        }

    }

    #[cfg(not(feature = "honest_prover"))]
    pub fn h_evaluation(table: &LookupTable<F>, prover_st: &mut ProverState<F>) {
        let num_polys = table.private_columns.len() + 1;
        let num_groups =  (num_polys + table.group_length - 1) / table.group_length;

        for k in 0..num_groups {
            let mut phi_product_temp = vec![F::one(); 1 << table.num_variables];
            for i in 0..table.group_length {
                let rhs;
                if i==0 && k==0 {
                    rhs = table.public_column.clone();                   
                }
                else {
                    rhs = table.private_columns[i + k * table.group_length - 1].clone();
                }
                let phi = DenseMultilinearExtension::from_evaluations_vec(
                    table.num_variables, 
                    vec![prover_st.univariate_randomness; 1 << table.num_variables]) + rhs;

                phi_product_temp = phi_product_temp.iter()
                                             .zip(phi.evaluations.iter())
                                             .map(|(prod, nxt)| *prod * nxt)
                                             .collect();
                prover_st.phi.push(phi);
            }
            prover_st.phi_prod.push(DenseMultilinearExtension::from_evaluations_vec(table.num_variables, phi_product_temp));
        }

        for k in 0..num_groups {
            let mut item = DenseMultilinearExtension::zero();
            let mut identity_rhs = DenseMultilinearExtension::zero();                                          
            for i in 0..table.group_length {
                let rhs;
                if i==0 && k==0 {
                    rhs = prover_st.m[0].evaluations.clone();                   
                }
                else {
                    rhs = prover_st.m[1].evaluations.clone();
                }

                let m_phi_inv_i: Vec<F> = prover_st.phi[i + k * table.group_length].evaluations.iter()
                                                                            .zip(rhs.into_iter())
                                                                            .map(|(point, multiplicand)| point.inverse().unwrap() * multiplicand)
                                                                            .collect();

                let step = DenseMultilinearExtension::from_evaluations_vec(table.num_variables, m_phi_inv_i.clone());
                item += step;

                let m_phi_inv_prod_i = prover_st.phi_prod[k].evaluations.iter()
                                                                    .zip(m_phi_inv_i.into_iter())
                                                                    .map(|(prod, x)| *prod * x)
                                                                    .collect();
                let prod_step = DenseMultilinearExtension::from_evaluations_vec(table.num_variables, m_phi_inv_prod_i);
                identity_rhs += prod_step;
            }
            let m_phi_prod = prover_st.phi_prod[k].evaluations.iter()
                                                        .zip(item.evaluations.iter())
                                                        .map(|(h, phi_prod)| *h * *phi_prod)
                                                        .collect();
            let identity_lhs = DenseMultilinearExtension::from_evaluations_vec(table.num_variables, m_phi_prod);
            assert_eq!(identity_lhs, identity_rhs);

            prover_st.h.push(item);

            prover_st.h_identity.push(identity_lhs - identity_rhs);
        }

    }

    pub fn lagrange_selector_evaluation(table: &LookupTable<F>, prover_st: &mut ProverState<F>) {
        if prover_st.z.len() != table.num_variables {
            panic!("illegal lagrange random challange!")
        }

        let mut lang_vec = vec![F::zero(); 1 << table.num_variables];
        for idx in 0..(1 << table.num_variables) as u32 {
            let index: Vec<u32> = format!("{:0width$b}", idx, width=table.num_variables)
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
                                .zip(prover_st.z.iter())
                                .map(|(x, z)| F::from(2*x) * *z - F::from(x) - *z + F::one())
                                .collect::<Vec<F>>()
                                .into_iter()
                                .fold(F::one(), 
                                      |prod, i| prod * i
                                    );

            lang_vec[index_rev] = value;
        }

        prover_st.lang = DenseMultilinearExtension::from_evaluations_vec(table.num_variables, lang_vec);   
    }

    #[cfg(feature = "honest_prover")]
    pub fn q_evaluation(table: &LookupTable<F>, prover_st: &mut ProverState<F>) {
        if prover_st.h.len() != (table.private_columns.len() + table.group_length) / table.group_length 
        {
            panic!("illegal h length!");
        }

        let q = DenseMultilinearExtension::zero();
        prover_st.q = prover_st.h.iter().fold(q, |acc, step| acc + step.clone());
    }

    #[cfg(not(feature = "honest_prover"))]
    pub fn q_evaluation(table: &LookupTable<F>, prover_st: &mut ProverState<F>) {
        if prover_st.h.len() != prover_st.h_identity.len() {
            panic!("illegal h length!");
        }

        let mut lamda = F::one();
        let mut q = DenseMultilinearExtension::zero();

        for k in 0..prover_st.h.len() {
            lamda *= prover_st.univariate_randomness;
            prover_st.lang *= lamda;
            let item: Vec<F> = prover_st.lang.evaluations.iter()
                                                  .zip(prover_st.h_identity[k].evaluations.iter())
                                                  .map(|(x, y)| *x * *y)
                                                  .collect();
            let mut q_i = DenseMultilinearExtension::from_evaluations_vec(table.num_variables, item);
            q_i += &prover_st.h[k];
            q += q_i;
        }

        prover_st.q = q;
    }

    pub fn q_sumcheck(table: &LookupTable<F>, prover_st: &ProverState<F>) -> ListOfProductsOfPolynomials<F> {
        let product = vec![Rc::new(prover_st.q.clone())];
        let mut q_poly = ListOfProductsOfPolynomials::new(table.num_variables);
        q_poly.add_product(product.into_iter(), F::one());

        q_poly
    }
}


