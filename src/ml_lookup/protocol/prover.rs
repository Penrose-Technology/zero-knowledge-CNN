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

pub struct ProverState<F: Field> {
    /// sampled univariate randomness beta given by the verifier
    pub beta: F,
    /// lamda, used for h identity check
    pub lamda: F,
    /// z, used for h identity check
    pub z: Vec<F>,
    /// list of m
    /// m_0 = m, m_i = -1
    pub m: Vec<DenseMultilinearExtension<F>>,
    /// lagrange selsctor
    pub lang: Rc<DenseMultilinearExtension<F>>,
    /// list of phi
    pub phi: Vec<Rc<DenseMultilinearExtension<F>>>,
    /// list of phi_inv
    pub phi_inv: Vec<DenseMultilinearExtension<F>>,
    /// list of h
    pub h :Vec<DenseMultilinearExtension<F>>,
    /// q
    pub q: DenseMultilinearExtension<F>,
}

impl<F: Field> ProverState<F> {
    pub fn init() -> Self {
        Self {
            beta: F::zero(),
            lamda: F::zero(),
            z: Vec::new(),
            m: Vec::new(),
            lang: Rc::new(DenseMultilinearExtension::zero()),
            phi: Vec::new(),
            phi_inv: Vec::new(),
            h: Vec::new(),
            q: DenseMultilinearExtension::zero(),
        }
    }

}

pub struct ProverMsg<F: Field> {
    pub m_r: F,
    pub f_r: Vec<F>,
    pub phi_inv_r: Vec<F>,
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

    pub fn phi_evaluation(table: &LookupTable<F>, prover_st: &mut ProverState<F>) {
        for k in 0..table.num_groups {
            for i in 0..table.group_length {
                let f_i;
                if i==0 && k==0 {
                    f_i = table.public_column.clone();                   
                }
                else {
                    f_i = table.private_columns[i + k * table.group_length - 1].clone();
                }
                let phi = DenseMultilinearExtension::from_evaluations_vec(
                    table.num_variables, 
                    vec![prover_st.beta; 1 << table.num_variables]) + f_i;

                let phi_inv: Vec<F> = phi.evaluations.iter()
                                                     .map(|point| point.inverse().unwrap())
                                                     .collect();

                prover_st.phi.push(Rc::new(phi));
                prover_st.phi_inv.push(DenseMultilinearExtension::from_evaluations_vec(table.num_variables, phi_inv));
            }
        }
    }

    pub fn h_evaluation(table: &LookupTable<F>, prover_st: &mut ProverState<F>) {
        for k in 0..table.num_groups {
            let mut item = DenseMultilinearExtension::zero();
            for i in 0..table.group_length {
                let m_i;
                if i==0 && k==0 {
                    m_i = prover_st.m[0].evaluations.clone();                   
           
                }
                else {
                    m_i = prover_st.m[1].evaluations.clone();                   
                }

                let m_phi_inv = prover_st.phi_inv[i + k * table.group_length].evaluations.iter()
                                                                                        .zip(m_i.into_iter())
                                                                                        .map(|(point, multiplicand)| *point * multiplicand)
                                                                                        .collect();

                let step = DenseMultilinearExtension::from_evaluations_vec(table.num_variables, m_phi_inv);
                item += step;
            }
            prover_st.h.push(item);
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
        prover_st.lang = Rc::new(DenseMultilinearExtension::from_evaluations_vec(table.num_variables, lang_vec));   
    }

    pub fn q_evaluation(table: &LookupTable<F>, prover_st: &mut ProverState<F>) {
        if prover_st.h.len() != table.num_groups
        {
            panic!("illegal h length!");
        }

        let q = DenseMultilinearExtension::zero();
        prover_st.q = prover_st.h.iter().fold(q, |acc, step| acc + step.clone());
    }


    pub fn load_ploy(table: &LookupTable<F>, prover_st: &ProverState<F>) 
                    ->  ListOfProductsOfPolynomials<F>
    {
        let mut poly = ListOfProductsOfPolynomials::new(table.num_variables);
        let mut lamda = prover_st.lamda;

        // load q(x) = sum_{k} h_k(x)
        let product = vec![Rc::new(prover_st.q.clone())];
        poly.add_product(product.into_iter(), F::one());

        // load h_1(x) identity
        // part 1, m_i , i \in I_1\{0}
        let mut product = vec![Rc::clone(&prover_st.lang)];
        let mut sum_phi_i1_inv = DenseMultilinearExtension::zero();
        for i in 1..table.group_length {
            sum_phi_i1_inv += &prover_st.phi_inv[i];
        }
        let h_1_plus_sum_phi_i1_inv = &prover_st.h[0] + &sum_phi_i1_inv;
        product.push(Rc::new(h_1_plus_sum_phi_i1_inv));
        for i in 0..table.group_length {
            product.push(Rc::clone(&prover_st.phi[i]));
        }
        poly.add_product(product.into_iter(), F::from(lamda));
        // part 2, m_0
        let mut product = vec![Rc::clone(&prover_st.lang)];
        product.push(Rc::new(prover_st.m[0].clone()));
        for i in 1..table.group_length {
            product.push(Rc::clone(&prover_st.phi[i]));
        }
        poly.add_product(product.into_iter(), F::from(- lamda));

        // load h_i(x) identity
        for k in 1..table.num_groups {
            lamda *= prover_st.lamda;
            let mut product = vec![Rc::clone(&prover_st.lang)];
            let mut sum_phi_i_inv = DenseMultilinearExtension::zero();
            for i in 0..table.group_length {
                sum_phi_i_inv += &prover_st.phi_inv[i + k * table.group_length];
            }
            let h_i_plus_sum_phi_i_inv = &prover_st.h[k] + &sum_phi_i_inv;
            product.push(Rc::new(h_i_plus_sum_phi_i_inv));
            for i in 0..table.group_length {
                product.push(Rc::clone(&prover_st.phi[i + k * table.group_length]));
            }
            poly.add_product(product.into_iter(), F::from(lamda));      
        }

        poly
    }

}


