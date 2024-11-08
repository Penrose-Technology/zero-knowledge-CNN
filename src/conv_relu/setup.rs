use ark_ff::FftField;
use ark_std::{fs::File, io::BufReader};
use ark_poly::DenseMultilinearExtension;
use ark_sumcheck::rng::{Blake2b512Rng, FeedableRNG};
use serde::Deserialize;
use crate::conv_relu::data_structure::{GlobalParameters, 
                            ProverParameters, InputDate,
                            IPForConvRelu, ConvInfo, ReluInfo, TruncateInfo};
use crate::ml_lookup::data_structures::LookupTableInfo;
use crate::conv_relu::data_structure::VerifierParameters;
use ark_sumcheck::ml_sumcheck::data_structures::PolynomialInfo;

#[derive(Deserialize)]
pub struct GlobalParametersFromJson {
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
    pub group_length: usize,
}

impl <F: FftField> IPForConvRelu<F> {
    pub fn load_global_parameters() -> GlobalParameters<F> {
        let file = File::open("else/data/info.json").expect("Failed to open info.json!");
        let reader = BufReader::new(file);
        let gp_json: GlobalParametersFromJson = serde_json::from_reader(reader).expect("Failed to read info.json!");
        
        let mut fs_rng = Blake2b512Rng::setup();
        let lamda = F::rand(&mut fs_rng);
        let g = F::rand(&mut fs_rng);

        let public_table_column = Self::public_table_cloumn_init(lamda, &gp_json);
        let private_columns_flat_len = (gp_json.input_size.0 - gp_json.kernel_size.0 + 1)
                                             * (gp_json.input_size.0 - gp_json.kernel_size.0 + 1);
        let num_private_columns = (private_columns_flat_len + public_table_column.evaluations.len() - 1) 
                                            / public_table_column.evaluations.len();

        GlobalParameters{
            x_zero_point: gp_json.x_zero_point,
            x_scalar: gp_json.x_scalar,
            w_zero_point: gp_json.w_zero_point,
            w_scalar: gp_json.w_scalar,
            r_scalar: gp_json.r_scalar,
            bitwidth: gp_json.bitwidth,
            lookup_ceil: gp_json.lookup_ceil,
            lookup_floor: gp_json.lookup_floor,
            int_scalar: gp_json.int_scalar,
            kernel_size: gp_json.kernel_size,
            input_size: gp_json.input_size,
            group_length: gp_json.group_length,
            convolution_check_point: g,
            relu_combination_point: lamda,
            num_private_columns,
            public_table_column,
        }

    }

    pub fn load_prover_parameters() -> ProverParameters<F> {
        ProverParameters::<F> {
            x_i_fp: Vec::new(),
            w_i_fp: Vec::new(),
            y_i_fp: Vec::new(),
            r_i_fp: Vec::new(),
        }
    }

    pub fn load_input() -> InputDate {
        let file = File::open("else/data/input_data.json").expect("Failed to open input_data.json!");
        let reader = BufReader::new(file);
        let input: InputDate = serde_json::from_reader(reader).expect("Failed to read input_data.json!");

        input
    }

    pub fn load_verifier_parameters(gp: &GlobalParameters<F>) -> VerifierParameters<F> {
        let y_i_size_0 = gp.input_size.0 - gp.kernel_size.0 + 1;
        let num_vars = (y_i_size_0 * y_i_size_0).next_power_of_two().trailing_zeros() as usize;
        let truncate_info = TruncateInfo{
            poly_info: PolynomialInfo{
                max_multiplicands: 1usize,
                num_variables: num_vars, 
            },
        };
        let relu_info = ReluInfo{
            table_info: LookupTableInfo{
                public_column: gp.public_table_column.clone(),
                num_private_columns: gp.num_private_columns,
                group_length: gp.group_length,
                num_variables: gp.public_table_column.num_vars,
                num_groups: (gp.num_private_columns + 1) / gp.group_length,
            },
        };

        let mut fs_rng = Blake2b512Rng::setup();
        fs_rng.feed(&gp.convolution_check_point).unwrap();
        let mut g = Vec::new();
        for _ in 0..(gp.input_size.0 * gp.input_size.1).trailing_zeros() {
            g.push(F::rand(&mut fs_rng));
            fs_rng.feed(&g).unwrap();
        }
        let conv_info = ConvInfo{
            g,
        };
        VerifierParameters {
            conv_info,
            truncate_info,
            relu_info,
        }
    }

    pub fn public_table_cloumn_init(
        lamda: F,
        gp: &GlobalParametersFromJson,
    ) -> DenseMultilinearExtension<F> {
        let mut t = Vec::new();
        let p_1 = F::from(-1);
        let p_1_trun = Self::right_shift(p_1, gp.int_scalar);

        for i in 0..((1 << gp.bitwidth) * gp.lookup_ceil) {
            t.push(F::from((1 << gp.bitwidth) + i) + lamda * F::from((1 << gp.bitwidth) - 1));
        }
        for i in 0..((1 << gp.bitwidth) * gp.lookup_floor) {
            t.push(p_1_trun - F::from(i) + lamda * F::from(0));
        }
        for i in 0..(1 << gp.bitwidth) {
            t.push(F::from(i) + lamda * F::from(i));
        }
        let min_len = t.len().next_power_of_two();
        let min_len_log = min_len.trailing_zeros() as usize;
        t.extend(vec![F::from(0); min_len - t.len()]);

        DenseMultilinearExtension::from_evaluations_vec(min_len_log, t)
    
    }

}