use ark_ff::FftField;
use ark_std::{rc::Rc, fs::File};
use ark_poly::{DenseMultilinearExtension, EvaluationDomain, GeneralEvaluationDomain};
use ark_sumcheck::ml_sumcheck::Proof as TruncateProof;
use ark_sumcheck::rng::{Blake2b512Rng, FeedableRNG};
use ark_sumcheck::ml_sumcheck::{data_structures::ListOfProductsOfPolynomials, MLSumcheck};
use serde_json::json;

use crate::convolution::{Conv, MLConvolution, fft::Convpolynomial, ConvProof};
use crate::ml_lookup::{data_structures::LookupTable, MLLookupTable};
use crate::ml_lookup::protocol::prover::ProverMsg as ReluProof;
use crate::conv_relu::data_structure::{GlobalParameters, 
                            ProverParameters, InputDate, OutputData,
                            IPForConvRelu};


impl <F: FftField> IPForConvRelu<F> {
    pub fn conv_prove(
        gp: &GlobalParameters<F>, 
        pp: &mut ProverParameters<F>,    
        input: &InputDate,    
    ) -> ConvProof<F> {
        assert_eq!(gp.input_size.0, gp.input_size.1, "Error input data shape!");
        assert_eq!(gp.kernel_size.0, gp.kernel_size.1, "Error kernel shape!");

        let x_zeropoint_field = F::from(gp.x_zero_point);
        let w_zeropoint_field = F::from(gp.w_zero_point);
        let m_int = (gp.x_scalar * gp.w_scalar / gp.r_scalar) * 
                         ((1 << (gp.int_scalar << 3)) as f64);
        let m_field = F::from(m_int as u32);
        
        // stride should be 1, input data size should be the power of 2
        // reverse kernel data and pad zero
        let mut w_i = vec![vec![F::from(0); gp.input_size.0]; gp.input_size.1];
        for (i, row) in input.w_i.iter().enumerate() {
            for (j, &item) in row.iter().enumerate() {
                w_i[gp.kernel_size.0 - 1 - i][gp.kernel_size.1 - 1 - j] = F::from(item) - w_zeropoint_field;
            }
        } 
        // flat input and kernel data
        let x_i = input.x_i.concat();
        let w_i = w_i.concat();

        // convert data into Fftield
        pp.x_i_fp = x_i.into_iter().map(|x| (F::from(x) - x_zeropoint_field) * m_field
                                        ).collect::<Vec<F>>();        
        pp.w_i_fp = w_i;  
        
        // calculate M(q_x - z_x)*(q_w - z_w)
        let domain = GeneralEvaluationDomain::<F>::new(gp.input_size.0 * gp.input_size.1).unwrap();
        let mut x_poly = Convpolynomial::new(pp.x_i_fp.clone());
        let mut w_poly = Convpolynomial::new(pp.w_i_fp.clone());
        let mut y_poly_raw = MLConvolution::evaluate(&domain, &mut x_poly, &mut w_poly);

        // for convolution, y_poly should be the untrimmed polynomial, which is y_poly_raw.
        let mut fs_rng = Blake2b512Rng::setup();
        fs_rng.feed(&gp.convolution_check_point).unwrap();
        let mut g = Vec::new();
        for _ in 0..(gp.input_size.0 * gp.input_size.1).trailing_zeros() {
            g.push(F::rand(&mut fs_rng));
            fs_rng.feed(&g).unwrap();
        }

        let proof = MLConvolution::prove(&domain, &g, &mut x_poly, &mut w_poly, &mut y_poly_raw);
        MLConvolution::verify(&g, &domain, &proof).unwrap();
        pp.y_i_fp = y_poly_raw.poly_after_fft_ifft;

        proof  
    }

    pub fn truncate_prove(
        gp: &GlobalParameters<F>, 
        pp: &mut ProverParameters<F>,          
    ) -> TruncateProof<F>{
        // truncation, right shift int_scalar bytes.
        // this operation separate y_q into y_q = y_q_ho << (int_scalar bytes) + y_q_lo.
        let (y_hi,y_lo):(Vec<F>, Vec<F>) = 
                                    pp.y_i_fp.iter()
                                    .map(|&x| {
                                            let mut buf: Vec<u8> = Vec::new();
                                            x.serialize_uncompressed(&mut buf).unwrap();
                                            let field_len = buf.len();
                                            let mut buf_hi = buf.split_off(gp.int_scalar);
                                            buf_hi.extend(vec![0; gp.int_scalar]);
                                            buf.resize(field_len, 0);
                                    
                                            (F::deserialize_uncompressed(&buf_hi[..]).unwrap(),
                                            F::deserialize_uncompressed(&buf[..]).unwrap())
                                        }
                                    ).unzip();

        let y_sum = pp.y_i_fp.clone();
        pp.y_i_fp = y_hi.clone();

        // use a single sumcheck to prove y_q_ho << (int_scalar bytes) + y_q_lo - y_q = 0
        let num_vars = pp.y_i_fp.len().next_power_of_two().trailing_zeros() as usize;
        let mut poly = ListOfProductsOfPolynomials::new(num_vars);
        let y_hi_poly = DenseMultilinearExtension::from_evaluations_vec(num_vars, y_hi);
        let y_lo_poly = DenseMultilinearExtension::from_evaluations_vec(num_vars, y_lo);
        let y_sum_poly = DenseMultilinearExtension::from_evaluations_vec(num_vars, y_sum);
        poly.add_product(vec![Rc::new(y_hi_poly)], F::from(1 << (gp.int_scalar << 3)));
        poly.add_product(vec![Rc::new(y_lo_poly)], F::from(1));
        poly.add_product(vec![Rc::new(y_sum_poly)], F::from(-1));

        let proof = MLSumcheck::prove(&poly).expect("truncation sumcheck prove failed!");

        proof
    }

    pub fn relu_prove(
        gp: &GlobalParameters<F>, 
        pp: &mut ProverParameters<F>,   
    ) -> ReluProof<F> {
        let p_1 = F::from(-1);
        let p_1_trun = Self::right_shift(p_1, gp.int_scalar);

        // after quantization, relu is the same as cliping, 
        // clip all y_q_ho into [0, 256).
        for value in pp.y_i_fp.iter() {
            if (*value > F::from((1 << gp.bitwidth) - 1)) 
                && (*value < F::from((1 << gp.bitwidth) * gp.lookup_ceil))  {
                pp.r_i_fp.push(F::from((1 << gp.bitwidth) - 1));
            }
            else if *value > p_1_trun - F::from((1 << gp.bitwidth) * gp.lookup_floor)  {
                pp.r_i_fp.push(F::from(0));
            }
            else if (*value >= F::from((1 << gp.bitwidth) * gp.lookup_ceil)) 
                && (*value <= p_1_trun - F::from((1 << gp.bitwidth) * gp.lookup_floor)) {
                panic!("Try pushing illegal item into the table!!")
            }
            else {
                pp.r_i_fp.push(*value);
            }
        }

        // use lookup table to prove relu operation.
        let relu_table = Self::table_init(&gp, &pp);
        let proof = MLLookupTable::prove(&relu_table).expect("relu lookup table prove failed!");
    
        proof
    }


    pub fn right_shift(x: F, bytes_num: usize) -> F {
        let mut buf: Vec<u8> = Vec::new();
        x.serialize_uncompressed(&mut buf).unwrap();
        buf.drain(0..bytes_num);
        buf.extend(vec![0; bytes_num]);

        F::deserialize_uncompressed(&buf[..]).unwrap()
    }


    fn table_init(
        gp: &GlobalParameters<F>,
        pp: &ProverParameters<F>,
    ) -> LookupTable<F>{
        // table items are y_q_hi*lamda + r_q, r_q is the clipped data of y_q_hi.
        assert_eq!(pp.y_i_fp.len(), pp.r_i_fp.len(), "y_i_fp and r_i_fp size mismatch!");
        let min_len = gp.public_table_column.evaluations.len();
        let min_len_log = min_len.trailing_zeros() as usize;

        // divide data into private columns, whose size should be same as min_len.
        let mut f =pp.y_i_fp.chunks(min_len).enumerate()
                                        .map(|(i, x)| 
                                                x.iter().enumerate()
                                                .map(|(j, value)| 
                                                        *value + gp.relu_combination_point * pp.r_i_fp[i * min_len + j]
                                                    ).collect::<Vec<F>>()
                                            ).collect::<Vec<Vec<F>>>();

        // add public column
        let mut table = LookupTable::<F>::new(min_len_log, gp.group_length);
        table.add_public_column(gp.public_table_column.clone());

        // add all privates columns
        while f.len() != 0 {
            if let Some(f_i) = f.pop() {
                let mut f_i = f_i;
                if f_i.len() < min_len {
                    f_i.extend(vec![F::from(0); min_len - f_i.len()]);
                }
                let private_column = DenseMultilinearExtension::from_evaluations_vec(
                                                                                            min_len_log, 
                                                                                            f_i);
                table.add_private_column(private_column);
            }
        }

        table
    }

    
    pub fn trim(
        gp: &GlobalParameters<F>, 
        y_raw: &Vec<F>,        
    ) -> Vec<F> {
        let start = (gp.kernel_size.0 - 1) * (gp.input_size.0 + 1);
        let mut y_i_remove = (0..start).collect::<Vec<usize>>();
        for i in 0..gp.input_size.0 {
            for j in 1..gp.kernel_size.0 {
                y_i_remove.push(start + (i + 1) * gp.input_size.0 -  j);
            }
        }
        
        let y = y_raw.clone()
                            .into_iter().enumerate()
                            .filter(|(i, _)| !y_i_remove.contains(i))
                            .map(|(_, value)| value).collect::<Vec<F>>();
        y
    }

    
    pub fn dump_output(
        gp: &GlobalParameters<F>,
        pp: &ProverParameters<F>,
    ) -> OutputData<F> {
        let r_i_raw = pp.r_i_fp.clone();
        let r_i_fp_trim = Self::trim(&gp, &pp.r_i_fp);
        let r_i_ref = r_i_fp_trim.iter()
                                            .map(|x| {
                                                let mut r_int: Vec<u8> = Vec::new();
                                                x.serialize_uncompressed(&mut r_int).unwrap();
                                                let zero_flag = r_int[1..].iter().all(|&i| i == 0);
                                                assert_eq!(zero_flag, true, "Overflow! Can't convert <F> into <u8>!");
                                                r_int[0]
                                            }).collect::<Vec<u8>>();
        let r_i_2d = r_i_ref.chunks(gp.input_size.0 - gp.kernel_size.0 + 1)
                                            .map(|chunk| chunk.to_vec())
                                            .collect::<Vec<Vec<u8>>>();
        let output_data = json!({
            "r_i_q": r_i_2d,
        });
        let file = File::create("else/data/output_data.json").expect("Create output file failed!");
        serde_json::to_writer_pretty(file, &output_data).expect("Output data dump failure!");

        OutputData {
            r_i_raw,
            r_i: r_i_2d,
        }

    }


}