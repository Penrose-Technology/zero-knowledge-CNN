#[cfg(test)]
mod tests{
    //use std::ops::{Add, Mul};
    use ark_test_curves::bls12_381::Fr;
    use crate::conv_relu::data_structure::IPForConvRelu;

    // fn conv_2d<T> (x: &Vec<Vec<T>>, w: &Vec<Vec<T>>) -> Vec<Vec<Fr>> 
    // where 
    //     T: Mul + Copy,
    //     T::Output: Add<Output = T::Output> + Copy,
    //     Fr: From<T::Output>, 
    // {
    //     let mut y = Vec::new();
    //     for (i, row) in x.iter().enumerate().take(x.len() - w.len() + 1) {
    //         let mut y_i = Vec::new();
    //         for (j, &item) in row.iter().enumerate().take(x.len() - w.len() + 1) {
    //             let value = item*w[0][0] + x[i][j+1]*w[0][1] + x[i+1][j]*w[1][0] + x[i+1][j+1]*w[1][1];
    //             y_i.push(Fr::from(value));
    //         }
    //         y.push(y_i);
    //     }
    //     y
    // }

    #[test]
    fn layer_test() {
        let (gp, mut pp, vp, input) = IPForConvRelu::<Fr>::setup();
        let proof = IPForConvRelu::<Fr>::prove(&gp, &mut pp, &input);
        let _ = IPForConvRelu::dump_output(&gp, &pp);
        let _ = IPForConvRelu::verify(&gp, &vp, &proof).expect("Layer test failed!");
    }
}