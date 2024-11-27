#[cfg(test)]
mod tests{
    //use std::ops::{Add, Mul};
    use ark_test_curves::bls12_381::Fr;
    use crate::conv_relu::data_structure::IPForConvRelu;
    use ark_std::time::Instant;
    use ark_std::fs::File;

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

    fn benchmark<F, R> (func: F) ->
        (R, f64)
    where 
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = func();
        let duration = start.elapsed();
        (result, duration.as_secs_f64())
    }

    #[test]
    fn layer_test() {
        let (setup_result, setup_time) = benchmark(|| IPForConvRelu::<Fr>::setup());
        let (gp, mut pp, vp, input) = setup_result;
 
        let (proof, prove_time) = benchmark(|| IPForConvRelu::<Fr>::prove(&gp, &mut pp, &input));
        let _ = IPForConvRelu::dump_output(&gp, &pp);

        let (_, verify_time) = benchmark(|| IPForConvRelu::verify(&gp, &vp, &proof).expect("Layer test failed!"));
        
        #[derive(serde::Serialize)]
        struct Performance {
            setup: f64,
            prove: f64,
            verify: f64,
        }

        let performance = Performance {
            setup: setup_time,
            prove: prove_time,
            verify: verify_time,  
        };
        let file = File::create("else/performance/benchmark.json").expect("Failed to open benchmark.json!");
        serde_json::to_writer_pretty(file, &performance).expect("Benchmark data dump failure!");
    }
}