#[cfg(test)]
mod tests {
    use ark_std::test_rng;
    use ark_ff::UniformRand;
    use ark_test_curves::bls12_381::Fr;
    use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
    use crate::convolution::{Conv, MLConvolution, fft::Convpolynomial};

    #[test]
    fn conv_test() {
        let num_coeffs = 4;
        let mut rng = test_rng();
        let domain = GeneralEvaluationDomain::<Fr>::new(num_coeffs).unwrap();
        let x_points = vec![Fr::from(11), Fr::from(12), Fr::from(13), Fr::from(14)];
        let mut x_poly = Convpolynomial::new(x_points);
        let w_points = vec![Fr::from(5), Fr::from(6), Fr::from(7), Fr::from(9)];
        let mut w_poly = Convpolynomial::new(w_points);
        let g = vec![Fr::rand(&mut rng), Fr::rand(&mut rng)];

        let mut y_poly = MLConvolution::evaluate(&domain, &mut x_poly, &mut w_poly);
        let proof = MLConvolution::prove(&domain, &g, &mut x_poly, &mut w_poly, &mut y_poly);
        MLConvolution::verify(&g, &domain, &proof).unwrap();

    }

}