use ark_sumcheck::gkr_round_sumcheck::{
        GKRRoundSumcheck, 
        initialize_phase_one, 
        start_phase1_sumcheck,
        initialize_phase_two,
        start_phase2_sumcheck
    };
use ark_poly::Polynomial;
use ark_ff::FftField;
use ark_poly::{DenseMultilinearExtension, SparseMultilinearExtension};
use ark_sumcheck::gkr_round_sumcheck::data_structures::GKRRoundSumcheckSubClaim;
use ark_sumcheck::ml_sumcheck::protocol::{IPForMLSumcheck, prover::ProverMsg};
use ark_sumcheck::ml_sumcheck::data_structures::PolynomialInfo;
use ark_sumcheck::Error;
use ark_sumcheck::rng::FeedableRNG;

/// Proof for GKR Round Function
pub struct GKRConvProof<F: FftField> {
    pub phase1_sumcheck_msgs: Vec<ProverMsg<F>>,
    pub phase2_sumcheck_msgs: Vec<ProverMsg<F>>,
}

// impl<F: FftField> GKRConvProof<F> {
//     /// Extract the witness (i.e. the sum of GKR)
//     pub fn extract_sum(&self) -> F {
//         self.phase1_sumcheck_msgs[0].evaluations[0] + self.phase1_sumcheck_msgs[0].evaluations[1]
//     }
// }

pub trait GKRConv <F: FftField> {
    fn prove<R: FeedableRNG>(
        rng: &mut R,
        f1: &SparseMultilinearExtension<F>,
        f2: &DenseMultilinearExtension<F>,
        f3: &DenseMultilinearExtension<F>,
        g: &[F],
    ) -> (GKRConvProof<F>, Vec<F>, Vec<F>); 

    fn verify<R: FeedableRNG>(
        rng: &mut R,
        f2_num_vars: usize,
        proof: &GKRConvProof<F>,
        claimed_sum: F,
    ) -> Result<GKRRoundSumcheckSubClaim<F>, Error>;
}

impl <F: FftField> GKRConv<F> for GKRRoundSumcheck<F> {
    fn prove<R: FeedableRNG>(
        rng: &mut R,
        f1: &SparseMultilinearExtension<F>,
        f2: &DenseMultilinearExtension<F>,
        f3: &DenseMultilinearExtension<F>,
        g: &[F],
    ) -> (GKRConvProof<F>, Vec<F>, Vec<F>) {
        assert_eq!(f1.num_vars, 3 * f2.num_vars);
        assert_eq!(f1.num_vars, 3 * f3.num_vars);

        let dim = f2.num_vars;
        let g = g.to_vec();

        let (h_g, f1_g) = initialize_phase_one(f1, f3, &g);
        let mut phase1_ps = start_phase1_sumcheck(&h_g, f2);
        let mut phase1_vm = None;
        let mut phase1_prover_msgs = Vec::with_capacity(dim);
        let mut u = Vec::with_capacity(dim);
        for _ in 0..dim {
            let pm = IPForMLSumcheck::prove_round(&mut phase1_ps, &phase1_vm);

            rng.feed(&pm).unwrap();
            phase1_prover_msgs.push(pm);
            let vm = IPForMLSumcheck::sample_round(rng);
            phase1_vm = Some(vm.clone());
            u.push(vm.randomness);
        }

        let f1_gu = initialize_phase_two(&f1_g, &u);
        let mut phase2_ps = start_phase2_sumcheck(&f1_gu, f3, f2.evaluate(&u));
        let mut phase2_vm = None;
        let mut phase2_prover_msgs = Vec::with_capacity(dim);
        let mut v = Vec::with_capacity(dim);
        for _ in 0..dim {
            let pm = IPForMLSumcheck::prove_round(&mut phase2_ps, &phase2_vm);
            rng.feed(&pm).unwrap();
            phase2_prover_msgs.push(pm);
            let vm = IPForMLSumcheck::sample_round(rng);
            phase2_vm = Some(vm.clone());
            v.push(vm.randomness);
        }

        (
            GKRConvProof {
                phase1_sumcheck_msgs: phase1_prover_msgs,
                phase2_sumcheck_msgs: phase2_prover_msgs,
            },
            u,
            v
        )
    }

    fn verify<R: FeedableRNG>(
        rng: &mut R,
        f2_num_vars: usize,
        proof: &GKRConvProof<F>,
        claimed_sum: F,
    ) -> Result<GKRRoundSumcheckSubClaim<F>, Error> {
        // verify first sumcheck
        let dim = f2_num_vars;

        let mut phase1_vs = IPForMLSumcheck::verifier_init(&PolynomialInfo {
            max_multiplicands: 2,
            num_variables: dim,
        });

        for i in 0..dim {
            let pm = &proof.phase1_sumcheck_msgs[i];
            rng.feed(pm).unwrap();
            let _result = IPForMLSumcheck::verify_round((*pm).clone(), &mut phase1_vs, rng);
        }
        let phase1_subclaim = IPForMLSumcheck::check_and_generate_subclaim(phase1_vs, claimed_sum)?;
        let u = phase1_subclaim.point;

        let mut phase2_vs = IPForMLSumcheck::verifier_init(&PolynomialInfo {
            max_multiplicands: 2,
            num_variables: dim,
        });
        for i in 0..dim {
            let pm = &proof.phase2_sumcheck_msgs[i];
            rng.feed(pm).unwrap();
            let _result = IPForMLSumcheck::verify_round((*pm).clone(), &mut phase2_vs, rng);
        }
        let phase2_subclaim = IPForMLSumcheck::check_and_generate_subclaim(
            phase2_vs,
            phase1_subclaim.expected_evaluation,
        )?;

        let v = phase2_subclaim.point;

        let expected_evaluation = phase2_subclaim.expected_evaluation;

        Ok(GKRRoundSumcheckSubClaim {
            u,
            v,
            expected_evaluation,
        })
    }

}
