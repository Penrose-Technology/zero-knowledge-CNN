use ark_ff::{Field, Zero};
use ark_poly::DenseMultilinearExtension;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

#[derive(Clone, Debug)]
pub struct LookupTable<F: Field> {
    /// f_i, i \in [1, M], private polynomials
    pub private_columns: Vec<DenseMultilinearExtension<F>>,
    /// t, public polynomial
    pub public_column: DenseMultilinearExtension<F>,
    /// max number of private polynomials
    pub max_private_columns: usize,
    /// number of polynomials in a single group
    pub group_length: usize,
    /// number of variables of the polynomial
    pub num_variables: usize,
    /// number of groups
    pub num_groups: usize,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug)]
pub struct LookupTableInfo<F: Field> {
    /// t, public polynomial
    pub public_column: DenseMultilinearExtension<F>,
    /// max number of private polynomials
    pub num_private_columns: usize,
    /// number of polynomials in a single group
    pub group_length: usize,
    /// number of variables of the polynomial
    pub num_variables: usize,
    /// number of groups
    pub num_groups: usize,
}

impl<F: Field> LookupTable<F> {
    pub fn info(&self) -> LookupTableInfo<F> {
        LookupTableInfo::<F> {
            public_column: self.public_column.clone(),
            num_private_columns: self.private_columns.len(),
            group_length: self.group_length,
            num_variables: self.num_variables,
            num_groups: self.num_groups,
        }
    }
}

impl<F: Field> LookupTable<F> {
    pub fn new(num_variables: usize, group_length: usize) -> Self {
        LookupTable {
            private_columns: Vec::new(),
            public_column: DenseMultilinearExtension::zero(),
            max_private_columns: 10,
            group_length,
            num_variables,
            num_groups: 1,
        }
    }

    pub fn add_public_column(&mut self, new_public_column: DenseMultilinearExtension<F>) {
        self.public_column = new_public_column;
    }

    pub fn add_private_column(&mut self, new_private_column: DenseMultilinearExtension<F>) {
        if self.max_private_columns < self.private_columns.len() {
            panic!("Exceed max column limitation");
        }

        self.private_columns.push(new_private_column);

        let public_col_int = if self.public_column == DenseMultilinearExtension::zero() { 0 } else { 1 };

        self.num_groups = (self.private_columns.len() + public_col_int + self.group_length - 1) / self.group_length;
    }
}
