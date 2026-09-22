//! Fit the same Gaussian lasso using dense and sparse CSC training data.

use faer::Mat;
use faer::sparse::{SparseColMat, Triplet};
use shrinkage::{Lasso, Termination};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rows = [
        [0.0, 1.0],
        [1.0, 0.0],
        [2.0, 3.0],
        [0.0, -1.0],
        [4.0, 0.0],
        [3.0, 2.0],
    ];
    let y = [1.0, 4.0, 2.0, 3.0, 8.0, 5.0];
    let dense = Mat::from_fn(rows.len(), 2, |i, j| rows[i][j]);
    let triplets: Vec<_> = rows
        .iter()
        .enumerate()
        .flat_map(|(i, row)| {
            row.iter()
                .enumerate()
                .filter_map(move |(j, &value)| (value != 0.0).then_some(Triplet::new(i, j, value)))
        })
        .collect();
    let sparse = SparseColMat::<usize, f64>::try_new_from_triplets(rows.len(), 2, &triplets)?;

    let model = Lasso::new(0.15).tolerance(1e-9);
    let fit = model.fit(&dense, &y)?;
    let sparse_fit = model.fit(&sparse, &y)?;
    assert_eq!(fit.termination(), Termination::Converged);
    assert_eq!(sparse_fit.termination(), Termination::Converged);
    for (&dense, &sparse) in fit.coefficients().iter().zip(sparse_fit.coefficients()) {
        assert!((dense - sparse).abs() < 1e-10);
    }

    let new_x = Mat::from_fn(2, 2, |i, j| [[1.5, 0.0], [3.0, 1.0]][i][j]);
    let prediction = fit.predict(&new_x)?;
    for (&dense, &sparse) in prediction.iter().zip(&sparse_fit.predict(&new_x)?) {
        assert!((dense - sparse).abs() < 1e-10);
    }
    println!("Original-scale coefficients: {:?}", fit.coefficients());
    println!("Original-scale intercept: {:.6}", fit.intercept());
    println!("Predictions on new rows: {prediction:?}");
    println!(
        "{:?} after {} sweeps; objective = {:.6}, KKT violation = {:.3e}",
        fit.termination(),
        fit.iterations(),
        fit.objective(),
        fit.kkt_violation()
    );
    Ok(())
}
