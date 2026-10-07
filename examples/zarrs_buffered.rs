//! Fit a file-backed Zarr design with a reusable block buffer.

use shrinkage::experimental::{BufferedDesign, fit_buffered};
use shrinkage::lazymatrix::{MatVecInto, ZarrMatrix};
use shrinkage::{Gaussian, Normalization, Problem, ProximalGradient, Ridge, Termination};
use std::sync::Arc;
use zarrs::array::{ArrayBuilder, DataType};
use zarrs::filesystem::FilesystemStore;

fn value(row: usize, column: usize) -> f64 {
    match column {
        0 => row as f64 / 127.0,
        1 => (row % 5) as f64,
        2 => (row as f64).sin(),
        _ => 5.0,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let store = Arc::new(FilesystemStore::new(directory.path())?);
    let array = ArrayBuilder::new(vec![127, 4], vec![32, 2], DataType::Float64, 0.0_f64)
        .build(store, "/design")?;
    array.store_metadata()?;
    for row_chunk in 0..4 {
        for col_chunk in 0..2 {
            let values: Vec<_> = (0..32)
                .flat_map(|i| (0..2).map(move |j| value(row_chunk * 32 + i, col_chunk * 2 + j)))
                .collect();
            array.store_chunk_elements(&[row_chunk as u64, col_chunk as u64], &values)?;
        }
    }
    let matrix = ZarrMatrix::<_, f64>::try_new(array)?;
    let response: Vec<_> = (0..127)
        .map(|i| 3.0 + 0.5 * value(i, 0) - 1.2 * value(i, 1))
        .collect();
    let problem = Problem::new(&matrix, Gaussian::new(&response), Ridge::new(0.1))
        .normalize(Normalization::Standardize);
    let mut buffer = vec![0.0; 32 * 2];
    let fit = fit_buffered(&problem, &ProximalGradient::new(), [32, 2], &mut buffer)?;
    assert_eq!(fit.termination(), Termination::Converged);
    let mut predictions = vec![0.0; response.len()];
    BufferedDesign::new(&matrix, [32, 2], &mut buffer)
        .matvec_into(&fit.coefficients().to_vec(), &mut predictions)?;
    let mse = predictions
        .iter()
        .zip(&response)
        .map(|(prediction, response)| (prediction + fit.intercept() - response).powi(2))
        .sum::<f64>()
        / response.len() as f64;
    println!(
        "{:?} after {} iterations; buffer: {} bytes; MSE: {mse:.6}",
        fit.termination(),
        fit.iterations(),
        buffer.len() * std::mem::size_of::<f64>()
    );
    Ok(())
}
