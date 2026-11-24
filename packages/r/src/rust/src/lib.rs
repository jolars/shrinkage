use extendr_api::Result;
use extendr_api::prelude::*;

use faer::MatRef;
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};

mod model;
mod penalty;
mod result;

fn optional(value: Option<impl Into<Robj>>) -> Robj {
    value.map(Into::into).unwrap_or_else(|| ().into())
}

fn error(message: impl ToString) -> Error {
    Error::Other(message.to_string())
}

fn field(list: &List, name: &str) -> Result<Robj> {
    list.iter()
        .find(|(key, _)| *key == name)
        .map(|(_, value)| value)
        .ok_or_else(|| error(format!("missing native argument '{name}'")))
}

#[extendr]
fn shrinkage_fit_native(
    design: List,
    y: &[f64],
    loss: List,
    penalty: List,
    options: List,
    cv: Robj,
) -> Result<List> {
    let request = model::Request::new(y, &loss, &penalty, &options, cv)?;
    let dims = field(&design, "dim")?;
    let [rows, cols] = dims.as_integer_slice().unwrap_or_default() else {
        return Err(error("expected a two-dimensional design"));
    };
    let rows = usize::try_from(*rows).map_err(error)?;
    let cols = usize::try_from(*cols).map_err(error)?;
    let values = field(&design, "values")?;
    let values = values
        .as_real_slice()
        .ok_or_else(|| error("expected double matrix values"))?;
    let offsets = field(&design, "offsets")?;
    if offsets.is_null() {
        if rows.checked_mul(cols) != Some(values.len()) {
            return Err(error("invalid dense matrix dimensions"));
        }
        let x = MatRef::from_column_major_slice(values, rows, cols).as_dyn();
        return request.fit(&x, y);
    }

    let indices = field(&design, "indices")?;
    let as_indices = |value: &Robj| -> Result<Vec<usize>> {
        value
            .as_integer_slice()
            .ok_or_else(|| error("invalid CSC indices"))?
            .iter()
            .map(|&index| usize::try_from(index).map_err(error))
            .collect()
    };
    let offsets = as_indices(&offsets)?;
    let indices = as_indices(&indices)?;
    if offsets.len() != cols + 1
        || offsets.first() != Some(&0)
        || offsets.last() != Some(&values.len())
        || indices.len() != values.len()
        || offsets.windows(2).any(|pair| pair[0] > pair[1])
        || indices.iter().any(|&index| index >= rows)
        || offsets.windows(2).any(|pair| {
            indices[pair[0]..pair[1]]
                .windows(2)
                .any(|indices| indices[0] >= indices[1])
        })
    {
        return Err(error("invalid dgCMatrix CSC storage"));
    }

    let symbolic = SymbolicSparseColMatRef::new_checked(rows, cols, &offsets, None, &indices);
    let x = SparseColMatRef::new(symbolic, values);
    request.fit(&x, y)
}

extendr_module! {
    mod shrinkage;
    fn shrinkage_fit_native;
}
