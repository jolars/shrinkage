//! Joint matrix coefficients through the shared proximal iteration.

use lazymatrix::{ColumnStats, MatTransposeVecInto, MatVecInto, RawColumns};

use super::{NativeDesign, iterate};
use crate::datafit::StackedGaussian;
use crate::error::invalid;
use crate::{
    AxisPenalty, BackendError, CoefficientLayout, FitError, MultivariateFit, MultivariateGaussian,
    Penalty, Preprocessing, Problem, ProximalDesign, ProximalGradient, ProximalPenalty,
    ProximalVector, RuntimeDesign, Solver,
};

// Repeating products avoids materializing either a Kronecker design or a
// normalized matrix. Appended intercept columns preserve the vector iteration's
// line search and mapping, while the penalty adapter leaves them unpenalized.
struct StackedDesign<'a, O: ?Sized> {
    design: &'a mut O,
    layout: CoefficientLayout,
    fit_intercept: bool,
    nrows: usize,
    ncols: usize,
}

impl<'a, O: ProximalDesign + ?Sized> StackedDesign<'a, O> {
    fn new(
        design: &'a mut O,
        layout: CoefficientLayout,
        fit_intercept: bool,
    ) -> Result<Self, FitError<O::Error>> {
        let nrows = design
            .nrows()
            .checked_mul(layout.nresponses())
            .ok_or_else(|| invalid("stacked observation dimensions overflow usize"))?;
        let ncols = layout
            .len()
            .checked_add(if fit_intercept {
                layout.nresponses()
            } else {
                0
            })
            .ok_or_else(|| invalid("stacked coefficient dimensions overflow usize"))?;
        Ok(Self {
            design,
            layout,
            fit_intercept,
            nrows,
            ncols,
        })
    }
}

impl<O: ProximalDesign + ?Sized> ProximalDesign for StackedDesign<'_, O> {
    type Error = O::Error;
    fn nrows(&self) -> usize {
        self.nrows
    }
    fn ncols(&self) -> usize {
        self.ncols
    }

    fn forward(&mut self, input: &[f64], output: &mut [f64]) -> Result<(), FitError<Self::Error>> {
        if input.len() != self.ncols || output.len() != self.nrows {
            return Err(invalid(
                "stacked forward product dimensions must match the layout",
            ));
        }
        let n = self.design.nrows();
        let p = self.layout.nfeatures();
        for response in 0..self.layout.nresponses() {
            let column = &mut output[response * n..(response + 1) * n];
            self.design
                .forward(&input[response * p..(response + 1) * p], column)?;
            if self.fit_intercept {
                let intercept = input[self.layout.len() + response];
                for value in column {
                    *value += intercept;
                }
            }
        }
        Ok(())
    }

    fn transpose(
        &mut self,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), FitError<Self::Error>> {
        if input.len() != self.nrows || output.len() != self.ncols {
            return Err(invalid(
                "stacked transposed product dimensions must match the layout",
            ));
        }
        let n = self.design.nrows();
        let p = self.layout.nfeatures();
        for response in 0..self.layout.nresponses() {
            let column = &input[response * n..(response + 1) * n];
            self.design
                .transpose(column, &mut output[response * p..(response + 1) * p])?;
            if self.fit_intercept {
                output[self.layout.len() + response] = column.iter().sum();
            }
        }
        Ok(())
    }

    fn into_preprocessing(self: Box<Self>) -> Preprocessing {
        unreachable!("the borrowed base design owns preprocessing")
    }
}

struct WithIntercepts<'a, P> {
    penalty: &'a AxisPenalty<P>,
    nintercepts: usize,
}

impl<P> WithIntercepts<'_, P> {
    fn validate(&self, input: &[f64]) -> Result<usize, FitError> {
        let len = self.penalty.layout().len();
        if input.len() != len + self.nintercepts {
            return Err(invalid(
                "parameter length must match coefficients and intercepts",
            ));
        }
        if input[len..].iter().any(|value| !value.is_finite()) {
            return Err(invalid("intercepts must be finite"));
        }
        Ok(len)
    }
}

impl<P: ProximalPenalty> Penalty for WithIntercepts<'_, P> {
    type Error = FitError;
    fn value(&self, input: &[f64]) -> Result<f64, FitError> {
        let len = self.validate(input)?;
        self.penalty.value(&input[..len])
    }
}

impl<P: ProximalPenalty> ProximalPenalty for WithIntercepts<'_, P> {
    fn prox(&self, input: &[f64], step: f64, output: &mut [f64]) -> Result<(), FitError> {
        let len = self.validate(input)?;
        if output.len() != input.len() {
            return Err(invalid("proximal input and output lengths must match"));
        }
        self.penalty.prox(&input[..len], step, &mut output[..len])?;
        output[len..].copy_from_slice(&input[len..]);
        Ok(())
    }
}

fn validate_layout<E, P>(
    nfeatures: usize,
    datafit: &MultivariateGaussian<'_>,
    penalty: &AxisPenalty<P>,
) -> Result<CoefficientLayout, FitError<E>> {
    let layout = penalty.layout();
    if layout.nfeatures() != nfeatures || layout.nresponses() != datafit.nresponses() {
        return Err(invalid(
            "penalty feature and response axes must match the design and datafit",
        ));
    }
    Ok(layout)
}

fn solve_joint<O, P, V>(
    design: &mut O,
    datafit: &MultivariateGaussian<'_>,
    penalty: &AxisPenalty<P>,
    options: &ProximalGradient<V>,
    threshold: f64,
) -> Result<super::Solution, FitError<O::Error>>
where
    O: ProximalDesign + ?Sized,
    P: ProximalPenalty,
{
    let layout = validate_layout(design.ncols(), datafit, penalty)?;
    let mut stacked = StackedDesign::new(design, layout, datafit.fits_intercept())?;
    let term = WithIntercepts {
        penalty,
        nintercepts: if datafit.fits_intercept() {
            layout.nresponses()
        } else {
            0
        },
    };
    iterate(
        &mut stacked,
        &StackedGaussian(datafit),
        &term,
        options,
        threshold,
    )
}

impl<M, P, V> Solver<Problem<&M, MultivariateGaussian<'_>, AxisPenalty<P>>> for ProximalGradient<V>
where
    M: RawColumns<f64> + ColumnStats<f64> + MatVecInto<V> + MatTransposeVecInto<V> + ?Sized,
    P: ProximalPenalty,
    V: ProximalVector,
{
    type Fit = MultivariateFit;
    type Error = FitError<M::Error>;

    fn solve(
        &self,
        problem: &Problem<&M, MultivariateGaussian<'_>, AxisPenalty<P>>,
    ) -> Result<Self::Fit, Self::Error> {
        let threshold = self.validate()?;
        let datafit = problem.datafit();
        datafit.validate(problem.design().nrows())?;
        let layout = validate_layout(problem.design().ncols(), datafit, problem.penalty())?;
        let prepared = problem.prepare(datafit.fits_intercept())?;
        let mut design = NativeDesign::<_, V>::new(prepared);
        let solution = solve_joint(&mut design, datafit, problem.penalty(), self, threshold)?;
        MultivariateFit::from_solution(
            solution,
            design.into_preprocessing(),
            layout,
            datafit.fits_intercept(),
        )
    }
}

impl<P: ProximalPenalty>
    Solver<Problem<Box<dyn RuntimeDesign + '_>, MultivariateGaussian<'_>, AxisPenalty<P>>>
    for ProximalGradient<Vec<f64>>
{
    type Fit = MultivariateFit;
    type Error = FitError<BackendError>;

    fn solve(
        &self,
        problem: &Problem<Box<dyn RuntimeDesign + '_>, MultivariateGaussian<'_>, AxisPenalty<P>>,
    ) -> Result<Self::Fit, Self::Error> {
        let threshold = self.validate()?;
        let datafit = problem.datafit();
        datafit.validate(problem.design().nrows())?;
        let layout = validate_layout(problem.design().ncols(), datafit, problem.penalty())?;
        let mut design = problem.design().prepare(
            problem.normalization(),
            problem.centers(),
            problem.scales(),
            datafit.fits_intercept(),
        )?;
        let solution = solve_joint(design.as_mut(), datafit, problem.penalty(), self, threshold)?;
        MultivariateFit::from_solution(
            solution,
            design.into_preprocessing(),
            layout,
            datafit.fits_intercept(),
        )
    }
}
