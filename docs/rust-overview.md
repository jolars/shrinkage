# Shrinkage for Rust

Fit regularized statistical models with reusable datafits, penalties, and
solvers. Start with `Lasso` for Gaussian lasso, or compose a `Problem` and
select a compatible solver with `fit_with`.

Enable the matrix integration you need through optional Cargo features. The
[installation guide](pages/installation.md) lists supported backend versions.
The [modeling guide](pages/modeling.md) explains coefficient scales, loss
normalization, penalties, and intercepts.

Browse the API reference below, or explore the mathematical definitions of
[datafits](pages/datafits/index.md), [penalties](pages/penalties/index.md), and
[solvers](pages/reference/solvers/index.md).

Fitted results retain original-scale coefficients and preprocessing for
prediction. Inspect the termination reason and stopping diagnostics before
treating a fit as converged. Single typed fits also support `fit_with_observer`
for observing accepted optimization states.
