# Shrinkage

Shrinkage is a Rust library for regularized statistical models. It currently
fits Gaussian lasso models with dense or sparse CSC input through coordinate
descent and fits lasso, ridge, elastic net, and SLOPE through proximal gradient.
The typed and runtime APIs also accept external smooth datafits and complete
proximal penalties. See [DESIGN.md](DESIGN.md) for the architecture and
statistical conventions, and [TODO.md](TODO.md) for the implementation
checklist.

## Gaussian lasso

Enable the `faer` feature for this example:

```rust
use faer::Mat;
use shrinkage::{Lasso, Termination};

let x = Mat::from_fn(3, 1, |i, _| i as f64);
let fit = Lasso::new(0.1).fit(&x, &[1.0, 3.0, 5.0])?;
assert_eq!(fit.termination(), Termination::Converged);
let predictions = fit.predict(&x)?;
```

The typed API composes the design, Gaussian response, L1 penalty, and coordinate
solver. It returns the same `LassoFit` as the convenience API:

```rust
use faer::Mat;
use shrinkage::{CoordinateDescent, Gaussian, L1, Normalization, Problem};

let x = Mat::from_fn(3, 1, |i, _| i as f64);
let y = [1.0, 3.0, 5.0];
let fit = Problem::new(&x, Gaussian::new(&y), L1::new(0.1))
    .normalize(Normalization::Standardize)
    .fit_with(&CoordinateDescent::new())?;
let predictions = fit.predict(&x)?;
```

`Gaussian::fit_intercept(false)` controls the fitted intercept. The typed
problem also accepts `with_centers` and `with_scales`. Its parameters and
predictions use the original design scale, including an intercept induced by
explicit centering when fitted intercepts are disabled.

`Lasso::new(lambda)` defaults to an unpenalized intercept, training-column means
and population standard deviations, a relative duality-gap tolerance of `1e-6`,
and a limit of 10,000 coordinate sweeps. At zero penalty, the default uses an
absolute KKT tolerance of `1e-6`; a residual-scaled certificate can be loose
when the dual constraint requires exact orthogonality. The objective is

$$
  \frac{1}{2n}\|y - \widetilde X\theta
  - \widetilde b\mathbf1\|_2^2 + \lambda\|\theta\|_1.
$$

The penalty acts on the normalized coefficients. `coefficients()`,
`intercept()`, and `predict()` use the original input scale. Fitted centers and
scales remain available through `preprocessing()`; prediction does not estimate
new statistics.

Choose normalization with `.normalize(Normalization::...)`:

  | Option                     | Centering                                      | Scaling                       |
  | -------------------------- | ---------------------------------------------- | ----------------------------- |
  | `Auto` (default)           | Mean when fitting an intercept; otherwise none | Population standard deviation |
  | `None`                     | None                                           | None                          |
  | `Center`                   | Mean                                           | None                          |
  | `Standardize`              | Mean                                           | Population standard deviation |
  | `MinMax`                   | Minimum                                        | Range                         |
  | `MaxAbs`                   | None                                           | Maximum absolute value        |
  | `L1`                       | None                                           | L1 norm                       |
  | `L2`                       | None                                           | L2 norm                       |
  | `Custom { center, scale }` | Selected independently                         | Selected independently        |

Custom choices reuse LazyMatrix's enums, re-exported from Shrinkage:

```rust
use shrinkage::{Centering, Lasso, Normalization, Scaling};

let model = Lasso::new(0.1).normalize(Normalization::Custom {
    center: Centering::Mean,
    scale: Scaling::L2,
});
```

Scales based on norms or maximum absolute values are computed after centering.
Use `.with_centers(vec![...])` and `.with_scales(vec![...])` to supply either
vector directly. Each supplied vector overrides only its own normalization rule.
Supplied centers also determine computed L1, L2, and maximum-absolute scales. A
supplied center must be finite; a supplied scale must be finite and strictly
positive. Both vectors must have one value per column. These values are
validated during fitting, before constructing a lazy matrix. Use
`Normalization::None` for the raw design. Only `Auto` changes centering when
`.fit_intercept(false)` is selected. Explicit centering is honored even without
a fitted intercept, and its induced original-scale intercept is retained.
Builder call order does not change this behavior. Fitted preprocessing exposes
the resolved `centering()` and `scaling()` rules as well as their values and
whether either vector was supplied. A supplied vector overrides its configured
rule. Computed zero scales become one, and zero-norm normalized columns stay at
zero. Inputs must be finite, and the solver does not impute missing values.

A successful `Result` contains a finite fit, which may have reached the
iteration limit. Check `termination()` before treating it as converged. The
result reports `iterations()`, `objective()`, `kkt_violation()`,
`dual_objective()`, and `duality_gap()`. The dual certificate satisfies the
constraints for the normalized design and an unpenalized intercept. The relative
gap uses the loss at zero coefficients with the intercept optimized as a fixed
reference scale. When this loss is zero, the threshold is zero. Select another
criterion with `terminate_on(StoppingCriterion::...)`; `tolerance()` retains its
original meaning as an absolute KKT tolerance. The fit reports its selected
criterion, final value, and threshold. A passing check is confirmed after
reconstructing residuals. Invalid input, numerical failures, and backend
preprocessing failures return distinct `LassoError` variants, with backend error
sources preserved.

The solver keeps sparse centering in a scalar residual offset. Coordinate dots
and updates touch stored column entries, with residual refreshes every 50 sweeps
and before final reporting. Solver storage is `O(n + p)`; fitting does not
materialize a normalized design or form a Gram matrix.

Run the dense and CSC fitting example:

```console
cargo run --locked --example lasso --features faer
```

## Proximal gradient

Use `ProximalGradient` with a differentiable convex predictor loss and an exact
proximal map for the complete convex penalty. The built-in terms are
`L1::new(lambda)`, `Ridge::new(lambda)` for `lambda * ||theta||² / 2`,
`ElasticNet::new(l1, l2)` for `l1 * ||theta||₁ + l2 * ||theta||² / 2`, and
`Slope::new(weights)` for the sorted L1 penalty. Strengths are finite and
nonnegative, and penalties act on normalized coefficients. The fitted intercept
remains unpenalized.

```rust
use faer::{Col, Mat};
use shrinkage::{ElasticNet, Gaussian, Problem, ProximalGradient, Termination};

let x = Mat::from_fn(3, 1, |i, _| i as f64);
let y = [1.0, 3.0, 5.0];
let fit = Problem::new(&x, Gaussian::new(&y), ElasticNet::new(0.1, 0.2))
    .fit_with(&ProximalGradient::<Col<f64>>::new())?;
assert_eq!(fit.termination(), Termination::Converged);
let predictions = fit.predict(&x)?;
```

Select the vector type supported by the design's LazyMatrix products: `Col<f64>`
for faer, `DVector<f64>` for nalgebra, `Array1<f64>` for ndarray, or `Vec<f64>`
for sprs. `ProximalVector` supplies owned CPU buffers for these types. Custom
matrices implement LazyMatrix's fallible `MatVecInto` and `MatTransposeVecInto`
in addition to the training-column capabilities.

The solver starts at zero, backtracks by halving the step until the smooth loss
satisfies its quadratic upper bound, and retains the accepted step. Configure
`initial_step`, `max_backtracks`, and `max_iterations` independently. Exhausting
the line search returns `FitError::LineSearchFailure`; exhausting accepted
updates returns a finite `ProximalFit` with `Termination::IterationLimit`.

The default stopping criterion is an absolute proximal-gradient mapping
tolerance of `1e-6`. For step `t`, the mapping is
`(theta - prox(theta - t * gradient)) / t`. Its infinity norm, maximized with
the absolute intercept derivative, is evaluated at the returned parameters. Set
it with `tolerance` or
`terminate_on(StoppingCriterion::proximal_gradient_mapping(...))`. Other
criteria return `FitError::UnsupportedCombination` with a supported alternative.
`ProximalFit` reports the criterion, value, threshold, objective, and final step
without claiming a duality gap.

External components implement `SmoothDatafit` and `ProximalPenalty`. The latter
requires the prox of the entire term; individual proximal maps do not establish
a valid prox for their sum. Elastic net implements its complete formula
directly.

`Slope::new(weights)` supplies the complete sorted L1 penalty
`sum_j weights[j] * |theta|_(j)`, where magnitudes are sorted from largest to
smallest. Supply one finite, nonnegative, nonincreasing weight per coefficient,
including the penalty strength. Equal weights recover L1; all zero weights leave
coefficients unpenalized. Parameters are checked when the penalty is evaluated
or fitted.

```rust
use shrinkage::{Penalty, ProximalPenalty, Slope};

let penalty = Slope::new(vec![3.0, 1.0]);
assert_eq!(penalty.value(&[-3.0, 3.0])?, 12.0);
let mut output = [0.0; 2];
penalty.prox(&[-3.0, 3.0], 1.0, &mut output)?;
assert_eq!(output, [-1.0, 1.0]);
```

Use `Slope` in typed or runtime proximal problems, or inside `AxisPenalty` with
one weight per coefficient in each selected group. Its exact prox sorts
magnitudes, pools adjacent ordering violations, and restores signs and order. It
reuses owned CPU scratch buffers. Both the penalty ordering and weights act on
normalized coefficients; back-transforming the fit preserves predictions without
changing the penalty that was optimized. Run `task bench:slope` for distinct and
tied magnitudes, and `task bench:proximal:heap` for isolated SLOPE value and
proximal allocation checks.

The solver shares the lasso's normalization preparation and
prediction-preserving parameter transformation. It reuses CPU work buffers and
never forms a Gram matrix or a normalized design. LazyMatrix 0.5.0 clones the
coefficient input on each scaled forward product, so normalized iterations still
allocate. Run `task bench:proximal` to measure dense and CSC lasso, ridge, and
elastic-net fitting through typed and runtime problems, including validation and
preprocessing.

Use `solver.solve_path(&problem, &penalties)` to fit penalties in their supplied
order. It prepares the design once, reuses native and iteration buffers, and
warms each point from the preceding optimization-scale coefficients and
intercept. Each point resets its step and iteration budget. Returned fits own
independent parameters and preprocessing. For runtime-selected penalties, use
`solve_runtime_path` with boxed `ProximalPenalty` entries. Check each fit's
termination status; an iteration-limit point still supplies the next warm start.

Run `task bench:proximal:heap` for isolated allocation measurements. On the
1,024-by-64 faer fixtures, 100 scaled forward products allocate 100 input clones
(51,200 bytes); transposed products and elastic-net proximal calls allocate
nothing. The published LazyMatrix 0.5.0 caller-owned scratch API also allocates
nothing for the measured dense and CSC forward products. Shrinkage's current
adapter uses the cloning product API. A three-point dense path uses 750
allocations versus 924 for independent fits; CSC uses 749 versus 921. These
counts include preparation and owned results and depend on convergence.

## Regularization paths

`Lasso::fit_path(&x, &y, &lambdas)` fits explicit strengths in caller order.
`Problem::fit_path_with(&solver, penalties)` accepts an owned vector of complete
penalties, including independent elastic-net strengths and mixed runtime
penalties. Both return `PathFit`, which retains the corresponding penalty
settings and fitted models. Iterate with `path.iter()` or inspect `path.fits()`
and `path.penalties()`. Each model owns its original-scale parameters,
preprocessing, and convergence diagnostics.

For automatic sequences, use `Lasso::fit_regularization_path` or
`Problem::fit_regularization_path_with`:

```rust
use faer::{Col, Mat};
use shrinkage::{ElasticNet, Gaussian, PathOptions, Problem, ProximalGradient};

let x = Mat::from_fn(3, 1, |i, _| i as f64);
let y = [1.0, 3.0, 5.0];
let problem = Problem::new(&x, Gaussian::new(&y), ElasticNet::from_alpha(0.0, 0.5));
let path = problem.fit_regularization_path_with(
    &ProximalGradient::<Col<f64>>::new(),
    &PathOptions::new().nlambda(50).lambda_min_ratio(0.01),
)?;
for ((penalty, fit), lambda) in path.iter().zip(path.lambdas().unwrap()) {
    println!("lambda={lambda}, l1={}, l2={}, status={:?}",
        penalty.l1(), penalty.l2(), fit.termination());
}
```

Automatic paths optimize the null intercepts, calculate the coefficient gradient
on the normalized design, and generate decreasing, evenly log-spaced strengths.
Defaults follow glmnet: 100 points and an endpoint ratio of `0.01` when there
are fewer observations than features, or `0.0001` otherwise. Configure the
length, ratio, and optional starting override through `PathOptions`. Every
requested point is fitted. A zero start produces a sequence of zeros.

`ElasticNet::from_alpha(lambda, alpha)` uses
`lambda * (alpha * ||theta||₁ + (1 - alpha) * ||theta||² / 2)`, with alpha in
`[0, 1]`. Automatic paths fix alpha while varying lambda. The existing
`ElasticNet::new(l1, l2)` retains independent strengths for explicit paths;
positive strengths also determine their equivalent mixing ratio. Two
independently supplied zero strengths do not identify alpha, so use `from_alpha`
when configuring such an automatic path.

`maximum_penalty_with` reports `PenaltyMaximum::Finite` or `Unbounded`. Lasso
also exposes `lambda_max`. The finite L1 threshold is the largest absolute null
gradient; positive-alpha elastic net divides it by alpha. SLOPE uses cumulative
sorted gradient magnitudes divided by cumulative base weights, and each lambda
multiplies the supplied weight shape. Multivariate axis penalties take the
largest group threshold without changing loss normalization. Ridge generally has
no finite zero-coefficient threshold. Its automatic path starts at the L1
threshold divided by `0.001`, including elastic net at alpha zero.
`path.maximum_penalty()` and `path.lambda_start()` distinguish this anchor from
a true threshold. An explicit starting override does not change the reported
threshold. A free-intercept binomial null model needs responses whose mean lies
strictly between zero and one; explicit paths remain available otherwise.

Each path validates and prepares the design once, retains optimization-scale
warm starts and numerical workspaces, and resets the iteration budget at each
point. Proximal steps also reset. Gaussian lasso paths use Gap Safe screening
and working sets, with refreshed full-design checks before convergence.
`CoordinateDescent::screening(false)` and `working_sets(false)` disable these
path optimizations. Screening is disabled at zero penalty, and the default lasso
stopping criterion resolves separately for each strength. Check each fit's
termination: iteration-limit results initialize the next point, while fitting
errors return no partial path.

The same APIs support `MultivariateGaussian` with `AxisPenalty`, retaining
response-specific intercepts. Runtime automatic paths use `RuntimePathProblem`
with a boxed `PathDatafit` and `ErasedPathPenalty`, preserving the required
capabilities before erasure. External components can implement
`NullModelDatafit` and `PathPenalty`; ordinary explicit paths still require only
the existing smooth-datafit and proximal-penalty capabilities. The existing
proximal `solve_path` and `solve_runtime_path` methods keep their vector return
types. Run `cargo run --example regularization_path --features faer_v0_24` for
lasso, elastic-net, and runtime binomial paths.

## Multivariate Gaussian

`MultivariateGaussian` fits several Gaussian responses jointly with the same
proximal iteration. Supply responses in response-major order: each response
column contains all observations contiguously. The coefficient matrix has
features as rows and responses as columns, with each response's feature vector
stored contiguously. `CoefficientLayout` records both dimensions, and
`AxisPenalty` makes the penalty axis explicit:

```rust
use faer::{Col, Mat};
use shrinkage::{
    AxisPenalty, CoefficientLayout, MultivariateGaussian, PenaltyAxis, Problem,
    ProximalGradient, Ridge,
};

let x = Mat::from_fn(3, 1, |i, _| i as f64);
let y = [1.0, 3.0, 5.0, -2.0, -1.0, 0.0];
let layout = CoefficientLayout::new(1, 2)?;
let fit = Problem::new(
    &x,
    MultivariateGaussian::new(&y, 3, 2),
    AxisPenalty::new(layout, PenaltyAxis::Features, Ridge::new(0.1)),
)
.fit_with(&ProximalGradient::<Col<f64>>::new())?;
let first_response = fit.coefficients_for_response(0).unwrap();
let intercepts = fit.intercepts();
let predictions = fit.predict(&x)?;
```

`PenaltyAxis::Features` applies the complete vector penalty independently to
each response's feature vector. `Responses` applies it to the response vector at
each feature, allowing an external proximal penalty to couple responses. `All`
applies it once to the full coefficient matrix in response-major order. Group
values are summed, and each group uses the supplied strength unchanged. The loss
is `||Y - predictor||_F² / (2n)`, averaged over observations and summed over
responses. Separable lasso, ridge, and elastic-net fits therefore match
independent response fits using the same strengths.

By default, each response has an unpenalized intercept. Shared training-feature
centers and scales apply to every response; original-scale coefficients and
intercepts preserve predictions, including when explicit centering induces an
intercept with `fit_intercept(false)`. `MultivariateFit` retains the layout,
preprocessing, joint objective, and joint stopping diagnostics. Prediction
returns response-major columns with the new batch's observation count.
`RuntimeMultivariateProblem` accepts a boxed `RuntimeDesign` and an
`AxisPenalty<Box<dyn ProximalPenalty>>`, and uses the same joint solver. These
coefficient interfaces remain provisional CPU layouts.

Run the example, which also checks independent ridge fits:

```console
cargo run --locked --example multivariate --features faer_v0_24
```

`task bench:proximal` includes dense and CSC multivariate fits with four
responses and both feature and response penalty axes.

## Runtime composition

`RuntimeProblem` owns independently selected design, smooth datafit, and
complete proximal penalty objects. `MatrixDesign` adapts a concrete LazyMatrix
backend without depending on the datafit or penalty type:

```rust
use faer::{Col, Mat};
use shrinkage::{
    ElasticNet, Gaussian, L1, MatrixDesign, ProximalGradient, ProximalPenalty,
    RuntimeDesign, RuntimeProblem, SmoothDatafit,
};

let x = Mat::from_fn(3, 1, |i, _| i as f64);
let y = [1.0, 3.0, 5.0];
let design: Box<dyn RuntimeDesign> = Box::new(MatrixDesign::<_, Col<f64>>::new(&x));
let datafit: Box<dyn SmoothDatafit + '_> = Box::new(Gaussian::new(&y));
let use_elastic_net = true;
let penalty: Box<dyn ProximalPenalty> = if use_elastic_net {
    Box::new(ElasticNet::new(0.1, 0.2))
} else {
    Box::new(L1::new(0.1))
};
let fit = RuntimeProblem::new(design, datafit, penalty)
    .fit_with(&ProximalGradient::new())?;
let predictions = fit.predict(&x)?;
```

Choose `MatrixDesign`'s vector type to match the concrete backend. Runtime
iteration uses slice buffers, so its solver is `ProximalGradient<Vec<f64>>`,
regardless of the selected backend. The adapters retain native vectors and
generic column views. The same iteration function handles typed and runtime
problems, with virtual calls only for whole products, datafit evaluations, and
proximal operations. It never boxes an already composed typed problem.

Normalization, supplied centers and scales, the unpenalized intercept, and
original-scale fitted parameters follow the typed API's conventions. Automatic
centering resolves during fitting using the selected datafit's intercept policy.
Runtime backend failures return `FitError<BackendError>` with their operation
context and original error accessible through `BackendError::as_error` and the
error source chain. Product buffers and iteration storage are reused within a
fit and across regularization path points. Independently requested fits prepare
their own storage.

Typed and runtime fits use the same component contracts and return matching
parameters, predictions, normalization metadata, and stopping diagnostics,
including when they reach an iteration limit. Runtime components must implement
`SmoothDatafit` and `ProximalPenalty`; value-only components are rejected by
Rust's trait bounds. A sum of penalties needs a proximal map for its complete
term.

Selecting `CoordinateDescent` for a `RuntimeProblem` returns
`FitError::UnsupportedCombination` before preparing the design. Runtime proximal
oracles do not expose its Gaussian/L1 coordinate updates or dual certificate.
Use `ProximalGradient<Vec<f64>>` for runtime composition, or pass a typed
`Problem` with `Gaussian` and `L1` to `CoordinateDescent`. Compatibility errors
provide the solver, reason, and suggested alternative. Invalid data and option
values still return `FitError::InvalidInput`, while iteration limits remain
fitted results with `Termination::IterationLimit`.

## Experimental block reads

The `experimental-block-reader` feature exposes `experimental::fit_buffered` and
`experimental::BufferedDesign`. Supply a positive rectangular block shape and an
initialized `&mut [f64]` large enough for that shape. Validation and forward and
transpose products reuse the buffer; normalization statistics retain the
backend's combined scan hook. Proximal iteration, backtracking, and fitted
coefficient transformations use the existing solver.

The prototype uses the published LazyMatrix `ReadBlock` capability:

```sh
cargo test --locked --features experimental-block-reader,zarrs_v0_22
cargo run --locked --example zarrs_buffered --features experimental-block-reader,zarrs_v0_22
```

The example writes a temporary file-backed Zarr design one chunk at a time, fits
ridge regression, and predicts through the buffered operator. `ProximalFit`'s
regular `predict` method still requires borrowed columns. Reader and adapter
interfaces remain provisional. A failed read yields no usable block; product
outputs may be partial and must be discarded. Observation and coefficient
vectors remain in RAM, and storage-chunk and codec workspace is additional to
the caller's buffer. Rectangles crossing chunk boundaries can repeat decoding
across successive reads.

## Matrix dependency

The crate pins LazyMatrix 0.5.0 from crates.io. No matrix backend is enabled by
default. Versioned features select LazyMatrix's integrations for a particular
release line; the short names remain convenience aliases:

  | Versioned feature | Alias      | Dependencies                           | Supported input                                      |
  | ----------------- | ---------- | -------------------------------------- | ---------------------------------------------------- |
  | `faer_v0_24`      | `faer`     | faer 0.24                              | Dense matrices, dense views, and sparse CSC matrices |
  | `nalgebra_v0_34`  | `nalgebra` | nalgebra 0.34 and nalgebra-sparse 0.11 | Dense matrices, dense views, and sparse CSC matrices |
  | `ndarray_v0_17`   | `ndarray`  | ndarray 0.17                           | Two-dimensional arrays and borrowed views            |
  | `sprs_v0_11`      | `sprs`     | sprs 0.11                              | Sparse CSC matrices and views wrapped in `SprsCsc`   |
  | `zarrs_v0_22`     | `zarrs`    | zarrs 0.22                             | Chunked arrays through the experimental adapter      |

Match your direct matrix dependency to the selected release line. Each versioned
feature works independently of its alias, including the faer examples and
benchmarks. Shrinkage currently exposes one release line per backend. LazyMatrix
0.5.0 implements every enabled backend release independently, preserving the
selected matrix types under feature unification.

With `ndarray_v0_17` or its `ndarray` alias, fit directly against an array or
borrowed view:

```rust
use ndarray::array;
use shrinkage::Lasso;

let x = array![[0.0], [1.0], [2.0]];
let fit = Lasso::new(0.1).fit(&x.view(), &[1.0, 3.0, 5.0])?;
let predictions = fit.predict(&x)?;
```

Row-major, column-major, and strided arrays are supported. Column-major storage
keeps each column contiguous for coordinate descent.

With `sprs_v0_11` or its `sprs` alias, wrap CSC storage in
`shrinkage::lazymatrix::SprsCsc`. The wrapper checks storage orientation without
copying entries:

```rust
use shrinkage::{Lasso, lazymatrix::SprsCsc};
use sprs::CsMat;

let x = CsMat::new_csc((3, 1), vec![0, 2], vec![1, 2], vec![1.0, 2.0]);
let columns = SprsCsc::try_new(x.view()).unwrap();
let fit = Lasso::new(0.1).fit(&columns, &[1.0, 3.0, 5.0])?;
let predictions = fit.predict(&columns)?;
```

`SprsCsc::try_new` accepts owned matrices and borrowed views. It returns CSR
input unchanged as an error; explicitly convert with `csr.to_csc()` before
wrapping. Custom matrices can implement LazyMatrix's `RawColumns<f64>` and
`ColumnStats<f64>` capabilities.

Run the dense/sparse normalization example:

```console
cargo run --locked --example normalization --features faer
```

The example checks that lazy products preserve predictions after transforming
coefficients and the intercept back to the original scale. It does not fit a
model.

## Development

Enter the development environment and run the complete local check:

```console
devenv shell
task check
```

The toolchain is pinned to Rust 1.89.0, which is also the minimum supported Rust
version. `Cargo.lock` and `devenv.lock` are tracked for reproducible
development. CI runs the same Rust checks, example, and Panache checks as
`task check`.

Useful focused commands are:

```console
task format
task lint
task test
task bench
task bench:heap
```

`task bench` measures end-to-end dense and CSC fitting time, including
validation and preprocessing, with only the faer backend enabled. Dataset
construction is outside the timed region.

`task bench:heap` runs a separate release-mode harness with the [DHAT
allocator](https://docs.rs/dhat/0.3.3/dhat/). It reports CSV rows for dense and
CSC fits at three sizes: 1,024 by 64, 4,096 by 64, and 1,024 by 1,024. Both
harnesses use the same deterministic data generator and require converged fits.
The heap harness reports allocation count, cumulative allocated bytes, peak live
bytes, and bytes retained by the fitted result, including its dual certificate.
It profiles validation, preprocessing, and solving, excluding input construction
and report printing. All tracked bytes must be released when the fit is dropped.

The heap harness also checks a loose linear budget of `128 * (n + p)` bytes. The
wide case makes both a dense normalized design and a full Gram matrix larger
than this budget. This guards against those materializations for the measured
cases; it is not a universal memory guarantee. These figures measure requested
heap bytes, not process RSS, allocator overhead, or stack usage. Keep timing
runs separate because allocation tracking adds overhead. DHAT is a development
dependency and does not change the library's default dependencies.

The heap checks run in `task check` and CI. On x86-64 with Rust 1.89.0, the
1,024-by-64 fixture takes 70 allocations for dense input and 69 for CSC input;
both peak at 28,160 live bytes. Increasing the row count to 4,096 raises the
peak to 101,888 bytes. Increasing the column count to 1,024 instead raises it to
81,920 bytes, compared with 8,388,608 bytes for a full Gram matrix. These are
fixture measurements, not allocation-free iteration claims or guarantees for
other datasets.

Modules use `name.rs` and `name/child.rs`; the repository does not use `mod.rs`.

## Releases

Versionary prepares release pull requests after CI passes on `main` and manages
`CHANGELOG.md`. Before enabling releases on GitHub, configure the
`RELEASE_TOKEN` repository secret with access to contents, pull requests, and
issues. Registry publication is not configured.

## License

MIT. See [LICENSE](LICENSE).
