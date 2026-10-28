# Optimization explorer

This private package runs Shrinkage's Rust solvers in a browser worker. It uses
a separate Cargo workspace and enables only the nalgebra integration. JavaScript
renders scalar fields and recorded iterates; Rust evaluates every loss, penalty,
and objective and determines solver compatibility and convergence.

From the repository root, inside `devenv shell`:

```sh
task docs:widget:build
task docs:widget:check
task docs:widget:figure
```

Wrap expensive commands once with `compute run --` on shared local machines. The
build requires the toolchain's `wasm32-unknown-unknown` target and
`wasm-bindgen` 0.2.127. The CLI version must match the locked Rust dependency.
Node dependencies are pinned in this package's lockfile. Generated bindings,
bundles, test artifacts, and Rust build output stay untracked.

The generated landing page reserves responsive space for the explorer and shows
a loading indicator until the first fit is ready. The static illustration stays
available when JavaScript is disabled or the component cannot load. Reduced
motion preferences disable the loading animation.

The explorer fixes the intercept at zero and disables additional normalization.
It constructs 32 deterministic observations with centered, unit-variance columns
and the selected correlation. Gaussian responses equal the selected linear
signal. Binomial responses are fractional proportions from its logistic
transformation, giving a finite unpenalized optimum throughout the control
range.

The penalty region uses the complete cost, including strength, with budget
`P(β̂)` at the penalized optimum. It overlays the datafit contours and shows the
equivalent constrained problem. The heavier datafit contour passes through the
optimum. Zero-budget regions appear as a point or line; a zero penalty leaves
the entire plane feasible. Both panels share coefficient coordinates. A tighter
proximal-gradient fit provides the reference optimum, independently of solver
selection. Filled dark orange dots mark that same penalized optimum in both
plots. Coordinate descent is selected initially, with its full trajectory shown
from zero coefficients on the penalized-objective panel. The datafit panel shows
only the contours, constraint region, and optimum. The solver dropdown lists
compatible methods. Changing the problem retains the selected solver when
supported and otherwise selects an applicable solver automatically. Dragging the
hand handle moves the starting point and refits the actual Rust solver. The
handle supports mouse and touch input, as well as arrow keys, and stays within
the current plot limits. Dragging and solver changes preserve the current
contours and axis limits. There are no playback controls. Binomial trajectories
show only accepted outer IRLS steps. Inner quadratic proposals are excluded from
the trace and viewport bounds because backtracking may shorten them before
acceptance. Gaussian trajectories also show inner coordinate and block updates,
marked with hollow dots, because these optimize the displayed objective
directly. Filled dots mark accepted outer updates.

The widget offers L1, ridge, elastic net, SLOPE, and clustered lasso. Group
lasso and chain fused lasso remain available through the library API.

Penalty strength defaults to 0.05. The slider maps its position to λ using
`λ = 0.001 × expm1(t × log1p(2 / 0.001))`, for `t` between zero and one. This
gives weak penalties more room while preserving zero and the upper limit of two.
The output and accessible value report the actual λ, rather than the slider
position. The constraint budget remains `P(β̂)` for that actual penalty.

`npm --prefix docs/widget run test:browser` tests the isolated component. Set
`SHRINKAGE_DOCS_SITE` to an absolute generated site directory to run the same
browser checks against the landing page. Screenshots are saved under
`docs/widget/artifacts`. The static fallback is generated from the native scene
evaluator and verified during documentation checks.
