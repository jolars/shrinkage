"""Single-response datafits for compositional fitting."""

from dataclasses import dataclass, field

__all__ = ["Binomial", "Gaussian"]


@dataclass(frozen=True, eq=False)
class Gaussian:
    """Average squared loss divided by two, with an unpenalized intercept.

    ``y`` must satisfy the dense response contract: an exact, aligned,
    native-endian float64 NumPy array of shape ``(n_samples,)``. The array is
    retained by reference and copied when the problem is fitted.
    """

    y: object = field(repr=False)
    fit_intercept: bool = field(default=True, kw_only=True)


@dataclass(frozen=True, eq=False)
class Binomial:
    """Average Bernoulli negative log likelihood with a logit link.

    ``y`` follows the same dense response contract as :class:`Gaussian` and
    must lie in ``[0, 1]``. Fractional responses represent proportions with
    equal observation weights. The intercept is unpenalized. Fitted
    predictions are logits, including when fitting with hybrid IRLS.
    """

    y: object = field(repr=False)
    fit_intercept: bool = field(default=True, kw_only=True)
