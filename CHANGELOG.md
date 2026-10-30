# Changelog

## [0.4.0](https://github.com/jolars/shrinkage/compare/v0.3.0...v0.4.0) (2026-10-30)

### Features

- **python:** add hybrid fitting and prediction ([`7bd8c6c`](https://github.com/jolars/shrinkage/commit/7bd8c6c5f3a3814551f8edf481d198914c912f13))
- add cross-validation with warm-start paths ([`6f70587`](https://github.com/jolars/shrinkage/commit/6f705874351453b2ce23b21eb5422abaf5838a7a))
- lower MSRV to 1.84 ([`97500f6`](https://github.com/jolars/shrinkage/commit/97500f69b4e7422bcd8423821f2d65989feec2fe))
- add warm-started paths for all solvers ([`d1a94e7`](https://github.com/jolars/shrinkage/commit/d1a94e79a0b3fefe3ee49f05696fe534d76d1ceb))
- add default working-set policies ([`3404fb2`](https://github.com/jolars/shrinkage/commit/3404fb2e3f092ed2e0e861d31a09c110c424a7aa))
- add dynamic Gap Safe zero screening ([`71f5ec9`](https://github.com/jolars/shrinkage/commit/71f5ec948e974a7887ef3e7f2286fe81eb0d2616))

## [0.3.0](https://github.com/jolars/shrinkage/compare/v0.2.0...v0.3.0) (2026-10-27)

### Features

- add logistic IRLS to hybrid solvers ([`fc40420`](https://github.com/jolars/shrinkage/commit/fc40420e1544a26bc26809bc45a6b35623b910ae))
- add SLOPE hybrid fitter ([`9e6d8ec`](https://github.com/jolars/shrinkage/commit/9e6d8ec3c66e9e9fe3b2a0d6456207c17798c77d))
- add chain fused lasso and hybrid solver ([`df1f5a5`](https://github.com/jolars/shrinkage/commit/df1f5a51bfe69cde7b04dbef67d79a551e6c54de))
- add clustered lasso and hybrid solver ([`a8c5007`](https://github.com/jolars/shrinkage/commit/a8c5007e0959cc8bc6656330e98e337739a2451d))

## [0.2.0](https://github.com/jolars/shrinkage/compare/v0.1.0...v0.2.0) (2026-10-23)

### Features

- add coordinate-descent IRLS ([`01df860`](https://github.com/jolars/shrinkage/commit/01df8601013aa17b2424d82782b0e01df88bcf71))
- add disjoint Group LASSO ([`d0446bc`](https://github.com/jolars/shrinkage/commit/d0446bc7eaf7302c7c79053c7bf915cb3718a91a))
- **python:** add Gaussian lasso fitting ([`81d67ad`](https://github.com/jolars/shrinkage/commit/81d67ad88b68f37eb845da0b7e702d64076139d8))
- validate dense Python inputs ([`9d073c5`](https://github.com/jolars/shrinkage/commit/9d073c56575c758e67f8d8914c3ddf325dec1c77))
- scaffold Python bindings and shared docs ([`5d844d5`](https://github.com/jolars/shrinkage/commit/5d844d5b561157ca8f7a647fbbf79611892ac10f))
- add validated disjoint groups ([`b719c8e`](https://github.com/jolars/shrinkage/commit/b719c8e793109ffb84e32e543eda9ba9b3d7f6de))
- add unpenalized coefficient masks ([`a6230b6`](https://github.com/jolars/shrinkage/commit/a6230b61616b48fea1109a3e5bf6b037587c2c21))
- add scaled penalty composition ([`bbd0c81`](https://github.com/jolars/shrinkage/commit/bbd0c81e534623a67999296cbf4c0d328a34be08))
- add regularization path fitting ([`7915a1c`](https://github.com/jolars/shrinkage/commit/7915a1c6f1334f9a7a02b2f71f3bd4596aa901a0))
- add exact SLOPE proximal map ([`c7201b1`](https://github.com/jolars/shrinkage/commit/c7201b16c147cc82008bede3721fb648f23e945e))
- add multivariate Gaussian fitting ([`021ba1c`](https://github.com/jolars/shrinkage/commit/021ba1c901e177a444b9b9ec1dda0f6d1e236fc4))
- add binomial proximal regression ([`5a0e4f2`](https://github.com/jolars/shrinkage/commit/5a0e4f22e624d117878b9e8d18c4e678f72326bd))
- reuse proximal workspaces across penalty paths ([`a4bf63e`](https://github.com/jolars/shrinkage/commit/a4bf63ef847dc083a5788ecd85d1683355087900))
- report unsupported runtime combinations ([`bfc48e3`](https://github.com/jolars/shrinkage/commit/bfc48e39f0c78baaa27bc1c19e2bc37be22fbe52))
- add runtime proximal composition ([`59bf09b`](https://github.com/jolars/shrinkage/commit/59bf09b1ca2d4d546367295c1b92fc86ded019db))
- add buffered proximal fitting ([`165cd6e`](https://github.com/jolars/shrinkage/commit/165cd6e64c42c00bec15d09388d4c19793da4be4))
- add proximal gradient, ridge, and elastic net ([`d2bd583`](https://github.com/jolars/shrinkage/commit/d2bd58358ef0c345d5368561d7467afa2d1a306c))
- add Gaussian lasso duality gap ([`795081b`](https://github.com/jolars/shrinkage/commit/795081bcef23aa6eea71e566fb75316d442cfacd))
- add typed Gaussian lasso composition ([`f2db7e7`](https://github.com/jolars/shrinkage/commit/f2db7e77b281e819f49241a059656a5658867a83))
- validate supplied normalization vectors ([`ef70ccd`](https://github.com/jolars/shrinkage/commit/ef70ccd06182092c573870cc17239c5a48d67144))
- add versioned backend features ([`7188cd4`](https://github.com/jolars/shrinkage/commit/7188cd4714895d350a954c5582e1d2c8262d7aee))
- support `ndarray` and `sprs` backends ([`b3e772e`](https://github.com/jolars/shrinkage/commit/b3e772ea36b46ce86e60c4fc1f718b5dc132abd4))
- use `normalize()` ([`6008259`](https://github.com/jolars/shrinkage/commit/6008259a04a84d7c347c47fddfbb9bab92fe2420))
- add the lasso ([`dd69c1b`](https://github.com/jolars/shrinkage/commit/dd69c1b18a7854ccbb99026d88d2dd7f08038b18))
- scaffold project ([`7f05b01`](https://github.com/jolars/shrinkage/commit/7f05b01ad1f8767b2cc0a2c1d88691d56e0aa693))

### Bug Fixes

- **ci:** install Python for the MSRV build ([`1b21d79`](https://github.com/jolars/shrinkage/commit/1b21d7936b7893fda17acf0889c21ea379722ca8))
- remove height to fix rendering bug ([`80c44f4`](https://github.com/jolars/shrinkage/commit/80c44f49485ad0f30da71f73d70251ab7d163c73))

### Performance Improvements

- add lasso allocation and heap benchmarks ([`56227dc`](https://github.com/jolars/shrinkage/commit/56227dce81c115a8c5c6344eaa13bb26666579f0))

This file is managed by Versionary.
