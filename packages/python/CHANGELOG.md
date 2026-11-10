# Changelog

## [0.3.0](https://github.com/jolars/shrinkage/compare/shrinkage-python-v0.2.0...shrinkage-python-v0.3.0) (2026-11-10)

### Breaking changes

- rename `Hybrid` to `HybridCoordinateDescent` ([`3c34886`](https://github.com/jolars/shrinkage/commit/3c348866f288534a0ddb94e9d6e8d0afbce99b95))
- support configurable GLM links ([`4f38932`](https://github.com/jolars/shrinkage/commit/4f38932f1b9473ec6269c7afb5570c46296b716c))

### Features

- **python:** bind Gaussian and logistic paths ([`4282b34`](https://github.com/jolars/shrinkage/commit/4282b34066d389093b170fd1919ef1adcfc821d0))
- support configurable GLM links ([`4f38932`](https://github.com/jolars/shrinkage/commit/4f38932f1b9473ec6269c7afb5570c46296b716c))
- upgrade to `lazymatrix` 0.10.0 ([`f283dec`](https://github.com/jolars/shrinkage/commit/f283decc1329dae6d6cfc2cdaa2b777c781061e0))
- **python:** add hybrid fitting and prediction ([`7bd8c6c`](https://github.com/jolars/shrinkage/commit/7bd8c6c5f3a3814551f8edf481d198914c912f13))
- add default working-set policies ([`3404fb2`](https://github.com/jolars/shrinkage/commit/3404fb2e3f092ed2e0e861d31a09c110c424a7aa))

### Performance Improvements

- accelerate dense Python lasso fits ([`3dd3c09`](https://github.com/jolars/shrinkage/commit/3dd3c095547f9893c429646a991323e870eb0d9a))
- share KKT and dual correlations ([`800af43`](https://github.com/jolars/shrinkage/commit/800af43f060265e31871411284e5dc283725942d))
- batch coordinate descent convergence checks ([`7eef81a`](https://github.com/jolars/shrinkage/commit/7eef81a7b5b92878c9adc488edc58ba6a0720cd1))

## [0.2.0](https://github.com/jolars/shrinkage/compare/shrinkage-python-v0.1.0...shrinkage-python-v0.2.0) (2026-10-23)

### Features

- **python:** add Gaussian lasso fitting ([`81d67ad`](https://github.com/jolars/shrinkage/commit/81d67ad88b68f37eb845da0b7e702d64076139d8))
- validate dense Python inputs ([`9d073c5`](https://github.com/jolars/shrinkage/commit/9d073c56575c758e67f8d8914c3ddf325dec1c77))
- scaffold Python bindings and shared docs ([`5d844d5`](https://github.com/jolars/shrinkage/commit/5d844d5b561157ca8f7a647fbbf79611892ac10f))

### Bug Fixes

- use typed memoryview in Python input test ([`70e6182`](https://github.com/jolars/shrinkage/commit/70e61827f3bda3e8969bdc082023819681b24d39))
