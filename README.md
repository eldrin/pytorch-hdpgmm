# PyTorch-HDPGMM

a `pytorch` implementation of `hierarchical Dirichlet process Gaussian mixture model (HDPGMM)`.


## TODOs

- 1. deployments
    - [x] continuous doc with Travis + Sphinx + Github pages (see [this](https://github.com/icgood/continuous-docs))
        - implemented using `github actions` following [this repo](https://github.com/eeholmes/readthedoc-test)
    - [ ] more documentations - module level docstrings, improving top-level README
    - [ ] PyPI publish

- 2. features
    - [ ] turning into CSR-like layout; 'masked-array-of-variable-sequence' is not worth for the computational efficiency while make the memory efficiency terrible. (especially on GPU)
