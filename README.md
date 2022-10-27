# Basic Quantitative Research Toolbox (BQRT)

This package is built to assist my daily quantitative research, containing research areas of:

- Asset pricing
- Option pricing
- Term structure/fixed income
- ...

## TODO

### Asset Pricing

Portfolio sort

- [ ] univariate sort
- [ ] bivariate sort (independent or conditional)
- [ ] porfolio return (equal-weighted or value-weighted)

CS regression

- [ ] two-step, first step ts regression; second step, get ts average and run cs only once

Tenical

- [ ] rolling and groupby apply acceleration (solving FM regression rolling beta)

FM regression

- [ ] Rolling beta (try run in parallel, or any acceleration)

## Notes

### Using jupyter notebook `autoreload`

[IPython `autoreload` Reference](https://ipython.readthedocs.io/en/stable/config/extensions/autoreload.html)

This allow to use newest version of module as testing.

```python
%load_ext autoreload
%autoreload 2
```

### Install package locally

To bqrt folder, or create a environment, in editable mode or develop mode

```python
pip install -e .
```

or

```python
py -m pip install -e path/to/SomeProject
```
