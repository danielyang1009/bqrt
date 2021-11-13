# Basic Quantitative Research Toolbox (BQRT)

This package is built to assist my daily quantitative research, containing research areas of:

- Term structure/fixed income
- Option pricing
- ...

## TODO

- Newey-West 1987 t-statistic (adjust for serial correlation)
- Implied jump risk (Booleslev and Todorov 2011)
- Implied skewnesss (Bakshi Kapadia and Madan 2003)
- Volatility spread (Yan 2011)
- Idiosyncratic volatility (Cao and Hand 2013)
- Slop of the volatility term structure (Vasquez 2017)
- ...

## Working with time series models

Lists of python packages maybe used in time series analysis

[Using python to work with time series data](https://github.com/MaxBenChrist/awesome_time_series_in_python)

[pyflux documentations](https://pyflux.readthedocs.io/en/latest/index.html)

## Quick Links

[Shibor data download](http://www.shibor.org/shibor/web/DataService.jsp)

## Using jupyter notebook `autoreload`

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

或者

```python
py -m pip install -e path/to/SomeProject
```
