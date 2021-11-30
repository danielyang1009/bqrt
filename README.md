# Basic Quantitative Research Toolbox (BQRT)

This package is built to assist my daily quantitative research, containing research areas of:

- Term structure/fixed income
- Option pricing
- ...

## TODO

- create an option class, method including BSIV, MFIV calculation
- create time series class for (g)arch and do (quasi-)MLE estimate
- Implied jump risk (Booleslev and Todorov 2011)
- Implied skewnesss (Bakshi Kapadia and Madan 2003)
- Volatility spread (Yan 2011)
- Idiosyncratic volatility (Cao and Hand 2013)
- Slop of the volatility term structure (Vasquez 2017)
- ...

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
