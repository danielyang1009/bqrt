#!/usr/bin/env python
# encoding: utf-8

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bqrt",
    version="0.0.2",
    author="Daniel Yang",
    description="Basic Quantitative Research Toolbox",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danielyang1009/bqrt",
    project_urls={
        "Bug Tracker": "https://github.com/danielyang1009/bqrt/issues",
    },
    install_requires=['numpy', 'pandas'],
    packages=setuptools.find_packages(exclude=['docs', 'tests']),
    python_requires=">=3.6",
)