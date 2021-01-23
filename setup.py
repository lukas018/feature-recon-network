#!/usr/bin/env python3
import setuptools

with open('README.org', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='foi_fewshot',  # Replace with your own username
    version='0.0.1',
    author='Lukas Lundmark',
    author_email='lukas.lundmark@foi.se',
    description='A package for quickly training/testing and benchmarking common fewshot methods',
    long_description=long_description,
    long_description_content_type='text/org',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
