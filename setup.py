#!/usr/bin/env python

from io import open
from setuptools import setup

"""
:authors: zaithevalex
:license: Apache License, Version 2.0, see LICENSE file
:copyright: (c) 2025 zaithevalex
"""

version = '1.0.9'

with open('README.md', encoding = 'utf-8') as f:
    long_description = f.read()

setup(
    name = 'minplus_algebra',
    version = version,

    author = 'zaithevalex',
    author_email = 'zaithevalex@gmail.com',

    description = (
        u'Python module for minplus algebra.'
    ),
    long_description = long_description,
    long_description_content_type = 'text/markdown',

    url = 'https://github.com/zaithevalex/minplus_algebra',
    download_url = 'https://github.com/zaithevalex/minplus_algebra/archive/main.zip',

    license = 'Apache License, Version 2.0, see LICENSE file',

    packages = ['minplus_algebra'],
    install_requires = ['numpy', 'pandas', 'scipy'],

    classifiers = [
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Programming Language :: Python :: Implementation :: CPython',
    ]
)