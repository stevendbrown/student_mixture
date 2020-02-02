# -*- coding: utf-8 -*-

"""Top-level package for Student's t-mixture model."""

__author__ = """Omri Tomer"""
__email__ = 'omritomer1@mail.tau.ac.il'
__version__ = '0.1.2'

import sys
from .student_mixture import StudentMixture
from .multivariate_t import multivariate_t as MultivariateT
from .mutlivariate_t_fit import MultivariateTFit
__all__ = ['StudentMixture',
           'MultivariateT',
           'MultivariateTFit']
