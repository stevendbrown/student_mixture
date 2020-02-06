# -*- coding: utf-8 -*-

"""Top-level package for Student's t-mixture model."""

__author__ = """Omri Tomer"""
__email__ = 'omritomer1@mail.tau.ac.il'
__version__ = '0.1.8'

import sys

if (sys.version_info < (3, 3)):
    from student_mixture import StudentMixture
    from multivariate_t import multivariate_t as multivariate_t
    from mutlivariate_t_fit import MultivariateTFit

else:
    from student_mixture.student_mixture import StudentMixture
    from student_mixture.multivariate_t import multivariate_t  as multivariate_t
    from student_mixture.mutlivariate_t_fit import MultivariateTFit

__all__ = ['StudentMixture',
           'multivariate_t',
           'MultivariateTFit']
