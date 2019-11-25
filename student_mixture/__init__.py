# -*- coding: utf-8 -*-

"""Top-level package for Student's t-mixture model."""

__author__ = """Omri Tomer"""
__email__ = 'omritomer1@mail.tau.ac.il'
__version__ = '0.0.1'

import sys
if (sys.version_info < (3, 3)):
    from student_mixture import StudentMixture
else:
    from student_mixture.student_mixture import StudentMixture
