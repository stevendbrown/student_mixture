#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('requirements.txt') as f:
    requirements = [line.strip()
                    for line in f
                    if line.strip() and not line.startswith('#')]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Omri Tomer",
    author_email='omritomer1990@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
    description="A package for fitting a Student's t-mixture model",
    install_requires=requirements,
    license="MIT license",
    long_description_content_type='text/markdown',
    long_description=readme,
    include_package_data=True,
    keywords='student_mixture',
    name='student_mixture',
    packages=find_packages(include=['student_mixture', 'student_mixture.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/omritomer/student_mixture',
    version='0.2.0',
    zip_safe=False,
)
