"""Setup for DeepECG"""
from setuptools import setup

setup(
    name='deepecg',
    version='0.0.1',
    description='A package to automatically classify ecg waveforms.',
    url='https://github.com/Seb-Good/deepecg.git',
    author='Sebastian D. Goodfellow, Ph.D.',
    license='MIT',
    keywords='deep learning',
    package_dir={'': 'deepecg'},
    zip_safe=False,
)
