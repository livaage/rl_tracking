""" To install this package, change to the directory of this file and run

    pip install -e .

(use the ``-e`` flag for development/editable mode).

Note: For full dependency management including git dependencies, use Poetry:
    poetry install
"""

from setuptools import setup, find_packages

setup(
    name='rl_tracking',
    version='0.1.0',
    description='Reinforcement Learning for Particle Tracking',
    author='liv',
    author_email='liv.helen.vage@cern.ch',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=[
        "torch>=2.4.1",
        "lightning>=2.4.0",
        "pandas>=2.2.2",
        "wandb>=0.19.3",
        "gymnasium>=1.0.0",
        "numba>=0.61.0",
        "scikit-learn>=1.5.2",
    ],
    python_requires='>=3.12',
)
