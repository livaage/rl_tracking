""" To install this package, change to the directory of this file and run

    pip3 install --user .

(the ``--user`` flag installs the package for your user account only, otherwise
you will need administrator rights).
"""


from setuptools import setup, find_packages

setup(
    name='rl_tracking',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=[],  # Add your dependencies here
)
