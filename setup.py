from setuptools import setup

INSTALL_REQUIRES = [
    'numpy >= 1.18',
    'pandas >= 1.0.5',
    'tensorflow == 2.4',
    'tensorflow-probability == 0.11',
    'gpbasics >= 1.0.0',
    'gpmretrieval >= 1.0.0',
]

setup(
    name='covariancefingerprint',
    version='1.0.0',
    packages=['CoffeeCapsules', 'NoaaExperiments', 'PerformanceEval', 'FinanceExperiments', 'TMY3SolarExperiments'],
    package_dir={'': 'main'},
    url='URL',
    license='Apache License 2.0',
    author='Fabian Berns',
    author_email='fabian.berns@googlemail.com',
    description='',
    install_requires=INSTALL_REQUIRES,
)