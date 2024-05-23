from setuptools import setup, find_packages

setup(
    name='chronocluster',
    version='0.1',
    packages=find_packages(include=['chronocluster', 'chronocluster.*']),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'pointpats',
        'pymc3'
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='A package for temporal uncertainty in spatial point pattern analysis and clustering',
    url='https://github.com/wccarleton/ChronoCluster',
)