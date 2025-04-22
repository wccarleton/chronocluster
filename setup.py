from setuptools import setup, find_packages

setup(
    name='chronocluster',
    version='0.1.0',
    packages=find_packages(include=['chronocluster', 'chronocluster.*']),
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
        'scikit-image',
        'matplotlib',
        'pointpats',
        'pymc',
        'pandas',
        'geopandas',
        'arviz',
        'seaborn'
    ],
    author='W. Christopher Carleton',
    author_email='ccarleton@protonmail.com',
    description='A package for temporality and chronological uncertainty in spatial point pattern analysis and clustering',
    url='https://github.com/wccarleton/ChronoCluster',
)