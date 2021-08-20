from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
     name='pyicon',
     version='0.1.0',
     description='python library for bias_correction',
     long_description=long_description,
     long_description_content_type='text/markdown',
     url='https://gitlab.dkrz.de/m300602/pyicon',
     author='Nils Brueggemann',
     author_email='nils.brueggemann@uni-hamburg.de',
     install_requires=[
        'numpy',
        'scipy',
        'xarray',
        'cmocean',
        'ipdb',
        'netcdf4',
        'matplotlib',
        'cartopy'
     ],
     setup_requires=['setuptools'],
)
