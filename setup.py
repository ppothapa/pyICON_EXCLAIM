from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    install_requires = f.read().strip().split("\n")

setup(
     name='pyicon',
     version='0.1.0',
     description='Diagnostic python software package for ICON',
     long_description=long_description,
     long_description_content_type='text/markdown',
     url='https://gitlab.dkrz.de/m300602/pyicon',
     author='Nils Brueggemann',
     author_email='nils.brueggemann@uni-hamburg.de',
     install_requires=install_requires,
     setup_requires=['setuptools'],
)
