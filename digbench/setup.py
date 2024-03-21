# setup.py file for the excavation_benchmark package

from setuptools import setup, find_packages

package_name = 'digbench'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    py_modules=[
        'digbench',
    ],
    install_requires=['setuptools']
)
