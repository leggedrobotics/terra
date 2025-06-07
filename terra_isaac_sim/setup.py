from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'terra_isaac_sim'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Stanislaw Piasecki',
    maintainer_email='spiasecki@ethz.ch',
    description='Package for deploying Terra plans in IsaacSim',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'nav2_plan_executor = nav2_plan_executor:main',
        ],
    },
)
