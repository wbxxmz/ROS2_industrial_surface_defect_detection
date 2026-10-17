from glob import glob

from setuptools import find_packages, setup

package_name = 'defect_core'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wbxxmz',
    maintainer_email='17625362+wbxxmz@user.noreply.gitee.com',
    description='统一检测接口、算法注册与检测流水线',
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'pipeline_node = defect_core.pipeline_node:main',
        ],
    },
)
