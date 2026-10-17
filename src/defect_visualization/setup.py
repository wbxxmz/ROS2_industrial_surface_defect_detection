from setuptools import find_packages, setup

package_name = 'defect_visualization'

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
    description='结果可视化、判定与统计',
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'visualize_node = defect_visualization.visualize_node:main',
        ],
    },
)
