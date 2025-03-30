from setuptools import find_packages, setup

package_name = 'yoloe_seg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='abivishaq',
    maintainer_email='b.abivishaq@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yoloe_node = yoloe_seg.yoloe_node:main',
            'yoloe_service = yoloe_seg.yoloe_service:main',
            'yoloe_client = yoloe_seg.yoloe_client:main'
        ],
    },
)
