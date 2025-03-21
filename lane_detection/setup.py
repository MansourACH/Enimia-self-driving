from setuptools import setup
import os
from glob import glob

package_name = 'lane_detection'
model_file = os.path.join('models', 'best.pt')


setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
	(os.path.join('share', package_name, 'models'), [model_file]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wafa',
    maintainer_email='wafa.ammar75@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
           # 'lane_detection_node = lane_detection.lane_detection_node:main',
            'lane_detection_node = lane_detection.lane_detection_node:main',
            'stop_sign_node = lane_detection.stop_sign_node:main'  
                            ],
    },
)
