from setuptools import find_packages, setup

package_name = 'controllers'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/pid_bridge_gazebo.launch.py']),
        ('share/' + package_name + '/launch', ['launch/pid_obs_bridge_gazebo.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shahab',
    maintainer_email='shokohishahab@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pid = controllers.pid:main',
            'pid_obs_avoidance = controllers.pid_obs_avoidance:main',
        ],
    },
)
