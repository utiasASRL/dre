from setuptools import setup

package_name = "dre"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=[
        "setuptools",
        "numpy",
        "pandas",
        "pyboreas",
    ],
    zip_safe=True,
    maintainer="Cedric Le Gentil",
    maintainer_email="le.gentil.cedric@gmail.com",
    description="ROS 2 Python package for Dr-PoGO",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "boreas_player = dre.boreas_player:main",
            "dro_node = dre.dro.dro_node:main",
            "raplace_node = dre.raplace.raplace_node:main",
            "registration_node = dre.raplace.registration_node:main",
        ],
    },
)
