from setuptools import setup
import os
from glob import *

package_name = "row_crop_follow"

setup(
    name=package_name,
    version="1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Marco Ambrosio",
    maintainer_email="marco.ambrosio@polito.it",
    description="Navigation algorithm to maintain a central path inside the row",
    license="GNU GPLv3",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "seg_controller_node = row_crop_follow.seg_controller_node:main",
            "otsu_thresholding_node = row_crop_follow.otsu_thresholding_node:main",
        ],
    },
)
