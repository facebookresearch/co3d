# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import setuptools

setuptools.setup(
    name="co3d",
    version="2.1.0",
    author="FAIR",
    author_email="dnovotny@fb.com",
    packages=setuptools.find_packages(exclude=["tests", "examples"]),
    license="LICENSE",
    description="Common Objects in 3D codebase",
    long_description=open("README.md").read(),
    install_requires=[
        "numpy",
        "Pillow",
        "requests",
        "tqdm",
        "plyfile",
    ],
)
