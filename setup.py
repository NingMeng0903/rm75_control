from setuptools import find_packages, setup

setup(
    name="rm75-control",
    version="0.1.0",
    description="RealMan RM75 integrated controller wrapper",
    packages=find_packages(include=["rm75_control*"]),
    python_requires=">=3.9",
    install_requires=["numpy", "pyyaml", "ruckig==0.17.3"],
)
