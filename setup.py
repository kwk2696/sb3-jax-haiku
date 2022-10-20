import os
from setuptools import find_packages, setup

setup(
    name="sb3_jax",
    packages=[package for package in find_packages() if package.startswith("sb3_jax")],
    discription='JAX implementation of stable_baselines3',
)
