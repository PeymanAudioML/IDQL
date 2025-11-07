#!/usr/bin/env python
"""
Setup script for JAXRL5-PyTorch
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="idql",
    version="1.0.0",
    author="PeymanAudioML",
    description="PyTorch implementation of IDQL (Implicit Diffusion Q-Learning)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PeymanAudioML/IDQL",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "torch>=1.9.0",
        "gym==0.18.0",
        "tqdm>=4.62.0",
        "einops>=0.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.6b0",
            "flake8>=3.9.0",
        ],
        "d4rl": [
            "mujoco-py==2.0.2.13",
            "d4rl @ git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl",
        ],
        "wandb": [
            "wandb>=0.12.0",
        ],
    },
)
