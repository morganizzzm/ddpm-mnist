"""Setup script for DDPM-MNIST package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ddpm-mnist",
    version="1.0.0",
    author="Margo Levin",
    author_email="morganizzzm@gmail.com",
    description="A PyTorch implementation of DDPM for MNIST",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/morganizzzm/ddpm-mnist",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ddpm-train=train:main",
        ],
    },
)
