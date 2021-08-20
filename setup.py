"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages

setup(
    name="torchquad",
    version="0.2.3",
    description="Package providing torch-based numerical integration methods.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/esa/torchquad",
    author="ESA Advanced Concepts Team",
    author_email="pablo.gomez@esa.int",
    install_requires=[
        "loguru>=0.5.3",
        "matplotlib>=3.3.3",
        "scipy>=1.6.0",
        "tqdm>=4.56.1",
        "torch>=1.7.1",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.8",
    ],
    packages=[
        "torchquad",
        "torchquad.integration",
        "torchquad.plots",
        "torchquad.utils",
    ],
    python_requires=">=3.8, <4",
    project_urls={
        "Source": "https://github.com/esa/torchquad/",
    },
)
