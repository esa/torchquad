"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages

setup(
    name="torchquad",
    version="0.1.0",
    description="Package providing a torch-based integration method.",
    url="https://github.com/esa/torchquad",
    author="ESA Advanced Concepts Team",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Integration Method :: Build Tools",
        "License :: OSI Approved :: GNU License",
        "Programming Language :: Python :: 3.8",
    ],
    keywords="integration, setuptools, development",
    package_dir="",
    python_requires=">=3.8, <4",
    project_urls={"Source": "https://github.com/esa/torchquad/",},
)
