from setuptools import setup, find_packages
import os

setup(
    name="dsr_project",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21",
        "matplotlib>=3.4",
        "scipy>=1.7",
    ],
    author="Cina Aghamohammadi",
    author_email="caghamohammadi@gmail.com",
    description="Doubly Stochastic Renewal Process implementation",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    license="MIT",
)
