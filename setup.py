from setuptools import find_packages, setup

setup(
    name="pepNmemb",  # Package name
    version="0.1.0",  # Version number
    author="Miruna S",
    description="Suite of scripts to analyse molecular dynamics "
    "(MD) simulations of peptide-membrane systems",
    long_description=open("README.md").read(),  # Read long description from a file
    long_description_content_type="text/markdown",
    packages=find_packages(),  # Automatically find and include all packages
    install_requires=[
        "numpy",  # List dependencies
        "requests",
    ],
    classifiers=[  # Optional metadata
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",  # Minimum Python version
)
