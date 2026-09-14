from setuptools import find_packages, setup
setup(name="qgloc", version="1.0.0", packages=find_packages(exclude=("tests",)),
      python_requires=">=3.10",
      install_requires=["pyteda==0.5.0", "numpy>=1.26,<3", "scipy>=1.11",
                        "pandas>=2.0", "matplotlib>=3.7"])
