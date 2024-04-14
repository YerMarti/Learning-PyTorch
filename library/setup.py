from setuptools import setup, find_packages

# Define package metadata
setup(
    name="pthfunctions",
    version="0.1.0",
    description="A library containing plotting and training functionalities",
    author="Yeray Martínez",
    packages=find_packages(where="lib"),
    include_package_data=True,
    install_requires=[
        "torch",
        "tqdm.auto",
        "pandas",
        "matplotlib.pyplot",
    ],
)