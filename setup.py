import pathlib
from setuptools import setup, find_packages
setup(
    name="climbinterp",
    version="0.1.1",
    description="Complex explosion and flammable data interpolation!",
    long_description=pathlib.Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    author="Jonathan Motta",
    author_email="jonathangmotta98@gmail.com",
    license="The Unlicense",
    project_urls={
        "Source": "https://github.com/Safe-Solutions-Engenharia/climb_interp"
    },
    classifiers=[
        "Intended Audience :: Developers",
        "Development Status :: 3 - Alpha",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">= 3.9, <3.12",
    install_requires=[
        "scipy",
        "matplotlib",
        "cvxpy",
    ],
    packages=find_packages(),
    include_package_data=True,
)