from os import path
from setuptools import setup, find_packages


def get_version():
    init_py_path = path.join(
        path.abspath(path.dirname(__file__)), "cvgear", "__init__.py"
    )
    init_py = open(init_py_path, "r").readlines()
    version_line = [l.strip() for l in init_py if l.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")
    return version


setup(
    name="cvgear",
    version=get_version(),
    author="ivan Ding",
    license="MIT",
    url="https://github.com/ivanpp/cvgear",
    description="Computer Vision Gears",
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "yacs>=0.1.6",
    ],
    packages=find_packages(),
)

