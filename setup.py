from setuptools import find_packages, setup

setup(
    name="softmaxshield",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "softmaxshield=softmaxshield.cli:main",
        ],
    },
)

