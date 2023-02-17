from setuptools import setup

setup(
    name='multifuzz',
    version='0.0.1',
    install_requires=[
        "pandas~=1.4.4",
        "rapidfuzz~=2.6.1"
    ],
    extras_require={
        "dev": [
            "pytest"
        ]
    },
    include_dirs=["src"]
)