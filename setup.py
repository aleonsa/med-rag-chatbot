from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="LLMOps_course",
    version="0.1.0",
    author="Alejandro Leon",
    packages=find_packages(),
    install_requires=requirements,
)
