from setuptools import setup, find_packages

setup(
    name='conformer_utils',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchaudio'
    ]
)