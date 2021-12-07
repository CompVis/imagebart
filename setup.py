from setuptools import setup, find_packages

setup(
    name='imagebart',
    version='0.0.1',
    description='autoregressive image modification via multinomial diffusion',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)
