from setuptools import setup, find_packages

setup(
    name="opendataval-finegrained",
    version="1.3.0",
    description="Fine-grained data valuation with modified opendataval",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.8.0",
        "transformers>=4.30.0",
        "tqdm>=4.60.0",
        "requests>=2.28.0",
        "ipython>=7.0.0",
        "geomloss>=0.2.6",
        "pykeops>=2.1.0",
    ],
)