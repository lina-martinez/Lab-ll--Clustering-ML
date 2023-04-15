from setuptools import setup, find_packages

dev_required = ["utils",'mypy','numpy', 'pandas', 'scikit-learn','matplotlib','scipy']

setup(
    name="Unsupervised_model",
    version="0.1",
    description="Unsupervised model",
    url="https://github.com/lina-martinez/Lab-I---Dimensionality-Reduction",
    python_requires=">=3.7",
    packages=find_packages(),
    include_package_data=True,
    extras_require={"dev": dev_required},
    package_dir={"": "."},
)
