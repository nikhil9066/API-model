"""
setup.py
Package setup for AutoML Phase 1 Pipeline
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="automl-phase1",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AutoML Phase 1 - All-Numeric Regression Pipeline",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/API-model",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Data Scientists",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "isort>=5.10.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
        ],
        "visualization": [
            "plotly>=5.10.0",
            "seaborn>=0.11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "automl=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "tests/sample_data/*"],
    },
    zip_safe=False,
    keywords="automl, machine learning, regression, feature engineering, model selection",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/API-model/issues",
        "Source": "https://github.com/yourusername/API-model",
        "Documentation": "https://github.com/yourusername/API-model/wiki",
    },
)