#!/usr/bin/env python3
"""
Setup script for miuACP-Tools package.
"""

from setuptools import setup, find_packages
import os

def read_readme():
    """Read README.md file."""
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

def read_requirements():
    """Read requirements.txt file."""
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="miuacp-tools",
    version="1.0.1",
    author="Arnab",
    author_email="hello@arnab.wiki",
    description="µACP Tools: Comprehensive testing, integration, and development suite for the µACP protocol library",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Arnab-m1/miuACP-Tools",
    project_urls={
        "Bug Tracker": "https://github.com/Arnab-m1/miuACP-Tools/issues",
        "Documentation": "https://github.com/Arnab-m1/miuACP-Tools#readme",
        "Source Code": "https://github.com/Arnab-m1/miuACP-Tools",
        "Core Library": "https://github.com/Arnab-m1/miuACP",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Communications",
        "Topic :: Internet :: Name Service (DNS)",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Networking",
        "Topic :: System :: Distributed Computing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Topic :: Software Development :: Testing",
        "Topic :: Software Development :: Quality Assurance",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "full": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "pandas>=1.3.0",
            "numpy>=1.21.0",
            "rich>=13.0.0",
            "click>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "miuacp-analyze=cli:analyze",
            "miuacp-benchmark=cli:benchmark",
            "miuacp-compare=cli:compare",
            "miuacp-demo=cli:demo",
            "miuacp-test=cli:test_uacp",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "agent", "communication", "protocol", "micro", "lightweight", "edge", "iot",
        "multi-agent", "distributed", "messaging", "pubsub", "rpc", "testing",
        "benchmarking", "analysis", "visualization", "integration", "tools"
    ],
    platforms=["any"],
    license="MIT",
    license_files=["LICENSE"],
)
