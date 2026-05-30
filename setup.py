"""
Package setup for Deep Research AI System.
"""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="deep-research",
    version="1.0.0",
    author="Deep Research Team",
    description=(
        "Production-grade Deep Research AI with web search, "
        "reasoning models, and inference-time scaling."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.11",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-asyncio>=0.23.0",
            "pytest-cov>=5.0.0",
            "httpx>=0.27.0",
            "black>=24.0.0",
            "ruff>=0.4.0",
            "mypy>=1.9.0",
        ],
        "training": [
            "torch>=2.2.0",
            "transformers>=4.40.0",
            "trl>=0.8.6",
            "peft>=0.10.0",
            "datasets>=2.19.0",
            "bitsandbytes>=0.43.0",
        ],
        "local": [
            "llama-cpp-python>=0.2.57",
            "vllm>=0.4.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "deep-research=scripts.cli:app",
            "deep-research-train=scripts.train:app",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)