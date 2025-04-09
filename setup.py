from setuptools import setup, find_packages

setup(
    name="grouped-kv-attention",
    version="0.1.0",
    description="Custom grouped key-value attention patch for HuggingFace Transformers",
    author="Kevin Kuo",
    author_email="your_email@example.com",  # optional
    packages=find_packages(),
    install_requires=[
        "torch>=1.13.0",
        "transformers>=4.36.0"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
