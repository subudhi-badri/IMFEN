from setuptools import setup, find_packages

setup(
    name="mfen",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.7.0",
        "torchvision>=0.8.0",
        "matplotlib>=3.3.0",
        "tensorboardX>=2.0",
        "opencv-python>=4.5.0",
        "lpips>=0.1.4",
        "tqdm>=4.50.0",
        "fvcore>=0.1.5",
        "Pillow>=8.0.0",
        "numpy>=1.19.0"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Multi-Feature Enhancement Network for underwater image enhancement",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mfen",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
) 