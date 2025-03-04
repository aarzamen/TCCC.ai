"""
TCCC.ai - Transcription, Compliance, and Customer Care AI System
Setup configuration for package installation
"""

import os
from setuptools import setup, find_packages

# Get long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Get version from package __init__.py
with open(os.path.join("src", "tccc", "__init__.py"), "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"').strip("'")
            break
    else:
        version = "0.1.0"  # Default if not found

# Define package requirements
requirements = [
    # Core dependencies
    "numpy>=1.20.0",
    "pyyaml>=6.0",
    "pydantic>=2.0.0",
    "python-dotenv>=1.0.0",
    
    # Audio processing
    "sounddevice>=0.4.5",
    "librosa>=0.10.0",
    "pyaudio>=0.2.13",
    "webrtcvad>=2.0.10",
    "noisereduce>=2.0.1",
    
    # Machine learning and inference
    "torch>=2.0.0",
    "torchaudio>=2.0.0",
    "transformers>=4.30.0",
    "onnxruntime-gpu>=1.15.0",
    "tensorrt>=8.6.0",
    
    # Speech recognition
    "openai-whisper>=20231117",
    "pyannote.audio>=3.0.0",
    "speechbrain>=0.5.14",
    
    # NLP and text processing
    "spacy>=3.6.0",
    "sentence-transformers>=2.2.2",
    "nltk>=3.8.1",
    
    # Database and storage
    "sqlalchemy>=2.0.0",
    "redis>=4.6.0",
    "pymongo>=4.4.1",
    
    # Web and API
    "fastapi>=0.100.0",
    "uvicorn>=0.22.0",
    "websockets>=11.0.3",
    
    # Testing
    "pytest>=7.3.1",
    "pytest-asyncio>=0.21.0",
]

# Define development requirements
dev_requirements = [
    "black>=23.3.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.3.0",
    "pytest-cov>=4.1.0",
    "pre-commit>=3.3.2",
]

setup(
    name="tccc",
    version=version,
    author="TCCC Team",
    author_email="dev@tccc.ai",
    description="Transcription, Compliance, and Customer Care AI System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tccc-ai/tccc-project",
    project_urls={
        "Documentation": "https://tccc.ai/docs",
        "Bug Tracker": "https://github.com/tccc-ai/tccc-project/issues",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "audio": [
            "pyworld>=0.3.3",
            "pyrubberband>=0.3.0",
            "soundfile>=0.12.1",
        ],
        "gpu": [
            "cupy-cuda12x>=12.0.0",
            "nvidia-tensorrt>=8.6.0",
            "onnx-graphsurgeon>=0.3.27",
        ],
        "docs": [
            "sphinx>=6.2.1",
            "sphinx-rtd-theme>=1.2.1",
            "sphinx-autodoc-typehints>=1.23.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: POSIX :: Linux",
    ],
    entry_points={
        "console_scripts": [
            "tccc=tccc.cli:main",
        ],
    },
    include_package_data=True,
)