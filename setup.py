from setuptools import setup, find_packages

setup(
    name="document-classifier",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Document Classification and Data Extraction System",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/document-classification-extraction",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        "streamlit>=1.22.0",
        "pillow>=9.5.0",
        "numpy>=1.24.3",
        "pandas>=2.0.2",
        "scikit-learn>=1.2.2",
        "tensorflow>=2.12.0",
        "pytesseract>=0.3.10",
        "pdf2image>=1.16.3",
        "opencv-python>=4.7.0.72",
        "matplotlib>=3.7.1",
        "easyocr>=1.7.0"
    ],
) 