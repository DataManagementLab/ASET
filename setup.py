from setuptools import setup

setup(
    name="aset",
    version="0.0",
    py_modules=["aset"],
    install_requires=[
        "pymongo==3.12.0",
        "torch==1.9.1",
        "numpy==1.21.2",
        "pandas==1.3.3",
        "scipy==1.7.1",
        "stanza==1.3.0",
        "spacy==3.1.3",
        "sentence-transformers==2.1.0",
        "matplotlib==3.4.3",
        "seaborn==0.11.2",
        "scikit-learn==1.0",
        "pytest==6.2.5",
        "transformers==4.9.2",
        "PyQt6==6.2.0",
        "flake8==4.0.1"
    ]
)
