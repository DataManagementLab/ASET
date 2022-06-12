from setuptools import setup

setup(
    name="aset",
    version="0.0",
    py_modules=["aset"],
    install_requires=[
        "pymongo==3.12.1",
        "torch==1.10.0",
        "numpy==1.21.4",
        "pandas==1.3.4",
        "scipy==1.7.2",
        "stanza==1.3.0",
        "spacy==3.2.0",
        "sentence-transformers==2.1.0",
        "matplotlib==3.5.0",
        "seaborn==0.11.2",
        "scikit-learn==1.0.1",
        "transformers==4.12.5",
        "PyQt6==6.2.1",
        "sqlparse==0.4.2"
    ]
)
