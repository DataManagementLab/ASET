from setuptools import setup


setup(
    name="aset",
    version="0.0",
    py_modules=["aset"],
    install_requires=[
        "torch==1.8.0",
        "numpy==1.19.0",
        "pandas==1.2.3",
        "scipy==1.6.1",
        "matplotlib==3.3.4",
        "transformers==4.3.3",
        "sentence-transformers==0.4.1.2",
        "stanza==1.2",
    ],
)
