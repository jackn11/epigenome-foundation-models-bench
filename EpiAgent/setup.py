from setuptools import setup, find_packages

setup(
    name="epiagent",
    version="0.0.3",
    description="Foundation model for single-cell epigenomic data.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Xiaoyang Chen",
    author_email="xychen20@mails.tsinghua.edu.cn",
    url="https://github.com/xy-chen16/EpiAgent",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy==1.26.4",
        "pandas==2.2.3",
        "scipy==1.16.3",
        "scikit-learn==1.7.2",
        "scanpy==1.11.5",
        "transformers==4.35.2",
        "anndata==0.12.6",
        "POT>=0.9.2",
        # "faiss", # need to conda install this when using pip install -e .
        "flash-attn==2.5.8",  # Flash-Attention: Installation details in README.md
        "leidenalg==0.11.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    extras_require={
        "gpu": ["torch>=2.0.0", "flash-attn>=2.5.7"],
    },
)
