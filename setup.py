import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="taco",
    version="0.0.1",
    author="Wenzheng Zhang",
    author_email="vincentzustc@gmail.com",
    description="multi-task retrieval",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Text Processing :: Indexing",
        "Intended Audience :: Information Technology"
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=[
        "transformers>=4.21.3",
        "datasets>=2.10.1",
        "sentencepiece",
    ]
)