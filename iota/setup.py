from setuptools import setup, find_packages

setup(
    name="iota",
    version="0.0.1",
    author="Justin Cho",
    author_email="hd.justincho@gmail.com",
    description="Inference-Only Task Alignment with Demonstrated Feedback and Preference Explanations",
    long_description="Repository for experiments with Inference-Only Task Alignment with Demonstrated Feedback and Preference Explanations",
    url="https://github.com/anon",
    packages=find_packages(),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    entry_points={
        "console_scripts": [
            "iota.run_iota=iota.run_iota:main",
            "iota.llm_eval=iota.llm_authorship_eval:main",
        ],
    },
    keywords="alignment, inference, feedback, explanations, preference",
)
