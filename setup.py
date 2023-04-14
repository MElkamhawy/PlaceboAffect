from setuptools import find_packages, setup

setup(
    name='src'
    , packages=find_packages()
    , install_requires = [
        'numpy'
        , 'pandas'
        , 'scikit-learn'
    ]
    , version='0.1.0'
    , description='This project implements a HatEval binary classifier'
    , author=['mohamed', 'nora', 'alex', 'karl']
    , author_email=["author1@example.com", "author2@example.com", "author3@example.com"]
)