from setuptools import setup, find_packages

setup(
    name='model-inspect',
    version='0.0.1',
    author='simpx',
    author_email='simpxx@gmail.com',
    description='Command-line tool for analyzing layer-wise parameters of Hugging Face models including shapes, data types, and memory footprints',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/simpx/model-inspect',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'model-inspect=model_inspect.cli:main',
        ],
    },
)
