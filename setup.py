from pathlib import Path
from setuptools import find_packages, setup

# Load version number
__version__ = None

src_dir = Path(__file__).parent.absolute()
version_file = src_dir / 'p_tqdm' / '_version.py'

# Long README
with open(version_file) as fd:
    exec(fd.read())

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='p_tqdm',
    version='1.5.0',
    author='Kyle Swanson',
    author_email='swansonk.14@gmail.com',
    description='Parallel processing with progress bars',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/swansonk14/p_tqdm',
    download_url='https://github.com/swansonk14/p_tqdm/v_1.3.3.tar.gz',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'tqdm>=4.45.0',
        'joblib>=1.3',
        'six>=1.13.0'
    ],
    tests_require=['pytest'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    keywords=[
        'tqdm',
        'progress bar',
        'parallel'
    ],
)
