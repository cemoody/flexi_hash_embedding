from setuptools import setup, find_packages

__version__ = '0.0.2'
url = 'https://github.com/cemoody/flexi_hash_embedding'

install_requires = ['torch', 'torch-scatter', 'sklearn']
tests_require = ['pytest']

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='flexi_hash_embedding',
    version=__version__,
    description='PyTorch Extension Library of Optimized Scatter Operations',
    author='Christopher Moody',
    author_email='chrisemoody@gmail.com',
    url=url,
    download_url='{}/archive/{}.tar.gz'.format(url, __version__),
    keywords=[
        'pytorch',
        'scatter',
        'groupby',
        'embedding',
        'hashing',
        'variable',
        'fixed'
    ],
    install_requires=install_requires,
    tests_require=tests_require,
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
