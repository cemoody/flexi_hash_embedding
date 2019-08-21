from setuptools import setup, find_packages

__version__ = '0.0.1'
url = 'https://github.com/cemoody/torch_hash_embedding'

install_requires = ['torch', 'torch-scatter', 'sklearn']
tests_require = ['pytest']

setup(
    name='torch_hash_embedding',
    version=__version__,
    description='PyTorch Extension Library of Optimized Scatter Operations',
    author='Christopher Moody',
    author_email='chrisemoody@gmail.com',
    url=url,
    download_url='{}/archive/{}.tar.gz'.format(url, __version__),
    keywords=[
        'pytorch',
        'scatter',
        'embedding',
        'hashing'
    ],
    install_requires=install_requires,
    tests_require=tests_require,
    packages=find_packages(),
)
