from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='latent-maximum-model',
    version='0.0.1',
    description='Linear Models for Unobserved Maximum',
    # long_description=long_description,
    # url='https://github.com/madrury/py-glm',
    author='Matthew Drury',
    author_email='matthew.drury.83@gmail.com',
    license='BSD',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    keywords='statistics',
    packages=['latent_maximum_model'],
    install_requires=['numpy', 'pandas'],
)
