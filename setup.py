import os
import setuptools

src_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(src_dir, 'README.rst')) as f:
    long_description = f.read()

setuptools.setup(
    name='sitq',
    version='0.1.1',
    author='Kazuaki Tanida',
    description='Learning to Hash for Maximum Inner Product Search',
    long_description=long_description,
    url='https://github.com/shiroyagicorp/sitq',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    license='MIT',
    keywords='MIPS, LSH, ITQ, Maximum Inner Product Search',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'pqkmeans',
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
)
