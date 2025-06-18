from distutils.core import setup
from setuptools import find_packages
import os

print(os.getcwd())

with open(
    'README.md', 
    encoding='utf-8'
) as file:
    description = file.read()

setup(
    name='ocr',
    version='0.0.1',
    packages=find_packages(),
    license='Copyright (c) 2023 Hieu Pham',
    zip_safe=True,
    description='Implementation of CRAFT: Character Region Awareness For Text detection.',
    long_description=description,
    long_description_content_type='text/markdown',
    author='Hieu Pham',
    author_email='64821726+hieupth@users.noreply.github.com',
    url='https://gitlab.com/hieupth/pycraft',
    keywords=[],
    install_requires=[],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'Programming Language :: Python :: 3'
    ],
)