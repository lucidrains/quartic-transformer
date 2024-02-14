from setuptools import setup, find_packages

setup(
  name = 'quartic-transformer',
  packages = find_packages(exclude=[]),
  version = '0.0.8',
  license='MIT',
  description = 'Quartic Transformer',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/quartic-transformer',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformer',
    'attention'
  ],
  install_requires=[
    'colt5-attention',
    'einops>=0.7.0',
    'einx[torch]>=0.1.3',
    'taylor-series-linear-attention',
    'torch>=2.0',
    'x-transformers'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
