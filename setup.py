# Apache 2.0 Jackson Loper 2021
# Modified from https://github.com/tensorflow/custom-op
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import Extension
from setuptools import find_packages
from setuptools import setup
from setuptools.dist import Distribution


__version__ = '0.0.2'
REQUIRED_PACKAGES = [
    'tensorflow >= 2.4.0',
]
project_name = 'tensorflow-felzenszwalb-edt'


from setuptools.command.install import install
class InstallPlatlib(install):
    def finalize_options(self):
        install.finalize_options(self)
        self.install_lib = self.install_platlib


class BinaryDistribution(Distribution):
  """This class is needed in order to create OS specific wheels."""

  def has_ext_modules(self):
    return True

  def is_pure(self):
    return False

setup(
    name=project_name,
    version=__version__,
    description=('tensorflow-felzenszwalb-edt is an examples for custom ops for TensorFlow'),
    author='Google Inc.',
    author_email='opensource@google.com',
    # Contained modules and scripts.
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    # Add in any packaged data.
    include_package_data=True,
    zip_safe=False,
    distclass=BinaryDistribution,
    cmdclass={'install': InstallPlatlib},
    # PyPI package information.
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
    ],
    license='Apache 2.0',
    keywords='tensorflow custom op machine learning',
)
