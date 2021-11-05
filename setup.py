"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
# https://setuptools.pypa.io/en/latest/setuptools.html
from setuptools import setup, find_packages
import pathlib
import os

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

VERSION_TEMPLATE = """
# Note that we need to fall back to the hard-coded version if either
# setuptools_scm can't be imported or setuptools_scm can't determine the
# version, so we catch the generic 'Exception'.
try:
    from setuptools_scm import get_version
    version = get_version(root='..', relative_to=__file__)
except Exception:
    version = '{version}'
""".lstrip()

setup(name='similarity',
      version='0.1.0',
      #package_dir = {'similarity/',''},
      #use_scm_version={'write_to': os.path.join('similarity', 'version.py'),'write_to_template': VERSION_TEMPLATE},
      include_package_data=True,
      package_data = { "": ["*.txt","data/*.dat"]},
      packages=['similarity'])
#      ext_modules=get_extensions())
