##################################################################
#                                                                #
# MultiMin: Multivariate Gaussian fitting                               #
#                                                                #
##################################################################
# License: GNU Affero General Public License v3 (AGPL-3.0)        #
##################################################################

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    # ######################################################################
    # BASIC DESCRIPTION
    # ######################################################################
    name="multimin",
    author="Jorge I. Zuluaga",
    author_email="jorge.zuluaga@udea.edu.co",
    description="MultiMin: Multivariate Gaussian fitting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seap-udea/multimin",
    keywords="fitting multivariate-normal statistics optimization",
    license="AGPL-3.0-only",
    # ######################################################################
    # CLASSIFIER
    # ######################################################################
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        # "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
    ],
    version='0.5.9',
    # ######################################################################
    # FILES
    # ######################################################################
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    # ######################################################################
    # TESTS
    # ######################################################################
    test_suite="pytest",
    tests_require=["pytest"],
    # ######################################################################
    # DEPENDENCIES
    # ######################################################################
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
        "spiceypy>=5.0.0",
        "pandas>=1.0.0",
        "plotly>=5.0.0",
    ],
    python_requires=">=3.8",
    # ######################################################################
    # OPTIONS
    # ######################################################################
    include_package_data=True,
    package_data={"": ["data/*.*", "tests/*.*"]},
)
