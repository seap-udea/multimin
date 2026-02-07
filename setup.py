##################################################################
#                                                                #
# MultiMin: Multivariate Gaussian fitting                        #
#                                                                #
##################################################################
# License: GNU Affero General Public License v3 (AGPL-3.0)       #
##################################################################
from setuptools import setup, find_packages
import os

##################################################################
# Prepare README.md for include Math in LaTeX format for PyPI
##################################################################
def _strip_unsafe_rst_directives(rst_text: str) -> str:
    """Remove .. raw:: (except html) and .. container:: blocks so PyPI's renderer accepts the RST.
    Keep .. raw:: html so images and divs in the README are shown on PyPI."""
    lines = rst_text.splitlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        # Skip .. container:: entirely
        if stripped == ".. container::":
            i += 1
            while i < len(lines) and (lines[i].startswith(" ") or lines[i].strip() == ""):
                i += 1
            continue
        # Skip .. raw:: only when not html (keep .. raw:: html for images)
        if stripped.startswith(".. raw::"):
            if "html" not in stripped.lower():
                i += 1
                while i < len(lines) and (lines[i].startswith(" ") or lines[i].strip() == ""):
                    i += 1
                continue
        out.append(line)
        i += 1
    return "\n".join(out).replace("\n\n\n\n", "\n\n")  # collapse excess blank lines


# Prefer RST long_description for PyPI so LaTeX math (via :math: and .. math::) can render
_readme_path = os.path.join(os.path.dirname(__file__), "README.md")
long_description_content_type = "text/markdown"
try:
    import pypandoc
    # Convert Markdown to RST; tex_math_dollars makes $...$ and $$...$$ become :math: and .. math::
    for _fmt in ("markdown+tex_math_dollars", "markdown"):
        try:
            long_description = pypandoc.convert_file(
                _readme_path,
                "rst",
                format=_fmt,
                extra_args=["--wrap=none"],
            )
            long_description = _strip_unsafe_rst_directives(long_description)
            long_description_content_type = "text/x-rst"
            break
        except RuntimeError:
            continue
    else:
        with open(_readme_path, "r", encoding="utf-8") as _fh:
            long_description = _fh.read()
except (ImportError, OSError):
    with open(_readme_path, "r", encoding="utf-8") as _fh:
        long_description = _fh.read()

##################################################################
# Setup the package
##################################################################
setup(
    name="multimin",
    author="Jorge I. Zuluaga",
    author_email="jorge.zuluaga@udea.edu.co",
    description="MultiMin: Multivariate Gaussian fitting",
    long_description=long_description,
    long_description_content_type=long_description_content_type,
    url="https://github.com/seap-udea/multimin",
    keywords="fitting multivariate-normal statistics optimization",
    license="AGPL-3.0-only",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    version='0.6.1',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    test_suite="pytest",
    tests_require=["pytest"],
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
        "spiceypy>=5.0.0",
        "pandas>=1.0.0",
        "plotly>=5.0.0",
        "pypandoc"
    ],
    python_requires=">=3.8",
    include_package_data=True,
    package_data={"": ["data/*.*", "tests/*.*"]},
)
