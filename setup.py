##################################################################
#                                                                #
# MultiMin: Multivariate Gaussian fitting                        #
#                                                                #
##################################################################
# License: GNU Affero General Public License v3 (AGPL-3.0)       #
##################################################################
from setuptools import setup, find_packages
import os
import re

##################################################################
# Prepare README.md for include Math in LaTeX format for PyPI
##################################################################
# Pattern: <div align="center">\n  <img src="URL" alt="ALT" width="W"/>\n</div>
_DIV_IMG_PAT = re.compile(
    r'<div\s+align="center">\s*<img\s+src="([^"]+)"\s+alt="([^"]*)"\s+width="([^"]+)"\s*/>\s*</div>',
    re.IGNORECASE | re.DOTALL,
)
# Pattern: logo block with link <div align="center">\n  <a href="HREF">\n  <img src="URL" alt="ALT" width="W"/>\n  </a>\n</div>
_DIV_A_IMG_PAT = re.compile(
    r'<div\s+align="center">\s*<a\s+href="([^"]+)">\s*<img\s+src="([^"]+)"\s+alt="([^"]*)"\s+width="([^"]+)"\s*/>\s*</a>\s*</div>',
    re.IGNORECASE | re.DOTALL,
)


def _preprocess_readme_for_rst(md_text: str) -> str:
    """Replace HTML image blocks with Markdown so pypandoc outputs .. figure:: / .. image:: (no raw)."""
    # Logo with link -> [![alt](imgurl)](href)
    def repl_logo(m):
        href, imgurl, alt = m.group(1), m.group(2), m.group(3)
        return "\n\n[![{alt}]({imgurl})]({href})\n\n".format(alt=alt, imgurl=imgurl, href=href)
    md_text = _DIV_A_IMG_PAT.sub(repl_logo, md_text)
    # Plain div+img -> ![alt](url)
    def repl(m):
        url, alt = m.group(1), m.group(2)
        return "\n\n![{alt}]({url})\n\n".format(alt=alt, url=url)
    return _DIV_IMG_PAT.sub(repl, md_text)


def _add_width_to_gallery_images(rst_text: str, width: str = "600") -> str:
    """Add :width: to .. image:: or .. figure:: lines (our gallery images and header logo)."""
    lines = rst_text.splitlines()
    out = []
    for line in lines:
        out.append(line)
        stripped = line.strip()
        if not (stripped.startswith(".. image::") or stripped.startswith(".. figure::")):
            continue
        # Gallery figures (png) or header logo (webp)
        if ("gallery" in line and "png" in line) or ("docs" in line and "webp" in line):
            out.append("   :width: " + width)
    return "\n".join(out)


def _strip_unsafe_rst_directives(rst_text: str) -> str:
    """Remove .. raw:: and .. container:: blocks (PyPI disables raw directive)."""
    lines = rst_text.splitlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped == ".. container::":
            i += 1
            while i < len(lines) and (lines[i].startswith(" ") or lines[i].strip() == ""):
                i += 1
            continue
        if stripped.startswith(".. raw::"):
            i += 1
            while i < len(lines) and (lines[i].startswith(" ") or lines[i].strip() == ""):
                i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out).replace("\n\n\n\n", "\n\n")


# Prefer RST long_description for PyPI so LaTeX math (via :math: and .. math::) can render
_readme_path = os.path.join(os.path.dirname(__file__), "README.md")
long_description_content_type = "text/markdown"
try:
    import pypandoc
    with open(_readme_path, "r", encoding="utf-8") as _fh:
        _md = _fh.read()
    _md = _preprocess_readme_for_rst(_md)  # HTML img -> ![alt](url) so RST has .. image:: (no raw)
    for _fmt in ("markdown+tex_math_dollars", "markdown"):
        try:
            long_description = pypandoc.convert_text(
                _md,
                "rst",
                format=_fmt,
                extra_args=["--wrap=none"],
            )
            long_description = _strip_unsafe_rst_directives(long_description)
            long_description = _add_width_to_gallery_images(long_description)
            long_description_content_type = "text/x-rst"
            break
        except RuntimeError:
            continue
    else:
        long_description = _md
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
    version='0.6.4',
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
