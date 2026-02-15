#! /usr/bin/env python3
import os
import glob
import shutil
import json


def fix_stream_outputs(nb):
    """Ensure every stream output has 'name' (stdout/stderr) for nbsphinx validation."""
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream" and "name" not in out:
                out["name"] = "stdout"


def fix_display_metadata(nb):
    """Ensure display_data and execute_result outputs have 'metadata' for nbsphinx validation."""
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            if out.get("output_type") in ("display_data", "execute_result") and "metadata" not in out:
                out["metadata"] = {}


def fix_execute_result(nb):
    """Ensure execute_result outputs have 'execution_count' for nbsphinx validation."""
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            if out.get("output_type") == "execute_result" and "execution_count" not in out:
                out["execution_count"] = cell.get("execution_count", None)


def process_notebook(src, dest):
    with open(src, "r") as f:
        nb = json.load(f)

    # Fix outputs for nbsphinx validation
    fix_stream_outputs(nb)
    fix_display_metadata(nb)
    fix_execute_result(nb)

    # Filter out footer cells
    new_cells = []
    for cell in nb.get("cells", []):
        is_footer = False
        if cell.get("cell_type") == "markdown":
            source = "".join(cell.get("source", []))
            if (
                "MultiMin - Multivariate Gaussian fitting" in source
                and "Jorge I. Zuluaga" in source
            ):
                is_footer = True

        if not is_footer:
            # Inject Plotly renderer fix for documentation build
            if cell.get("cell_type") == "code":
                new_source = []
                for line in cell.get("source", []):
                    if "fig.show()" in line:
                        # Replace fig.show() with HTML embedding
                        # We use include_plotlyjs='cdn' to load lib from CDN (lightweight)
                        # This bypasses nbsphinx MIME rendering issues and iframe path issues
                        new_line = "from IPython.display import HTML; display(HTML(fig.to_html(include_plotlyjs='cdn')))\n"
                        new_source.append(new_line)
                    else:
                        new_source.append(line)
                cell["source"] = new_source

            new_cells.append(cell)

    nb["cells"] = new_cells

    with open(dest, "w") as f:
        json.dump(nb, f, indent=1)


def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    examples_dir = os.path.join(root_dir, "examples")
    docs_examples_dir = os.path.join(root_dir, "docs", "examples")
    rst_file = os.path.join(root_dir, "docs", "examples.rst")
    excluded = {
        "paper-codes.ipynb",
    }

    # Ensure docs/examples exists
    if os.path.exists(docs_examples_dir):
        shutil.rmtree(docs_examples_dir)
    os.makedirs(docs_examples_dir)

    # Read index.rst to get priority order
    index_file = os.path.join(examples_dir, "index.rst")
    priority_order = []
    if os.path.exists(index_file):
        with open(index_file, "r") as f:
            priority_order = [line.strip() for line in f if line.strip()]

    # Collect all notebooks (excluding internal/CI notebooks)
    all_notebooks = [
        f
        for f in glob.glob(os.path.join(examples_dir, "*.ipynb"))
        if os.path.basename(f) not in excluded
    ]
    notebook_map = {os.path.basename(f): f for f in all_notebooks}

    # Determine final order
    ordered_files = []

    # 1. Add files from index.rst if they exist
    for name in priority_order:
        if name in notebook_map:
            ordered_files.append(notebook_map[name])
            del notebook_map[name]
        else:
            print(f"Warning: {name} listed in index.rst but not found in examples/")

    # 2. Add remaining files alphabetically
    for name in sorted(notebook_map.keys()):
        ordered_files.append(notebook_map[name])

    # Copy files and build list for RST
    notebooks = []
    for f in ordered_files:
        basename = os.path.basename(f)
        dest = os.path.join(docs_examples_dir, basename)
        # shutil.copy2(f, dest)
        process_notebook(f, dest)
        notebooks.append(basename)
        print(f"Copied and processed {basename} to docs/examples/")

    # Generate examples.rst
    print(f"Updating {rst_file}...")
    with open(rst_file, "w") as rst:
        rst.write("\n")
        rst.write("Tutorials and Examples\n")
        rst.write("======================\n\n")
        rst.write(".. toctree::\n")
        rst.write("   :maxdepth: 2\n")
        rst.write("   :caption: Examples\n\n")

        for nb in notebooks:
            name = os.path.splitext(nb)[0]
            rst.write(f"   examples/{name}\n")

    print("Done.")


if __name__ == "__main__":
    main()
