import nbformat
import glob


def rename_in_notebook(filepath):
    print(f"Processing {filepath}...")
    with open(filepath, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    changed = False
    for cell in nb.cells:
        if cell.cell_type == "code":
            if "Ngauss" in cell.source:
                cell.source = cell.source.replace("Ngauss", "ngauss")
                changed = True
            if "Nvars" in cell.source:
                cell.source = cell.source.replace("Nvars", "nvars")
                changed = True
        elif cell.cell_type == "markdown":
            if "Ngauss" in cell.source:
                cell.source = cell.source.replace("Ngauss", "ngauss")
                changed = True
            if "Nvars" in cell.source:
                cell.source = cell.source.replace("Nvars", "nvars")
                changed = True

    if changed:
        with open(filepath, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        print(f"Updated {filepath}")
    else:
        print(f"No changes in {filepath}")


notebooks = glob.glob("examples/*.ipynb") + glob.glob("docs/examples/*.ipynb")
for nb in notebooks:
    rename_in_notebook(nb)
