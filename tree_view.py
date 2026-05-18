from pathlib import Path

BASE_DIR = Path("/content/data")

def print_tree_summary(path, max_depth=2, current_depth=0, prefix="", max_files_preview=5):
    if current_depth >= max_depth:
        return

    items = sorted(path.iterdir())
    files = [p for p in items if p.is_file()]
    dirs = [p for p in items if p.is_dir()]

    for i, d in enumerate(dirs):
        connector = "└── " if (i == len(dirs) - 1 and not files) else "├── "
        print(prefix + connector + d.name)
        extension = "    " if connector == "└── " else "│   "
        print_tree_summary(
            d,
            max_depth=max_depth,
            current_depth=current_depth + 1,
            prefix=prefix + extension,
            max_files_preview=max_files_preview
        )


    if files:
        print(prefix + f"└── [files: {len(files)}]")
        for f in files[:max_files_preview]:
            print(prefix + f"    ├── {f.name}")
        if len(files) > max_files_preview:
            print(prefix + "    └── ...")

for subfolder in sorted(BASE_DIR.iterdir()):
    if subfolder.is_dir():
        print(f"\n{subfolder.name}")
        print_tree_summary(subfolder, max_depth=3)