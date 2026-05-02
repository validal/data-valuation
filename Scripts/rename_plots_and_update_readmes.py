#!/usr/bin/env python3
import os
import re
from pathlib import Path

root = Path("Plots") / "fine_grained"
if not root.exists():
    print(f"Root path not found: {root.resolve()}")
    raise SystemExit(1)

# Rename files starting with Figure_1_
renamed_count = 0
skipped_count = 0
for p in root.rglob("Figure_*"):
    try:
        # remove leading 'Figure_<id>_' where <id> can be numbers or other tokens
        new_name = re.sub(r"^Figure_[^_]+_", "", p.name, count=1)
        if new_name == p.name:
            # nothing to do
            continue
        new_path = p.with_name(new_name)
        if new_path.exists():
            print(f"Skipping rename {p} -> {new_path} (target exists)")
            skipped_count += 1
        else:
            p.rename(new_path)
            print(f"Renamed: {p} -> {new_path}")
            renamed_count += 1
    except Exception as e:
        print(f"Error renaming {p}: {e}")

# Update README in each dataset folder under root
updated_readmes = []
for dataset in sorted(root.iterdir()):
    if dataset.is_dir():
        readme_md = dataset / "README.md"
        if readme_md.exists():
            try:
                text = readme_md.read_text(encoding='utf-8')
            except Exception:
                text = readme_md.read_text(encoding='latin-1')
            marker = "High and low value parameter chosen"
            note = (
                "High and low value parameter chosen: the `tuning/high` and `tuning/low` directories\n"
                "contain plots produced for the high and low parameter settings used in the paper.\n"
            )
            if marker not in text:
                new_text = text.rstrip() + "\n\n" + note
                try:
                    readme_md.write_text(new_text, encoding='utf-8')
                except Exception:
                    readme_md.write_text(new_text, encoding='latin-1')
                updated_readmes.append(str(readme_md))
                print(f"Updated README: {readme_md}")
            else:
                print(f"README already mentions parameters: {readme_md}")
        else:
            print(f"No README.md in {dataset}")

print('\nSummary:')
print(f'Renamed files: {renamed_count}')
print(f'Skipped (target exists): {skipped_count}')
print(f'Readmes updated: {len(updated_readmes)}')
if updated_readmes:
    for r in updated_readmes:
        print(' - ' + r)
