"""Static checks for the documented FADetr release path; no ML imports."""

import ast
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'rotated_rtdetr'
CONFIGS = ROOT / 'configs'
PRIMARY = CONFIGS / 'o2_rtdetr_r34vd_2xb4_72e_dota_daff_faahead.py'


def main():
    errors = []
    trees = {}
    sources = sorted(PACKAGE.glob('*.py')) + sorted(CONFIGS.glob('*.py'))
    sources += sorted((ROOT / 'tools').glob('*.py'))
    for path in sources:
        try:
            trees[path] = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        except (OSError, SyntaxError) as exc:
            errors.append(f'{path.relative_to(ROOT)}: {exc}')

    for path, tree in trees.items():
        if path.parent != PACKAGE:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                target = PACKAGE.joinpath(*node.module.split('.'))
                if not target.with_suffix('.py').is_file() and not (target / '__init__.py').is_file():
                    errors.append(f'{path.name}: missing local module {node.module}')

    init_tree = trees.get(PACKAGE / '__init__.py')
    if init_tree:
        imported = {
            alias.asname or alias.name
            for node in ast.walk(init_tree) if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        for node in init_tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == '__all__'
                for target in node.targets
            ):
                for name in ast.literal_eval(node.value):
                    if name not in imported:
                        errors.append(f'__init__.py: undefined export {name}')

    # Follow only the supported R34 entry's configuration inheritance.
    seen = set()

    def visit_config(path):
        if path in seen:
            return
        seen.add(path)
        tree = trees.get(path)
        if tree is None:
            errors.append(f'missing or invalid config: {path.relative_to(ROOT)}')
            return
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                visit_config(CONFIGS.joinpath(*node.module.split('.')).with_suffix('.py'))

    visit_config(PRIMARY)
    for required in ('README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md', '.gitignore'):
        if not (ROOT / required).is_file():
            errors.append(f'missing release file: {required}')
    if errors:
        for error in errors:
            print('ERROR:', error)
        return 1
    print(f'PASS: parsed {len(trees)} Python files; local imports and exports resolve.')
    print(f'PASS: primary config inheritance resolves across {len(seen)} files.')
    print('Scope: static checks only; no model construction, execution or result verification.')
    print('Legacy/other-dataset config imports are outside the supported runtime path.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
