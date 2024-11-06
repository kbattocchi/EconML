# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

import argparse
import re
from collections import defaultdict, namedtuple
from os import listdir, path

import packaging.version
from packaging.version import Version

# We have a list of requirements files, one per python version and OS.
# We want to generate a single requirements file that specifies the requirements
# for each package contained in any of those files, along with the constraints on python version
# and OS that apply to each package.

FileParts = namedtuple('FileParts', ['os', 'py_version'])

# For each version of a package (say numpy==0.24.1), we'll have a set of os/py_version pairs
# where it was installed; the correct constraint will be the union of all these pairs.
# However, for readability we'd like to simplify that when possible to something more readable.
# For example, if numpy==0.24.1 is installed on all versions of python and all OSes, we can just say
# "numpy==0.24.1"; if it's installed on all versions of python on ubuntu, we can say
# "numpy==0.24.1; platform_system=='Linux'".


# We'll precompute a dictionary mapping from certain simple sets of pairs to a string representation of the constraint
# For simplicity, we won't consider all possible constraints, just some easy to generate ones.
# In the most general case we'll OR together constraints grouped by os
def simple_constraint_map(all_combos: frozenset[FileParts]) -> tuple[dict[frozenset[FileParts], str],
                                                                     dict[tuple[str, frozenset[Version]], str]]:
    all_os = frozenset({fp.os for fp in all_combos})
    all_py_versions = frozenset({fp.py_version for fp in all_combos})

    constraint_map = {}
    for os in all_os:
        filtered_combos = frozenset({fp for fp in all_combos if fp.os == os})
        constraint_map[filtered_combos] = f"platform_system=='{os}'"
        constraint_map[all_combos - filtered_combos] = f"platform_system!='{os}'"

    for i, py_version in enumerate(sorted(all_py_versions)):
        filtered_combos = frozenset({fp for fp in all_combos if fp.py_version == py_version})
        constraint_map[filtered_combos] = f"python_version=='{py_version}'"
        if i > 0 and i < len(all_py_versions)-1:
            constraint_map[all_combos - filtered_combos] = f"python_version!='{py_version}'"

        if i > 0:
            less_than = frozenset({fp for fp in all_combos if fp.py_version < py_version})
            constraint_map[less_than] = f"python_version<'{py_version}'"
        if i < len(all_py_versions)-2:
            greater_than = frozenset({fp for fp in all_combos if fp.py_version > py_version})
            constraint_map[greater_than] = f"python_version>'{py_version}'"

    constraint_map[all_combos] = None

    # generate per-os python version constraints
    os_map = {}
    for os in all_os:
        for i, py_version in enumerate(all_py_versions):
            filtered_combos = frozenset({fp for fp in all_combos if fp.os == os and fp.py_version == py_version})
            os_map[(os, frozenset({py_version}))] = f"python_version=='{py_version}'"
            if i > 0 and i < len(all_py_versions)-1:
                os_map[(os, all_py_versions - frozenset({py_version}))] = f"python_version!='{py_version}'"

            if i > 0:
                os_map[(os, frozenset({py for py in all_py_versions
                                       if py < py_version}))] = f"python_version<'{py_version}'"
            if i < len(all_py_versions)-1:
                os_map[(os, frozenset({py for py in all_py_versions
                                       if py > py_version}))] = f"python_version>'{py_version}'"

        os_map[(os, all_py_versions)] = None

    return constraint_map, os_map


# Convert between GitHub Actions' platform names and Python's platform.system() names
platform_map = {'macos': 'Darwin', 'ubuntu': 'Linux', 'windows': 'Windows'}


def make_req_file(requirements_directory, regex):
    req_regex = r'^(.*?)==(.*)$'  # parses requirements from pip freeze results
    files = listdir(requirements_directory)

    all_combos = set()
    req_dict = defaultdict(lambda: defaultdict(set))  # package -> package_version -> set of FileParts

    for file in files:
        match = re.match(regex, file)
        if not match:
            print(f"Skipping {file} because it doesn't match the regex")
            continue
        os = platform_map[match.group('os')]
        py_version = packaging.version.parse(match.group('pyversion'))
        parts = FileParts(os, py_version)
        all_combos.add(parts)

        # read each line of the file
        with open(path.join(requirements_directory, file)) as lines:
            for line in lines:
                # Regex to match requirements file names as stored by ci.yml
                match = re.search(req_regex, line)
                pkg = match.group(1)
                pkg_version = packaging.version.parse(match.group(2))
                req_dict[pkg][pkg_version].add(parts)

    constraint_map, os_map = simple_constraint_map(frozenset(all_combos))
    reqs = []
    print(f"All combos: {all_combos}")
    for pkg, versions in sorted(req_dict.items()):
        for version, parts in sorted(versions.items()):

            if pkg == 'scipy':
                print(f"{version}: {parts}")

            parts = frozenset(parts)
            req = f"{pkg}=={version}"

            if parts in constraint_map:
                constraint = constraint_map[parts]
                if constraint is None:
                    suffix = ''  # don't need to add any constraint
                else:
                    suffix = f"; {constraint}"

            else:
                os_constraints = []
                os_parts = defaultdict(set)
                for fp in parts:
                    os_parts[fp.os].add(fp.py_version)

                for os in sorted(os_parts.keys()):
                    os_key = (os, frozenset(os_parts[os]))
                    if os_key in os_map:
                        constraint = os_map[os_key]
                        if constraint is None:
                            os_constraints.append(f"platform_system=='{os}'")
                        else:
                            os_constraints.append(f"platform_system=='{os}' and {constraint}")
                    else:
                        version_constraint = " or ".join([f"python_version=='{py_version}'"
                                                          for py_version in sorted(os_parts[os])])
                        os_constraints.append(f"platform_system=='{os}' and ({version_constraint})")
                if len(os_constraints) == 1:
                    suffix = f"; {os_constraints[0]}"
                else:
                    suffix = f"; ({') or ('.join(os_constraints)})"

            reqs.append(f"{req}{suffix}")

    return '\n'.join(reqs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate requirements files for CI')
    parser.add_argument('requirements_directory', type=str, help='Directory containing requirements files')
    parser.add_argument('regex', type=str,
                        help='Regex to match requirements file names, must have named groups "os" and "pyversion"')
    parser.add_argument('output_name', type=str, help='File to write requirements to')
    args = parser.parse_args()
    print(f"Regex: {args.regex}")
    reqs = make_req_file(args.requirements_directory, args.regex)
    with open(args.output_name, 'w') as f:
        f.write(reqs)
