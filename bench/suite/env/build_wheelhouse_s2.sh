#!/bin/bash
# Download suite 2's extra wheels (hash-checked) into W/main, block2's MKL runtime into W/mkl, and
# write W/SHA256SUMS. Run on a pod of the benchmark image with network, or anywhere with
# LINUX_WHEELS=1 (manylinux x86-64 CPython 3.11 wheels whatever the host is).
# usage: build_wheelhouse_s2.sh W
set -euo pipefail
W=$1; D=$(cd "$(dirname "$0")" && pwd)
[ -e "$W" ] && { echo "$W exists: refusing to overwrite"; exit 3; }
mkdir -p "$W/main" "$W/mkl"
PLAT=()
[ "${LINUX_WHEELS:-0}" = 1 ] && PLAT=(--platform manylinux_2_28_x86_64 --platform manylinux2014_x86_64
  --python-version 3.11 --implementation cp --abi cp311 --abi abi3)
dl() { python3 -m pip download --no-deps --only-binary=:all: --require-hashes "${PLAT[@]}" -r "$D/$1" -d "$W/$2"; }
dl requirements-s2.txt main
dl requirements-s2-mkl.txt mkl
(cd "$W" && sha256sum main/*.whl mkl/*.whl > SHA256SUMS)
python3 -m pip --version > "$W/pip_version.txt"
cat "$W/SHA256SUMS"
