#!/bin/bash
# Download the pinned wheels (hash-checked) and build YASTN's wheel from its tag commit, into W.
# Run once, on a pod of the benchmark image (or any x86-64 Linux with Python 3.11) with network.
# Without network on the pod: run it on a machine with network and add LINUX_WHEELS=1, which asks
# pip for manylinux x86-64 CPython 3.11 wheels whatever the host is.
# usage: build_wheelhouse.sh W
set -euo pipefail
W=$1; D=$(cd "$(dirname "$0")" && pwd)
YASTN_SHA=4670be709907863042aa968b2e5014529a1ca524      # tag 1.6.1
[ -e "$W" ] && { echo "$W exists: refusing to overwrite"; exit 3; }
mkdir -p "$W/main" "$W/qtea"
PLAT=()
[ "${LINUX_WHEELS:-0}" = 1 ] && PLAT=(--platform manylinux_2_28_x86_64 --platform manylinux2014_x86_64
  --python-version 3.11 --implementation cp --abi cp311)
for e in main qtea; do
  python3 -m pip download --no-deps --only-binary=:all: --require-hashes "${PLAT[@]}" \
    -r "$D/requirements-$e.txt" -d "$W/$e"
done
SETUPTOOLS_SCM_PRETEND_VERSION=1.6.1 python3 -m pip wheel --no-deps \
  "git+https://github.com/yastn/yastn@$YASTN_SHA" -w "$W/main"
(cd "$W" && sha256sum main/*.whl qtea/*.whl > SHA256SUMS)
python3 -m pip --version > "$W/pip_version.txt"
cat "$W/SHA256SUMS"
