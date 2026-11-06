#!/bin/bash
# Install the wheelhouse W into /tmp/env_main and /tmp/env_qtea (no network, no dependency
# resolution) after checking every wheel against W/SHA256SUMS; then check what each env imports.
# A second wheelhouse W2 (suite 2: block2, psutil), checked the same way, goes into /tmp/env_main;
# its MKL wheel's shared libraries (block2 links them) are unpacked to /tmp/env_mkl/lib.
# usage: install_env.sh W [W2]
set -euo pipefail
W=$1; W2=${2:-}
(cd "$W" && sha256sum -c --quiet SHA256SUMS)
for e in main qtea; do
  python3 -m pip install --quiet --no-index --no-deps --target "/tmp/env_$e" "$W"/$e/*.whl
done
if [ -n "$W2" ]; then
  (cd "$W2" && sha256sum -c --quiet SHA256SUMS)
  python3 -m pip install --quiet --no-index --no-deps --target /tmp/env_main "$W2"/main/*.whl
  # not pip: --target drops a wheel's data files, which is where MKL's libraries are
  rm -rf /tmp/env_mkl
  python3 -I -m zipfile -e "$W2"/mkl/mkl-*.whl /tmp/env_mkl
  ln -s /tmp/env_mkl/mkl-*.data/data/lib /tmp/env_mkl/lib
  ls /tmp/env_mkl/lib/libmkl_core.so.2
fi
export PYTHONNOUSERSITE=1
PYTHONPATH=/tmp/env_main python3 -c "
import numpy, scipy, torch, yastn, tenpy
from importlib import metadata as m
assert numpy.__version__ == '2.4.6', numpy.__version__
print('main', numpy.__version__, scipy.__version__, torch.__version__, m.version('yastn'), tenpy.__version__)"
[ -z "$W2" ] || PYTHONPATH=/tmp/env_main python3 -c "
import numpy, pyblock2.driver.core, psutil
from importlib import metadata as m
assert numpy.__version__ == '2.4.6', numpy.__version__
print('main+s2', 'block2', m.version('block2'), 'psutil', psutil.__version__)"
PYTHONPATH=/tmp/env_qtea python3 -c "
import numpy, scipy, torch, qtealeaves, qredtea
from importlib import metadata as m
assert numpy.__version__ == '2.1.3', numpy.__version__
print('qtea', numpy.__version__, scipy.__version__, torch.__version__, m.version('qtealeaves'), m.version('qredtea'))"
