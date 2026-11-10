# Releasing

Releases are published by hand-run GitHub Actions workflows with trusted publishing (no stored
secrets). `release-build.yml` builds the sdist and wheel of a tagged version and checks them:
the tag, `pyproject.toml`, `CITATION.cff` and the CHANGELOG heading must agree; `twine check`;
the archive contents; and a clean-venv install of the wheel that imports tnkit and runs a small
DMRG. `release-testpypi.yml` uploads the result to TestPyPI and `release-pypi.yml` to PyPI. Both
run only from a tag `v<version>`.

A version number can be uploaded once only: an index refuses a second upload of the same file
name, even after a deletion, so a fix needs the next version.

## One-time setup

1. Environments `testpypi` and `pypi`, each with the maintainer as required reviewer and
   deployments allowed only from `v*` tags (required reviewers need a public repository):

   ```sh
   for env in testpypi pypi; do
     gh api -X PUT repos/Aditya-Tandon/tnkit/environments/$env --input - <<'JSON'
   {"reviewers": [{"type": "User", "id": 76618291}],
    "prevent_self_review": false,
    "deployment_branch_policy": {"protected_branches": false, "custom_branch_policies": true}}
   JSON
     gh api -X POST repos/Aditya-Tandon/tnkit/environments/$env/deployment-branch-policies \
       -f name='v*' -f type=tag
   done
   ```

2. A trusted publisher on each index, under the account's publishing settings
   (<https://test.pypi.org/manage/account/publishing/> and
   <https://pypi.org/manage/account/publishing/>): project `tnkit`, owner `Aditya-Tandon`,
   repository `tnkit`, workflow `release-testpypi.yml` or `release-pypi.yml`, environment
   `testpypi` or `pypi`.

## Each release

1. Merge the release pull request (version in `pyproject.toml` and `CITATION.cff`, dated
   CHANGELOG heading).
2. Tag the merge commit and push the tag:

   ```sh
   git fetch origin && git switch main && git pull --ff-only
   git tag -a v0.2.0 -m "tnkit 0.2.0" origin/main
   git push origin v0.2.0
   ```

3. Run the workflow from the tag, then approve the deployment when the publish job waits
   ("Review deployments" on the run's page):

   ```sh
   gh workflow run release-pypi.yml -R Aditya-Tandon/tnkit --ref v0.2.0 -f version=0.2.0
   ```

   Use `release-testpypi.yml` for a trial on TestPyPI.

4. Check the upload in a clean environment, outside the checkout:

   ```sh
   cd "$(mktemp -d)" && python3.11 -m venv v && . v/bin/activate
   pip install tnkit==0.2.0      # TestPyPI: add -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple
   python -c "import tnkit; print(tnkit.__version__, tnkit.__file__)"
   ```
