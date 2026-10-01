# Replacing the default branch on GitHub with v2

I (the assistant) have no access to `github.com/ikramkhaecu/MADRL_BlackStart`; run this yourself.
Nothing is lost: the old `main` (prototype with the hard-coded multipliers) and `master` (v1, the
code behind the submitted manuscript) stay reachable through tags, which is what you want to be
able to show a reviewer.

```bash
git clone https://github.com/ikramkhaecu/MADRL_BlackStart.git
cd MADRL_BlackStart

# 1. permanent pointers to what existed before
git tag archive/main-prototype origin/main
git tag v1-submitted          origin/master
git push origin archive/main-prototype v1-submitted

# 2. v2 as one commit on top of v1 (so v1 -> v2 is a reviewable diff)
git checkout -b v2 origin/master
git rm -r -q .
unzip -q /path/to/MADRL_BlackStart_v2.zip -d /tmp/v2
cp -r /tmp/v2/MADRL_BlackStart_v2/. .
git add -A
git commit -m "v2.1: physics-consistent environment, 7 algorithms, IEEE 14/39/118 (see CHANGELOG.md)"
git tag v2.1.0

# 3. publish and overwrite main
git push origin v2 v2.1.0
git push --force origin v2:main
```

Then on GitHub: *Settings → Branches* – default branch `main`; remove branch protection on `main`
beforehand if the force-push is rejected; afterwards delete the `v2` branch (identical to `main`) and,
if you like, `master` (kept by the tag `v1-submitted`).

After your full experiment run, commit `sim/results/*.json`, `table_*.tex`, `claims.md` and the
figures, tag that commit (e.g. `v2.1.0-results`) and cite **that tag** in the Data-availability
statement.
