# Proposed HydroRL deployment-source correction

`hydrorl-proposer-input.patch` targets the separate deployed HydroRL source tree
inspected during the audit. It preserves color for the GPU path's proposer and
allows the explicitly requested `refine_iters=0` diagnostic case.

The patch has not been applied to the server. The source tree there has no Git
metadata. Apply and review it in the actual HydroRL source repository, verify it
against the current version, and evaluate both proposal recall and decoded IDs
on identical native-resolution annotated frames before deployment.

From that repository, check applicability with:

```sh
git apply --check /path/to/hydrorl-proposer-input.patch
```

The local audit checked applicability against its downloaded source snapshot.
The patch does not change the active detector selection or camera exposure.
