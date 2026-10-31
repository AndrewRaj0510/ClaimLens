#!/usr/bin/env bash
# Package the chart dependencies: the library into every service chart, then
# every service chart into the umbrella chart.
#
# Run it after any change under k8s/charts/. All of them, every time: each
# service chart carries its own copy of claimlens-lib, and Helm's named
# templates are global across a release, so one stale copy can silently
# override the others (it did: a fix in the library had no effect until
# every chart was rebuilt).
set -euo pipefail
cd "$(dirname "$0")/../charts"

SERVICES=(frontend gateway agent guardrails vision-worker eval policy pricing claims-history postgres redis seaweedfs)
for chart in "${SERVICES[@]}"; do
  rm -rf "$chart/charts"
  helm dependency build "$chart" >/dev/null
done
rm -rf claimlens/charts
helm dependency build claimlens >/dev/null
echo "charts built: ${#SERVICES[@]} services + claimlens"
