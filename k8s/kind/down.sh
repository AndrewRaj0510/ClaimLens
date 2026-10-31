#!/usr/bin/env bash
# Delete the kind cluster and everything in it (data included).
set -euo pipefail
kind delete cluster --name claimlens
