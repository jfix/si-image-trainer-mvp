#!/usr/bin/env bash
set -euo pipefail

siit prepare-references --config configs/base.yaml
siit prepare-queries --config configs/base.yaml
siit split-dataset --config configs/base.yaml
siit build-index --config configs/base.yaml
