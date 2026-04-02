#!/usr/bin/env bash
set -euo pipefail

siit evaluate --config configs/base.yaml
siit evaluate-calibration --config configs/base.yaml
siit error-analysis --config configs/base.yaml
