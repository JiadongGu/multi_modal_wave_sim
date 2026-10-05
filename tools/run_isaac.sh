#!/usr/bin/env bash
# Launch Isaac Sim with this extension, and only this extension, enabled.
#
# Nothing is registered globally: no symlink into extsUser, no entry in the
# Extension Search Paths list, no autoload. Every other Isaac Sim session on
# this machine - ASTRA, astra_rl, someone else's work - is untouched.
#
# That matters more as this grows. An extension that subscribes to physics
# steps and applies forces to rigid bodies is not something you want silently
# active while debugging a different simulator.
#
#   ./tools/run_isaac.sh                      # default install
#   ISAAC=/path/to/isaacsim ./tools/run_isaac.sh
#   ./tools/run_isaac.sh /path/to/stage.usd   # extra args pass through

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAAC="${ISAAC:-$HOME/isaacsim-6.0.1}"

if [[ ! -x "$ISAAC/isaac-sim.sh" ]]; then
    echo "Isaac Sim not found at: $ISAAC" >&2
    echo "Set it explicitly:  ISAAC=/path/to/isaacsim $0" >&2
    echo >&2
    echo "Candidates on this machine:" >&2
    ls -d "$HOME"/isaacsim* /opt/isaacsim* 2>/dev/null >&2 || echo "  (none found)" >&2
    exit 1
fi

echo "Isaac Sim : $ISAAC"
echo "extension : $REPO/exts/mmws.wave_sim"

exec "$ISAAC/isaac-sim.sh" \
    --ext-folder "$REPO/exts" \
    --enable mmws.wave_sim \
    "$@"
