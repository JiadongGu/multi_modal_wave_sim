#!/usr/bin/env bash
# Launch Isaac Sim with this extension, and only this extension, enabled.
#
# Nothing is registered globally: no symlink into extsUser, no entry in the
# Extension Search Paths list, no autoload. Every other Isaac Sim session on
# this machine - ASTRA, astra_rl, someone else's work - is untouched. That
# matters more on a shared machine, and it matters a lot for an extension that
# subscribes to physics steps and applies forces to rigid bodies.
#
#   ./tools/run_isaac.sh                      # find Isaac Sim, whatever version
#   ISAAC=/path/to/isaacsim ./tools/run_isaac.sh
#   ./tools/run_isaac.sh /path/to/stage.usd   # extra args pass through
#
# The search is deliberately version-agnostic. Pinning a version in a launcher
# script means the script breaks on upgrade, which is exactly what happened
# going from 6.0.1 to 6.2.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Newest first, so an upgrade is picked up without touching this file.
candidates() {
    shopt -s nullglob
    local dirs=(
        "$HOME"/isaacsim*
        "$HOME"/isaac-sim*
        "$HOME"/.local/share/ov/pkg/isaac*
        /opt/isaacsim* /opt/isaac-sim*
        /isaac-sim
    )
    printf '%s\n' "${dirs[@]}" | sort -rV
}

find_isaac() {
    if [[ -n "${ISAAC:-}" ]]; then
        [[ -x "$ISAAC/isaac-sim.sh" ]] && { echo "$ISAAC"; return 0; }
        echo "ISAAC is set to '$ISAAC' but there is no isaac-sim.sh there" >&2
        return 1
    fi
    local dir
    while read -r dir; do
        [[ -x "$dir/isaac-sim.sh" ]] && { echo "$dir"; return 0; }
    done < <(candidates)
    return 1
}

if ISAAC_DIR="$(find_isaac)"; then
    echo "Isaac Sim : $ISAAC_DIR"
    echo "extension : $REPO/exts/mmws.wave_sim"
    exec "$ISAAC_DIR/isaac-sim.sh" \
        --ext-folder "$REPO/exts" \
        --enable mmws.wave_sim \
        "$@"
fi

# No directory install. Isaac Sim 4.0+ also ships on pip, which puts an
# `isaacsim` launcher on PATH that takes the same Kit arguments.
if command -v isaacsim >/dev/null 2>&1; then
    echo "Isaac Sim : $(command -v isaacsim)  (pip install)"
    echo "extension : $REPO/exts/mmws.wave_sim"
    exec isaacsim --ext-folder "$REPO/exts" --enable mmws.wave_sim "$@"
fi

echo "Isaac Sim not found." >&2
echo >&2
echo "Looked for isaac-sim.sh in:" >&2
candidates | sed 's/^/  /' >&2
echo "  (and for an 'isaacsim' command on PATH)" >&2
echo >&2
echo "Point at it directly:" >&2
echo "  ISAAC=/path/to/isaacsim $0" >&2
echo >&2
echo "To hunt for it:" >&2
echo "  find \$HOME /opt -maxdepth 4 -name isaac-sim.sh 2>/dev/null" >&2
exit 1
