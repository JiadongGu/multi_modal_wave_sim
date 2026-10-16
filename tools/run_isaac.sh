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
#   MMWS_KEEP_ENV=1 ./tools/run_isaac.sh      # do not scrub the environment
#
# The search is deliberately version-agnostic. Pinning a version in a launcher
# script means the script breaks on upgrade, which is exactly what happened
# going from 6.0.1 to 6.2.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- scrub the environment Kit inherits -------------------------------------
# Kit bundles its own CUDA and its own Python. A conda environment on
# LD_LIBRARY_PATH puts a second, older set of those libraries ahead of them,
# and the two do not mix: the bundled libcusparse.so.12 needs the symbol
# __nvJitLinkCreate_12_8 from CUDA 12.8's libnvJitLink.so.12, resolves against
# conda's older copy instead, and `import torch` dies. Every Isaac extension
# that touches torch then fails in turn, and the process is finally killed by
#
#     Inconsistency detected by ld.so: dl-tls.c: _dl_allocate_tls_init:
#     Assertion `listp != NULL' failed!
#
# which is what mixing two C++/CUDA runtimes in one process looks like.
#
# This is not hypothetical on this machine: there was a pip-installed Isaac Sim
# under miniconda3/lib/python3.12/site-packages/isaacsim, so an active (base)
# environment shipped a competing Isaac Sim as well as competing CUDA.
#
# Scrubbing here rather than in anyone's shell profile is deliberate - the
# machine is shared, and a launcher that fixes itself beats a dotfile that
# fixes one user and surprises the next.
sanitize_env() {
    if [[ -n "${MMWS_KEEP_ENV:-}" ]]; then
        echo "environment : left as-is (MMWS_KEEP_ENV set)"
        return
    fi
    local dropped=()
    for var in LD_LIBRARY_PATH PYTHONPATH PYTHONHOME LD_PRELOAD; do
        [[ -n "${!var:-}" ]] && dropped+=("$var")
        unset "$var"
    done

    # Remove conda-family entries from PATH, keeping everything else in order.
    local parts part cleaned=() removed=0
    IFS=':' read -ra parts <<< "$PATH"
    for part in "${parts[@]}"; do
        case "$part" in
            */miniconda3/*|*/anaconda3/*|*/miniforge3/*|*/mambaforge/*|*/conda/*)
                removed=$((removed + 1)) ;;
            *) cleaned+=("$part") ;;
        esac
    done
    if (( removed )); then
        PATH="$(IFS=':'; printf '%s' "${cleaned[*]}")"
        export PATH
        dropped+=("PATH(-${removed} conda)")
    fi

    if (( ${#dropped[@]} )); then
        echo "environment : scrubbed ${dropped[*]}"
    fi
}

# Put Kit's OWN CUDA libraries ahead of everything else.
#
# Scrubbing the environment is necessary but not sufficient. ldconfig may have
# a system CUDA registered - on the lab machine /usr/local/cuda is 12.1 - and
# the loader finds that before Kit's bundled 12.8. Isaac Sim's torch then dies
# on a symbol the older runtime does not export:
#
#     libc10_cuda.so: undefined symbol: cudaGetDriverEntryPointByVersion
#
# and Kit's deprecation manager treats that as fatal, so the app exits 0 after
# about six seconds with no crash and no useful message. Prepending the
# bundled nvidia/*/lib directories makes Kit's own copies win.
#
# This is why the fix cannot simply be "unset LD_LIBRARY_PATH": the system
# path is not in LD_LIBRARY_PATH at all, it is in the ldconfig cache.
prefer_bundled_cuda() {
    local root="$1" dirs
    dirs="$(find "$root/exts/isaacsim.pip.nv/pip_prebundle/nvidia" \
                 -maxdepth 2 -name lib -type d 2>/dev/null | tr '\n' ':')"
    if [[ -n "$dirs" ]]; then
        export LD_LIBRARY_PATH="${dirs%:}"
        echo "cuda      : using Kit's bundled libraries"
    fi
}

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

sanitize_env

if ISAAC_DIR="$(find_isaac)"; then
    version="$(cat "$ISAAC_DIR/VERSION" 2>/dev/null || echo unknown)"
    echo "Isaac Sim : $ISAAC_DIR  ($version)"
    prefer_bundled_cuda "$ISAAC_DIR"
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
