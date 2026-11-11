#!/usr/bin/env bash
# HOST side: copies the desktop's X cookie into /tmp/.docker.xauth (the container's /root/.Xauthority).
# Run by build-images.sh; re-run by hand after a host reboot/re-login. Best-effort: never fails.
F=/tmp/.docker.xauth
touch "$F" 2>/dev/null || exit 0
d="${DISPLAY:-}"; n="${d##*:}"; n="${n%%.*}"
if [ -z "$n" ] || [ ! -S "/tmp/.X11-unix/X$n" ]; then  # VS Code's initializeCommand may lack DISPLAY
  for s in /tmp/.X11-unix/X[0-9]; do [ -S "$s" ] && { d=":${s##*/X}"; break; }; done
fi
if [ -z "${XAUTHORITY:-}" ]; then  # GNOME (Xwayland) keeps its cookie under /run/user
  for a in /run/user/"$(id -u)"/.mutter-Xwaylandauth.* "$HOME/.Xauthority"; do
    [ -s "$a" ] && { export XAUTHORITY="$a"; break; }
  done
fi
cookie="$(xauth nlist "$d" 2>/dev/null | sed -e 's/^..../ffff/')"
if [ -z "$cookie" ]; then
  echo "[xauth] no X cookie for display '${d:-none}': GUI apps in the container get no window"; exit 0
fi
# Write in place: xauth renames its output, and a running container's bind mount would keep the old file.
tmp="$(mktemp)" && printf '%s\n' "$cookie" | xauth -f "$tmp" nmerge - 2>/dev/null && cat "$tmp" >"$F"
rm -f "$tmp"
echo "[xauth] X cookie for display $d -> $F"
