#!/bin/sh
# Adjusts the app user to match PUID/PGID (defaults: 1000/1000) before dropping
# root, so files this container creates under the mounted media/data/backup
# volumes come out owned by whatever user actually owns those paths on the
# host -- a fixed build-time UID is a common permission mismatch for
# self-hosters whose host user isn't 1000.
set -e

PUID="${PUID:-1000}"
PGID="${PGID:-1000}"

if [ "$(id -u)" = "0" ]; then
    CURRENT_UID="$(id -u spf)"
    CURRENT_GID="$(id -g spf)"

    if [ "$PGID" != "$CURRENT_GID" ]; then
        groupmod -o -g "$PGID" spf
    fi
    if [ "$PUID" != "$CURRENT_UID" ]; then
        usermod -o -u "$PUID" spf
    fi

    mkdir -p /data
    chown -R spf:spf /data

    exec su spf -s /bin/sh -c 'exec "$0" "$@"' -- "$@"
else
    exec "$@"
fi
