#!/usr/bin/env bash
# Replace `FROM image:tag` with `FROM image:tag@sha256:...` in every Dockerfile,
# so builds use an immutable base. Run once after bumping a base-image tag,
# then commit the result. Needs docker with buildx and registry access.
set -euo pipefail

cd "$(dirname "$0")/.."

mapfile -t dockerfiles < <(git ls-files '*Dockerfile' 'docker/*.Dockerfile' 2>/dev/null || find . -name '*Dockerfile' -not -path './.venv/*')

for file in "${dockerfiles[@]}"; do
  while read -r image; do
    [[ "${image}" == *@sha256:* ]] && continue
    digest="$(docker buildx imagetools inspect "${image}" --format '{{json .Manifest.Digest}}' | tr -d '"')"
    if [[ ! "${digest}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
      echo "could not resolve ${image}" >&2
      exit 1
    fi
    sed -i "s|^FROM ${image}\$|FROM ${image}@${digest}|" "${file}"
    echo "${file}: ${image} -> ${digest}"
  done < <(awk '/^FROM /{print $2}' "${file}" | sort -u)
done
