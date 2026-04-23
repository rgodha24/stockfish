#!/bin/sh

# download commands with a 5min time-out to ensure things fail if the server stalls
wget_or_curl=$( (command -v wget >/dev/null 2>&1 && echo "wget -qO- --timeout=300 --tries=1") ||
  (command -v curl >/dev/null 2>&1 && echo "curl -skL --max-time 300"))

sha256sum=$( (command -v shasum >/dev/null 2>&1 && echo "shasum -a 256") ||
  (command -v sha256sum >/dev/null 2>&1 && echo "sha256sum"))

if [ -z "$sha256sum" ]; then
  >&2 echo "sha256sum not found, NNUE files will be assumed valid."
fi

get_nnue_filename() {
  sed -n "s/^#define[[:space:]]\+$1[[:space:]]\+\"\([^\"]*\)\"/\1/p" evaluate.h | head -n 1
}

is_hashed_network() {
  case "$1" in
    nn-[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f].nnue)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

validate_network() {
  if ! is_hashed_network "$1"; then
    return 0
  fi

  # If no sha256sum command is available, assume the file is always valid.
  if [ -n "$sha256sum" ] && [ -f "$1" ]; then
    if [ "$1" != "nn-$($sha256sum "$1" | cut -c 1-12).nnue" ]; then
      rm -f "$1"
      return 1
    fi
  fi
}

fetch_network() {
  _filename="$(get_nnue_filename "$1")"

  if [ -z "$_filename" ]; then
    >&2 echo "NNUE file name not found for: $1"
    return 1
  fi

  if ! is_hashed_network "$_filename"; then
    if [ -f "$_filename" ]; then
      echo "Using local custom NNUE $_filename"
      return 0
    fi

    >&2 printf "%s\n" \
      "Missing required custom NNUE: $_filename" \
      "Place it in $(pwd) before building with STACKS=${STACKS:-layer}."
    return 1
  fi

  if [ -f "$_filename" ]; then
    if validate_network "$_filename"; then
      echo "Existing $_filename validated, skipping download"
      return
    else
      echo "Removing invalid NNUE file: $_filename"
    fi
  fi

  if [ -z "$wget_or_curl" ]; then
    >&2 printf "%s\n" "Neither wget or curl is installed." \
      "Install one of these tools to download NNUE files automatically."
    exit 1
  fi

  for url in \
    "https://tests.stockfishchess.org/api/nn/$_filename" \
    "https://github.com/official-stockfish/networks/raw/master/$_filename"; do
    echo "Downloading from $url ..."
    if $wget_or_curl "$url" >"$_filename"; then
      if validate_network "$_filename"; then
        echo "Successfully validated $_filename"
      else
        rm -f $_filename
        echo "Downloaded $_filename is invalid, and has been removed."
        continue
      fi
    else
      rm -f $_filename
      echo "Failed to download from $url"
    fi
    if [ -f "$_filename" ]; then
      return
    fi
  done

  # Download was not successful in the loop, return false.
  >&2 echo "Failed to download $_filename"
  return 1
}

case "${STACKS:-layer}" in
  moe)
    big_macro=EvalFileDefaultNameBigMoe
    ;;
  none)
    big_macro=EvalFileDefaultNameBigNone
    ;;
  *)
    big_macro=EvalFileDefaultNameBigLayer
    ;;
esac

fetch_network "$big_macro" &&
  fetch_network EvalFileDefaultNameSmall
