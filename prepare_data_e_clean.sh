#!/usr/bin/env bash

set -u
set -o pipefail

SRC_ROOT="/home/teaching/dl_mamba/data_e_raw"
DST_ROOT="/home/teaching/dl_mamba/data_e_clean"
MIN_ID=248
MAX_ID=542
EXPECTED_IMAGES=20

warn() {
  printf 'WARNING: %s\n' "$*" >&2
}

info() {
  printf '%s\n' "$*"
}

if [[ ! -d "$SRC_ROOT" ]]; then
  printf 'ERROR: Source directory not found: %s\n' "$SRC_ROOT" >&2
  exit 1
fi

mkdir -p -- "$DST_ROOT"

declare -A subject_src=()

# Discover all numeric folder names recursively and keep first match per ID.
while IFS= read -r -d '' dir_path; do
  base_name="$(basename "$dir_path")"

  [[ "$base_name" =~ ^[0-9]+$ ]] || continue

  subject_id=$((10#$base_name))
  (( subject_id >= MIN_ID && subject_id <= MAX_ID )) || continue

  if [[ -v "subject_src[$subject_id]" ]]; then
    warn "Duplicate subject ID ${subject_id}: skipping '$dir_path' (already using '${subject_src[$subject_id]}')"
    continue
  fi

  subject_src["$subject_id"]="$dir_path"
done < <(find "$SRC_ROOT" -type d -print0 | sort -z)

if (( ${#subject_src[@]} == 0 )); then
  warn "No valid subject folders found in range ${MIN_ID}-${MAX_ID}."
  exit 0
fi

mapfile -t sorted_ids < <(printf '%s\n' "${!subject_src[@]}" | sort -n)

processed=0
renamed_ok=0
warn_count=0

for id in "${sorted_ids[@]}"; do
  src_dir="${subject_src[$id]}"
  dst_dir="$DST_ROOT/$id"

  if [[ -e "$dst_dir" ]]; then
    warn "Destination already exists for subject ${id}: '$dst_dir' (skipping to avoid overwrite)"
    ((warn_count++))
    continue
  fi

  mkdir -p -- "$dst_dir"
  if ! cp -a -- "$src_dir"/. "$dst_dir"/; then
    warn "Copy failed for subject ${id} from '$src_dir'"
    ((warn_count++))
    rm -rf -- "$dst_dir"
    continue
  fi

  # Some source folders are read-only; ensure we can rename within destination.
  if ! chmod -R u+rwX -- "$dst_dir"; then
    warn "Could not set writable permissions for subject ${id}: '$dst_dir'"
    ((warn_count++))
    continue
  fi

  ((processed++))

  mapfile -d '' images < <(
    find "$dst_dir" -maxdepth 1 -type f \
      \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.tif' -o -iname '*.tiff' \) \
      -print0 | sort -zV
  )

  image_count=${#images[@]}

  if (( image_count != EXPECTED_IMAGES )); then
    warn "Subject ${id} has ${image_count} image(s), expected ${EXPECTED_IMAGES}: '$dst_dir'"
    ((warn_count++))
    continue
  fi

  tmp_files=()
  rename_failed=0

  # Two-phase rename prevents collisions with existing names like 1.jpg.
  for i in "${!images[@]}"; do
    idx=$((i + 1))
    tmp_file="$dst_dir/.rename_tmp_${idx}_$$"

    if ! mv -- "${images[$i]}" "$tmp_file"; then
      warn "Temporary rename failed for subject ${id}: '${images[$i]}'"
      rename_failed=1
      break
    fi

    tmp_files+=("$tmp_file")
  done

  if (( rename_failed == 1 )); then
    ((warn_count++))
    continue
  fi

  for i in "${!tmp_files[@]}"; do
    idx=$((i + 1))
    final_file="$dst_dir/$idx.jpg"

    if [[ -e "$final_file" ]]; then
      warn "Final target already exists for subject ${id}: '$final_file'"
      rename_failed=1
      break
    fi

    if ! mv -- "${tmp_files[$i]}" "$final_file"; then
      warn "Final rename failed for subject ${id}: '${tmp_files[$i]}' -> '$final_file'"
      rename_failed=1
      break
    fi
  done

  if (( rename_failed == 1 )); then
    ((warn_count++))
    continue
  fi

  ((renamed_ok++))
done

info "Done."
info "Subjects copied: ${processed}"
info "Subjects renamed to 1.jpg..20.jpg: ${renamed_ok}"
info "Warnings: ${warn_count}"
