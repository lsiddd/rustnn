#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
baseline="$repo_dir/baseline.txt"
binary="$repo_dir/target/release/rustnn"
fast=0

if [[ ${1:-} == "--fast" ]]; then
    fast=1
elif [[ $# -gt 0 ]]; then
    echo "uso: $0 [--fast]" >&2
    exit 2
fi

value() {
    local key=$1
    awk -v key="$key" '$1 == key { print $2; found = 1 } END { if (!found) exit 1 }' "$baseline"
}

check_floor() {
    local key=$1
    local actual=$2
    local expected
    expected=$(value "$key")
    awk -v key="$key" -v actual="$actual" -v expected="$expected" 'BEGIN {
        floor = expected * 0.97
        if (actual + 0 < floor) {
            printf "REGRESSAO %s: %.3f < %.3f (baseline %.3f, tolerancia -3%%)\n", key, actual, floor, expected > "/dev/stderr"
            exit 1
        }
        printf "OK %-20s %.3f (baseline %.3f)\n", key, actual, expected
    }'
}

cd "$repo_dir"
cargo build --release

echo "== gradcheck =="
"$binary" --gradcheck

echo "== fingerprint =="
# 20 passos com 64 imagens e batch 8 sao 2,5 epocas: cruzam a fronteira de
# epoca, exercitam o reembaralhamento e saem do passo 0 do warmup. Custa 12 s.
fingerprint_output=$("$binary" --fingerprint 20 --threads 1 --batch 8 --subset 64)
echo "$fingerprint_output"
actual_fingerprint=$(awk '/^fingerprint:/ { print $2 }' <<<"$fingerprint_output")
expected_fingerprint=$(value fingerprint)
if [[ $actual_fingerprint != "$expected_fingerprint" ]]; then
    echo "REGRESSAO fingerprint: $actual_fingerprint != $expected_fingerprint" >&2
    exit 1
fi

echo "== gemmbench =="
gemm_output=""
for _ in 1 2 3; do
    sample=$("$binary" --gemmbench)
    gemm_output+="${sample}"$'\n'
done
echo "$gemm_output"
best() {
    local pattern=$1
    awk -v pattern="$pattern" 'index($0, pattern) { v = $(NF - 1); if (v > max) max = v } END { print max + 0 }' <<<"$gemm_output"
}
check_floor gemm_s3_qkv_fwd "$(best "S3 qkv fwd")"
check_floor gemm_s3_swiglu_wg "$(best "S3 swiglu wg")"
check_floor gemm_s3_qkv_dx "$(best "S3 qkv dX")"
check_floor gemm_s3_qkv_dw "$(best "S3 qkv dW")"
check_floor gemm_s3_swiglu_wd "$(best "S3 swiglu wd fwd")"
check_floor gemm_merge_fwd "$(best "merge fwd")"

if (( fast == 0 )); then
    echo "== bench 5 =="
    bench_output=$("$binary" --bench 5)
    echo "$bench_output"
    check_floor bench_img_s "$(awk '/^bench:/ { print $2 }' <<<"$bench_output")"
fi

echo "regressao: OK"
