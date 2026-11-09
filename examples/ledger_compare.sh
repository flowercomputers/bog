#!/bin/sh
# Run the ledger benchmark on fold2 and fold3, each config in a fresh
# process, and print one CSV. Expects fold2 checked out beside fold3.
set -e
here=$(cd "$(dirname "$0")/.." && pwd)
fold2=${FOLD2:-$here/../fold2}
(cd "$fold2" && cargo build -q --release --example ledger_bench)
(cd "$here" && cargo build -q --release --example ledger_bench)
"$here/target/release/examples/ledger_bench" header
for mode in memory deferred strict; do
  for cfg in "1 20000" "100 3000" "1000 500"; do
    set -- $cfg
    txs=$2
    if [ "$mode" = strict ] && [ "$1" = 1 ]; then txs=5000; fi
    for impl in "$fold2" "$here"; do
      "$impl/target/release/examples/ledger_bench" "$mode" "$1" "$txs"
    done
  done
done
