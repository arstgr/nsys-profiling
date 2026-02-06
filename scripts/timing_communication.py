# Copyright (c) 2026 Amirreza Rastegari
# Licensed under the MIT License

import sqlite3
from collections import defaultdict

def bucket_for_size(msg_bytes):
    if msg_bytes < 65536:
        return "msg < 64KB"
    elif msg_bytes < 1048576:
        return "64KB <= msg < 1MB"
    elif msg_bytes < 16777216:
        return "1MB <= msg < 16MB"
    else:
        return "msg >= 16MB"

def _extract_int_after_key(name, key):
    """
    this looks for 'key' in string and parses the following integer
    I'm assuming a pattern like '..., data_size 114459648, ...' 
    adjust if it is different
    """
    idx = name.find(key)
    if idx == -1:
        return None
    i = idx + len(key)
    # skip any spaces
    while i < len(name) and name[i].isspace():
        i += 1
    j = i
    while j < len(name) and name[j].isdigit():
        j += 1
    if j == i:
        return None
    return int(name[i:j])

def parse_nccl_nvtx_name(name):
    """
    this essentially parses lines like
      "ncclAllGather(): ... data_size 114459648, type_size 2, ..."
    then returns (op, data_size, type_size) or None if it doesn't match/exist
    """
    if not (name.startswith("ncclAll") or name.startswith("ncclReduce")):
        return None

    try:
        lparen = name.index("(")
    except ValueError:
        return None
    op = name[:lparen]  # "ncclAllGather", "ncclReduceScatter", "ncclAllReduce"

    data_size = _extract_int_after_key(name, "data_size")
    if data_size is None:
        return None

    type_size = _extract_int_after_key(name, "type_size")
    if type_size is None:
        return None

    return op, data_size, type_size

def analyze_nccl_by_msg_size(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    #this requires very detailed NVTX events with data_size + type_size in the text
    cur.execute("""
        SELECT
          e.start,
          e.end,
          COALESCE(s.value, e.text) AS name
        FROM NVTX_EVENTS AS e
        LEFT JOIN StringIds AS s ON e.textId = s.id
        WHERE (
                COALESCE(s.value, e.text) LIKE 'ncclAllGather(): commHash%'
             OR COALESCE(s.value, e.text) LIKE 'ncclReduceScatter(): commHash%'
             OR COALESCE(s.value, e.text) LIKE 'ncclAllReduce(): commHash%'
          )
          AND e.start IS NOT NULL
          AND e.end   IS NOT NULL
    """)
    rows = cur.fetchall()
    conn.close()

    stats = defaultdict(list)  # (op, bucket) -> [durations_ns]

    for start, end, name in rows:
        if end <= start or not name:
            continue

        parsed = parse_nccl_nvtx_name(name)
        if not parsed:
            continue
        op, data_size, type_size = parsed

        msg_bytes = data_size * type_size
        bucket = bucket_for_size(msg_bytes)
        dur = end - start
        stats[(op, bucket)].append(dur)

    to_sec = 1e9
    print(f"{'Op':20} | {'Bucket':20} | {'Inst':>8} | {'Sum (s)':>12} | "
          f"{'Avg (ms)':>10} | {'Min (ms)':>10} | {'Max (ms)':>10}")
    print("-" * 110)
    for (op, bucket), durs in sorted(stats.items()):
        inst = len(durs)
        sum_ns = sum(durs)
        avg_ns = sum_ns / inst if inst else 0
        min_ns = min(durs) if durs else 0
        max_ns = max(durs) if durs else 0
        print(
            f"{op:20} | {bucket:20} | {inst:8d} | "
            f"{sum_ns/to_sec:12.6f} | {avg_ns/1e6:10.3f} | "
            f"{min_ns/1e6:10.3f} | {max_ns/1e6:10.3f}"
        )

if __name__ == "__main__":
    analyze_nccl_by_msg_size("profile_test_db.sqlite")
