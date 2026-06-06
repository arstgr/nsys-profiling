import sqlite3
import os
import sys


def merge_intervals(intervals):
    """merges overlapping/adjacent intervals and returns sorted list of (start, end)"""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def total_duration(intervals):
    return sum(e - s for s, e in intervals)


def subtract_intervals(a, b):
    """returns (a \\ b) both inputs must already be merged & sorted."""
    result = []
    bi = 0
    for s, e in a:
        cur = s
        while bi < len(b) and b[bi][1] <= cur:
            bi += 1
        j = bi
        while j < len(b) and b[j][0] < e:
            bs, be = b[j]
            if bs > cur:
                result.append((cur, min(bs, e)))
            cur = max(cur, be)
            if cur >= e:
                break
            j += 1
        if cur < e:
            result.append((cur, e))
    return result


def analyze_nvidia_hpl(db_path):
    if not os.path.exists(db_path):
        print(f"Error: {db_path} not found.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Total GPU activity envelope
    try:
        cursor.execute("SELECT MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL")
        row = cursor.fetchone()
        if not row or row[0] is None:
            print("Error: No GPU events found in CUPTI_ACTIVITY_KIND_KERNEL table.")
            conn.close()
            return
        global_start, global_end = row
        total_time_ns = global_end - global_start
    except sqlite3.OperationalError as e:
        print(f"Database error: {e}")
        print(f"Please run 'sqlite3 {db_path} \".tables\"' to verify your database schema.")
        conn.close()
        return

    # 2. Pull every kernel with its demangled + short name so we can classify compute vs NCCL.
    #    Checking both name fields avoids leaking NCCL kernels into the compute bucket.
    query = """
        SELECT k.start, k.end, sd.value, ss.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        LEFT JOIN StringIds sd ON k.demangledName = sd.id
        LEFT JOIN StringIds ss ON k.shortName    = ss.id
        ORDER BY k.start ASC
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("Error: No kernels resolved from database.")
        return

    compute_intervals = []
    nccl_intervals = []
    for start, end, demangled, shortname in rows:
        name = f"{demangled or ''} {shortname or ''}".lower()
        if 'nccl' in name:
            nccl_intervals.append((start, end))
        else:
            compute_intervals.append((start, end))

    # build the three disjoint buckets
    compute_merged = merge_intervals(compute_intervals)
    nccl_merged    = merge_intervals(nccl_intervals)
    all_merged     = merge_intervals(compute_merged + nccl_merged)

    compute_time_ns      = total_duration(compute_merged)
    nccl_exposed_time_ns = total_duration(subtract_intervals(nccl_merged, compute_merged))
    gpu_idle_time_ns     = total_time_ns - total_duration(all_merged)
    comm_total_ns        = nccl_exposed_time_ns + gpu_idle_time_ns

    # verify the three buckets must sum to the envelope
    assert compute_time_ns + nccl_exposed_time_ns + gpu_idle_time_ns == total_time_ns

    # convert to second and percents
    def to_s(ns): return ns / 1e9
    def pct(s):   return (s / to_s(total_time_ns) * 100) if total_time_ns > 0 else 0.0

    total_s = to_s(total_time_ns)
    comp_s  = to_s(compute_time_ns)
    nccl_s  = to_s(nccl_exposed_time_ns)
    idle_s  = to_s(gpu_idle_time_ns)
    comm_s  = to_s(comm_total_ns)

    print("============================================================")
    print(" NVIDIA HPL BINARY PROFILE: RANK 0 (CUPTI SCHEMAS)")
    print("============================================================")
    print(f"Total GPU Wall Time:        {total_s:10.4f} s (100.00%)")
    print(f"  Active Compute Time:      {comp_s:10.4f} s ({pct(comp_s):6.2f}%)")
    print(f"  NCCL Exposed Time:        {nccl_s:10.4f} s ({pct(nccl_s):6.2f}%)")
    print(f"  GPU Idle (verbs/sync):    {idle_s:10.4f} s ({pct(idle_s):6.2f}%)")
    print("------------------------------------------------------------")
    print(f"  Comm/Sync Total:          {comm_s:10.4f} s ({pct(comm_s):6.2f}%)")
    print("============================================================")
    print("Compute      = non-NCCL kernels (overlapped NCCL is hidden inside).")
    print("NCCL exposed = NCCL kernel time not overlapped with compute.")
    print("GPU Idle     = no kernel running: verbs RDMA waits, CPU")
    print("               post/poll, cudaDeviceSynchronize, launch latency.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path-to-nsys-report.sqlite>")
        sys.exit(1)
    analyze_nvidia_hpl(sys.argv[1])
