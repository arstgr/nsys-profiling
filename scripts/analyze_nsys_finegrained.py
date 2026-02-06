# Copyright (c) 2026 Amirreza Rastegari
# Licensed under the MIT License.

import sqlite3

def merge_intervals(intervals):
    """Merge overlapp intervals and return total covered duration"""
    valid = [(s, e) for s, e in intervals if s is not None and e is not None and e > s]
    if not valid:
        return 0
    valid.sort(key=lambda x: x[0])

    merged_total = 0
    curr_start, curr_end = valid[0]

    for next_start, next_end in valid[1:]:
        if next_start <= curr_end:
            if next_end > curr_end:
                curr_end = next_end
        else:
            merged_total += (curr_end - curr_start)
            curr_start, curr_end = next_start, next_end

    merged_total += (curr_end - curr_start)
    return merged_total

def analyze_distributed_performance(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # duration of profiling session 
    cur.execute("SELECT startTime, stopTime FROM ANALYSIS_DETAILS LIMIT 1")
    row = cur.fetchone()
    if not row:
        raise RuntimeError("ANALYSIS_DETAILS is empty")
    session_start, session_end = row
    total_time_ns = session_end - session_start

    # duration of NCCL kernels (on device), assumes this is comm time
    cur.execute("""
        SELECT k.start, k.end
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
        JOIN StringIds AS s ON k.demangledName = s.id
        WHERE s.value LIKE '%nccl%'
    """)
    comm_kernels = cur.fetchall()
    comm_time_ns = merge_intervals(comm_kernels)

    # duration of memory ops, for now assumes memcpy and memset (on device)
    cur.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY")
    memcpy_intervals = cur.fetchall()

    cur.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMSET")
    memset_intervals = cur.fetchall()

    mem_intervals = memcpy_intervals + memset_intervals
    mem_time_ns = merge_intervals(mem_intervals)

    # duration of non-NCCL, non-obvious-mem kernels
    # should be adjusted after reviewing kernels in the db/code
    cur.execute("""
        SELECT k.start, k.end
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
        JOIN StringIds AS s ON k.demangledName = s.id
        WHERE s.value NOT LIKE '%nccl%'
          AND s.value NOT LIKE '%memcpy%'
          AND s.value NOT LIKE '%memset%'
          AND s.value NOT LIKE '%mem_copy%'
    """)
    compute_kernels = cur.fetchall()
    compute_time_ns = merge_intervals(compute_kernels)

    # gpu active vs idle time
    # here active means all kernels + memcpy/memset
    cur.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL")
    all_kernels = cur.fetchall()
    all_active_intervals = all_kernels + mem_intervals  # include memcpys/memsets
    gpu_active_ns = merge_intervals(all_active_intervals)
    gpu_idle_ns = total_time_ns - gpu_active_ns

    # overlap times
    # compute n communication
    comp_comm_union = merge_intervals(comm_kernels + compute_kernels)
    comp_comm_overlap = (compute_time_ns + comm_time_ns) - comp_comm_union

    # compute n mem
    comp_mem_union = merge_intervals(compute_kernels + mem_intervals)
    comp_mem_overlap = (compute_time_ns + mem_time_ns) - comp_mem_union

    #communication n mem
    comm_mem_union = merge_intervals(comm_kernels + mem_intervals)
    comm_mem_overlap = (comm_time_ns + mem_time_ns) - comm_mem_union

    to_sec = 1e9
    print(f"\n{'Distributed Training Metric':<35} | {'Duration (s)':>15}")
    print("-" * 60)
    print(f"{'Total Session Time':<35} | {total_time_ns / to_sec:15.6f}")
    print(f"{'GPU Active (any work)':<35} | {gpu_active_ns / to_sec:15.6f}")
    print(f"{'GPU Idle Time':<35} | {gpu_idle_ns / to_sec:15.6f}")
    print("-" * 60)
    print(f"{'Compute (FLOP-ish kernels)':<35} | {compute_time_ns / to_sec:15.6f}")
    print(f"{'Communication (NCCL kernels)':<35} | {comm_time_ns / to_sec:15.6f}")
    print(f"{'Memory ops (memcpy/memset)':<35} | {mem_time_ns / to_sec:15.6f}")
    print("-" * 60)
    print(f"{'Compute ∩ Comm overlap':<35} | {comp_comm_overlap / to_sec:15.6f}")
    print(f"{'Compute ∩ Mem overlap':<35} | {comp_mem_overlap / to_sec:15.6f}")
    print(f"{'Comm ∩ Mem overlap':<35} | {comm_mem_overlap / to_sec:15.6f}")
    print("-" * 60)

    if total_time_ns > 0:
        util = (gpu_active_ns / total_time_ns) * 100.0
        print(f"Overall GPU Utilization: {util:.2f}%")

    conn.close()

if __name__ == "__main__":
    analyze_distributed_performance("profile_30_0_0.sqlite")
