#!/bin/bash
# Copyright (c) 2026 Amirreza Rastegari
# Licensed under the MIT License.
##note run like ./3-per-op-stats.awk nccl-sizes-45-50.log

fl=$1

awk '{
  op=$1; bytes=$4;
  count[op] += 1;
  total[op] += bytes;
  if (!(op in minb) || bytes < minb[op]) minb[op]=bytes;
  if (!(op in maxb) || bytes > maxb[op]) maxb[op]=bytes;
}
END {
  for (op in count) {
    avg = total[op] / count[op];
    print op, "calls=" count[op],
              "min=" minb[op],
              "avg=" avg,
              "max=" maxb[op],
              "total=" total[op];
  }
}' $fl
