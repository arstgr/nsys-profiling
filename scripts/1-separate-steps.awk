#!/bin/bash
# Copyright (c) 2026 Amirreza Rastegari
# Licensed under the MIT License.
##note use like ./separate-steps.awk  nccl.log nccl-45-50.log

fl=$1
ou=$2

awk '
  /Training epoch/ && /iteration/ {
    for (i=1; i<=NF; i++) {
      if ($i == "iteration") {
        split($(i+1), it, "/");   # it[1] = iteration_number, it[2] = total
        iter = it[1] + 0;
        break;
      }
    }
  }

  # prints lines that are either 
  #  iterations within [45,50)
  #  or NCCL lines within [45,50)
  {
    if (iter > 44 && iter <= 50) {
      print $0;
    }
  }
' $fl > $ou
