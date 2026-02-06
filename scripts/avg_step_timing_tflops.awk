#!/bin/bash
logfile="$1"

awk '
  /Training epoch/ && /iteration/ {
    iter = -1
    for (i=1; i<=NF; i++) {
      if ($i == "iteration") {
        split($(i+1), a, "/")
        iter = a[1] + 0
        break
      }
    }
    if (iter >= 45 && iter <= 49) {
      t = f = ""
      for (i=1; i<=NF; i++) {
        if ($i == "train_step_timing" && $(i+1) == "in" && $(i+2) == "s:") {
          t = $(i+3)
          gsub("\\|", "", t)
        }
        if ($i == "TFLOPS_per_GPU:") {
          f = $(i+1)
          gsub("\\|", "", f)
        }
      }
      if (t != "" && f != "") {
        n++
        tsum += t
        fsum += f
      }
    }
  }
  END {
    if (n > 0) {
      printf "Steps: %d (iterations 45-49)\n", n
      printf "Average train_step_timing (s): %.6f\n", tsum / n
      printf "Average TFLOPS_per_GPU:        %.6f\n", fsum / n
    } else {
      print "No matching iterations found in range 45-49"
    }
  }
' "$logfile"
