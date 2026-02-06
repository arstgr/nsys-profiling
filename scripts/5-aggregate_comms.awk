#!/bin/bash

##note run like ./5-aggregate_comms.awk nccl-sizes-45-50.log

fl=$1

cat $fl | sort | uniq -c | \
    awk 'BEGIN {
       printf "%-10s %-15s %-12s %-10s %-12s\n", "calls", "op", "count", "dtype", "bytes_per_call";
     }
     {
       # $1=calls, $2=op, $3=count, $4=dtype, $5=bytes
       printf "%-10s %-15s %-12s %-10s %-12s\n", $1, $2, $3, $4, $5;
     }'
