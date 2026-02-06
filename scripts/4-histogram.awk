#!/bin/bash

##note run like ./4-histogram.awk nccl-sizes-45-50.log

fl=$1

awk '{
  op=$1; bytes=$4;
  if (bytes < 64*1024) bucket="<64KB";
  else if (bytes < 1024*1024) bucket="64KB-1MB";
  else if (bytes < 16*1024*1024) bucket="1MB-16MB";
  else bucket=">16MB";
  key=op":"bucket;
  cnt[key] += 1;
}
END {
  for (k in cnt) {
    print k, cnt[k];
  }
}' $fl | sort
