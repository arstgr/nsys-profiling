#!/bin/bash
# Copyright (c) 2026 Amirreza Rastegari
# Licensed under the MIT License.
##note use like ./2-comm_analyzer.awk nccl-45-50.log > nccl-sizes-45-50.log

fl=$1
grep "NCCL INFO" $fl | grep "opCount" | \
awk '{
  op=""; count=0; dtype="";
  # extract op name and numeric datatype
  for (i=1;i<=NF;i++) {
    # op is the thing ending with a colon just after "NCCL" and "INFO"
    #or a word ending in ":" before "opCount" in the nccl log file
    if ($i ~ /:$/ && op=="" && $i != "INFO:") {
      tmp=$i;
      gsub(":", "", tmp);
      op=tmp;
    }
    # count N (next word after "count")
    if ($i == "count") {
      c=$(i+1); gsub(",", "", c); count=c;
    }
    # datatype N (number)
    if ($i == "datatype") {
      d=$(i+1); gsub(",", "", d); dtype=d;
    }
  }

  ##map numeric dtype to element size, this was extracted from nccl 2.28 documentations
  # 0 ncclInt8/ncclChar
  # 1  ncclUint8
  # 2 ncclInt32/ncclInt
  # 3 ncclUint32
  # 4 ncclInt64
  # 5  ncclUint64
  #6  ncclFloat16/ncclHalf
  # 7  ncclFloat32 / ncclFloat
  #8  ncclFloat64/ncclDouble
  # 9  ncclBfloat16
  # 10 ncclFloat8e4m3
  # 11 ncclFloat8e5m2

  esize=0;
  if (dtype==0 || dtype==1) esize=1;                    #int8/uint8
  else if (dtype==2 || dtype==3) esize=4;               # int32/uint32
  else if (dtype==4 || dtype==5) esize=8;               # int64/uint64
  else if (dtype==6 || dtype==9) esize=2;               # fp16/bf16
  else if (dtype==7) esize=4;                           #fp32
  else if (dtype==8) esize=8;                           # fp64
  else if (dtype==10 || dtype==11) esize=1;             # fp8 e4m3/e5m2 (8 bits)

  dtype_name="";
  if (dtype==0) dtype_name="int8";
  else if (dtype==1) dtype_name="uint8";
  else if (dtype==2) dtype_name="int32";
  else if (dtype==3) dtype_name="uint32";
  else if (dtype==4) dtype_name="int64";
  else if (dtype==5) dtype_name="uint64";
  else if (dtype==6) dtype_name="fp16";
  else if (dtype==7) dtype_name="fp32";
  else if (dtype==8) dtype_name="fp64";
  else if (dtype==9) dtype_name="bf16";
  else if (dtype==10) dtype_name="fp8_e4m3";
  else if (dtype==11) dtype_name="fp8_e5m2";

  if (op != "" && count > 0 && esize > 0) {
    bytes = count * esize;
    print op, count, dtype_name, bytes;
  }
}'
