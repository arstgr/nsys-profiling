# nsys-profiling
Profiling workflow for AI &amp; HPC workloads on NVIDIA GPUs 

Nvidia's nsight system is used for profiling workloads on NVIDIA's GPUs. The profiler can be run using
```
nsys profile <args> <exec> <parameters>
```
where the output is one (or several) nsys representation files. 

To view a profiling summary, try
```
nsys stats <>.nsys-rep 
```
Besides a summary table of the profiles, this also exports a sqlite database that can be used for further analyses using sql queries. Script './scripts/analyze_nsys_finegrained.py' provides a basic analysis of the results, demonstrating a breakdown of the performance into compute, communication and memory operations. 

To obtain this breakdown, adjust the name of the sql file obtained from your nsys representation file, at the last line of './scripts/analyze_nsys_finegrained.py', 'analyze_distributed_performance("profile_16124_0_0.sqlite")", then try
```
python3 ./scripts/analyze_nsys_finegrained.py
```

## Extract details from NCCL logs
NCCL logs can also contain important details about the initialization, topology, and communication patterns of the workload. To export such details, the training or simulation should be run with 
```
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL,NET,NVLS
export NCCL_DEBUG_TIME=1
```
present in the environment. From the NVIDIA NCCL documentation, NCCL_DEBUG SUBSYSTEM allows the user to filter the output of NCCL_DEBUG into various subsystems. The supported subsystems are INIT, COLL, P2P, SHM, NET, GRAPH, TUNING, ENV, ALLOC, CALL, PROXY, NVLS, BOOTSTRAP, REG, PROFILE, RAS and ALL.

Above set of subsystems provide a minimal set to analyze the communication patterns. For more detailed tracing, INIT and GRAPH are also necessary to obtain a view of the topology. 

With these NCCL environment variables present in the runtime environment, the job exports a log file, 'nccl.log' (the name depends on your job launch settings and can be controlled by the NCCL_DEBUG_FILE environment variable). To analyze the log file, you can use the AWK scripts in the 'scripts' directory. Firts, extract the part of the log containing the iterations (or training steps) intended for the analysis (here it is assumed iterations/steps 45-50), then run
```
./scripts/1-separate-steps.awk  nccl.log nccl-l45-50.log
```
This will output a log file containing only the part of the nccl log within the intended range for the analysis (here nccl-l45-50.log). Next, run
```
./scripts/2-comm_analyzer.awk nccl-l45-50.log > nccl-sizes-45-50.log
```
This extracts all the nccl communication calls, with their message sizes and number of calls into a file named 'nccl-sizes-45-50.log'. Next, run
```
./scripts/3-per-op-stats.awk nccl-sizes-45-50.log
```
This presents an aggregated view of all NCCL calls within the intended iteration range, with the range of message sizes. Next, run
```
./scripts/4-histogram.awk nccl-sizes-45-50.log
```
This will present a histogram of the NCCL calls, with a breakdown of their message sizes into '64KB < msg < 1MB', '1MB < msg < 16MB', 'msg > 16MB', along with the number of calls per GPU. Next, run
```
./scripts/5-aggregate_comms.awk nccl-sizes-45-50.log
```
This presents an aggregated view of all NCCL calls, message sizes and their data types. 

Additionally, you can extract basic data regarding the average training step size and average gpu TFLOPs from the logs (averaged over the intended iterations), by running
```
./scripts/avg_step_timing_tflops.awk nccl.log
```

Maintained by Amirreza Rastegari
Licensed under the MIT License
