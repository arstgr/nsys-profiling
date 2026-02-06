# nsys-profiling
Profiling workflow for AI &amp; HPC workloads on NVIDIA GPUs 


## Extract details from NCCL logs
NCCL logs can also contain important details about the initialization, topology, and communication patterns of the workload. To export such details, the training or simulation should be run with 
```
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYSTEMS=COLL,NET,NVLS
export NCCL_DEBUG_TIME=1
```
present in the environment. From the NVIDIA NCCL documentation, NCCL_DEBUG SUBSYSTEM allows the user to filter the output of NCCL_DEBUG into various subsystems. The supported subsystems are INIT, COLL, P2P, SHM, NET, GRAPH, TUNING, ENV, ALLOC, CALL, PROXY, NVLS, BOOTSTRAP, REG, PROFILE, RAS and ALL.

Above set of subsystems provide a minimal set to analyze the communication patterns. For more detailed tracing, INIT and GRAPH are also necessary to obtain a view of the topology. 

With these NCCL environment variables present in the runtime environment, the job exports a log file, 'nccl.log' (the name depends on your job launch settings and can be controlled by the NCCL_DEBUG_FILE environment variable). To analyze the log file, you can use the AWK scripts in the 'scripts' directory. Firts, extract the part of the log containing the iterations (or training steps) intended for the analysis (here it is assumed iterations/steps 45-50), then run
```
./1-separate-steps.awk  nccl.log nccl-l45-50.log
```
This will output a log file containing only the part of the nccl log within the intended range for the analysis (here nccl-l45-50.log). Next, 

Maintained by Amirreza Rastegari
Licensed under the MIT License
