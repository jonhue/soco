# Analysis

## Intermediate job format

Jobs are represented as a table with the following columns:

* `submission_time` - arrival time of the job
* `start_time` - time when job was started to be processed
* `end_time` - time when job processing finished
* `job_type` - index of job type
* `source` - index of geographical source
* `server_type` - index of server type that the job was processed on
* `location` - index of data center location that the job was processed in

Times are given in seconds beginning from the start of the trace. Indexes begin from `0`.

<!-- ### Cluster

* `server_types`
  * `utilization` - utilization for each machine over time
* `locations` - for each server type number of machines in each location -->