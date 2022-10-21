# Tasks to implement TraceRCA
1. 10/18 Understand the code structure and order of execution for each script
2. 10/18 Understand the data structure that it requires as its input 
3. 10/19 Run the scripts with datasets provided
4. 10/x
  - Deploy distributed tracing
  - Create a script to gather the metrics required and format it into the data structure

## Code Structure
### Order of execution based on the makefile
1. ./src/run_dataset_summary.py
  - INPUT: invocations in pickle
    - file ends in * invo.pkl
  - OUTPUT: logs dataset summary:
    - Number of traces, invocations, abnormal traces and invocations, injected faults, and time range
2. ./src/run_anomaly_detection_collect_result.py
  - used to compare the trace anomaly detections
  - *NOT IN THE PROCESS OF TRACERCA*
  - INPUT: 
    - invocation files: trace_id
    - trace files: trace_id 
    - output file
  - OUTPUT:
    - csv file containing the summery for all of the anomaly detection models
3. ./src/run_anomaly_detection_prepare_model.py
  - applies different anomaly detection models to historical trace and invocation data
  - *NOT IN THE PROCESS OF TRACERCA*
  - INPUT: 
    - invocation files: trace_id
    - trace files: trace_id 
    - output file
  - OUTPUT:
    - pickle file containing results of applying different anomaly detectin models
4. ./src/run_anomaly_detection_invo.py
  - Anomaly detection used in the process of TraceRCA
  - INPUT: 
    - input files
      - `*.pkl`
      - pickle file containing the metrics from the time of fault
    - output file
      - file to dump the results
    - historical data 
      - `*.pkl`
      - pickle file containing the historical metrics
      - not used as the historical data is summerized in cache file
    - useful_features 
      - provided by other script
      - given in a text file
    - cache file
      - `*.pkl`
      - contains the cache for the mean and std of the historical data
  - PROECSS:
    - anomaly_detection_3sigma
      - adds new column to input data df that contains a boolean indicating whether an anomaly was detected or not
    - other two anomaly detections are for reference 
  - OUTPUT: 
    - `output.pkl`
    - output pickle extends input file with a new column indicating whether the invocations conain anomaly
5. ./src/run_selecting_features.py
  - selects useful features for the root cause analysis
  - INPUT: 
    - input files
      - `*.pkl`
      - pickle file containing the metrics from the time of fault
    - output file
      - file to dump the results
    - historical data 
      - `*.pkl`
      - pickle file containing the historical metrics
      - not used as the historical data is summerized in cache file
      - may want to consider using cache file with mean and std like run_anomaly_detection_invo
    - fisher threshold
      - threshold for determining if the feature is useful
    - FEATURES_NAMES
      - defined in trainticket_config.py
  - PROCESS:
    - stderr_criteria checks if the metric values for the feature had significant change at the time of the fault
  - OUTPUT:
    - a text file containing a dictionary with (source, target) as keys and [features] as values
6. ./src/run_anomaly_detection_trace.py *NOT INCLUDED*
7. ./src/run_invo_encoding.py
  - Encodes trace data into individual invocations 
  - INPUT:
    - input pickle file in the format of those found in `../datasets/A/`
  - OUTPUT:
    - output pickle file in format similar to files found in `../datasets/B/` 
8. ./src/run_trace_encoding.py
9. ./src/run_concatenate.py *NOT INCLUDED* *APPLIED TO ALL HISTORICAL DATA*
10. ./src/run_concatenate.py *NOT INCLUDED* *APPLIED TO NORMAL HISTORICAL DATA*
11. ./src/run_localization_collect.py
  - summary of the root cause localization
12. ./src/run_effect_of_trace_localization_collect.py *NOT INCLUDED*
13. ./src/run_effect_of_trace_inject.py *NOT INCLUDED*
14. ./src/run_localization_prepare_model.py *NOT INCLUDED*
15. ./src/run_localization_association_rule_mining_20210516.py
  - INPUT:
    - pickle file containing the metrics for invocation and label for anomaly
14. ./src/run_localization_pagerank.py *NOT INCLUDED*
14. ./src/run_localization_RCSF.py *NOT INCLUDED*
14. ./src/run_localization_MEPFL.py *NOT INCLUDED*
14. ./src/run_localization_microscope.py *NOT INCLUDED*
### Actual order of execution
0. run_invo_encoding.py
1. run_selecting_features.py
- need to encode historical data into cache file containing mean and std
2. run_anomaly_detection_invo.py
3. run_localization_association_rule_mining_20210516.py
### Dataset files
A. Follows the same format in pickle.
  - includes the metrics of the services at the time of the invocation
  - presumed that all the metrics are reported by the request source
