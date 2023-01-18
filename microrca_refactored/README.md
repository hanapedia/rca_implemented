# Refactoring MicroRCA
## Steps
### Break into query and rca parts
- query: queries required metrics from prometheus and store as pickle file
- rca: reads the data from pickle and execute rca
1. purely query only methods
  - `self.latency_source_50`
    - creates `pd.DataFrame` and converts it to csv `latency_source_50.csv`
    - returns `df`
  - `self.latency_destination_50`
    - creates `pd.DataFrame` and converts it to csv `latency_destination_50.csv`
    - returns `df`
  - `self.svc_metrics`
    - calls multiple query methods
    - creates `pd.DataFrame` and converts it to csv per service: `{sevice}.csv`
    - does not return anything
2. query and rca mixed methods
  - `self.mpg`
    - creates `pd.DataFrame` and converts it to csv: `mpg.csv`
    - returns `DG`
3. purely rca only methods
  - `self.birch_ad_with_smoothing`
  - `anomaly_subgraph`
    - `node_weight`: reads each `{service}.csv`
    - `svc_personalization`: reads each `{service}.csv`

### node_dict parameter
- need to pass a dict where 
  - the keys represent hostname of the node
  - the values represent IP address of the node
- required for the conversion between hostname and ip


