from invo_encoding import train_ticket_invo_encoding_main
from selecting_features import selecting_feature_main
from anomaly_detection_invo import invo_anomaly_detection_main
from localization_association_rule_mining import localization_main
from trainticket_config import TrainTicketConfig
from pathlib import Path

class TraceRCA:
    def __init__(self, path_root, enable_all_features=True):
        self.path_root = path_root
        self.config = TrainTicketConfig(enable_all_features)

    def trace_to_invo(self, input_path, output_path):
        # input_path = Path(input_path)
        # output_path = Path(output_path)
        train_ticket_invo_encoding_main(input_path, output_path, self.config)

    def tracerca(self, input_path, output_path, history_path, log_file, fisher_threshold=1.0, ad_threshold=1.0, min_support_rate=0.1, quiet=False, k=100):
        sf_output = Path(self.path_root + '/features')
        sf_cache_output = Path(self.path_root + '/cache.pkl')
        sf_output.parent.mkdir(parents=True, exist_ok=True)
        sf_cache_output.parent.mkdir(parents=True, exist_ok=True)
        selecting_feature_main(input_file=input_path, output_file=sf_output, output_cache_file=sf_cache_output,  history=history_path, fisher_threshold=fisher_threshold, config=self.config)

        ad_output = Path(self.path_root + '/anomalies.pkl')
        invo_anomaly_detection_main(input_file=input_path, output_file=ad_output, useful_feature=sf_output, cache_file=sf_cache_output, main_threshold=ad_threshold, config=self.config)

        localization_main(input_file=ad_output, output_file=output_path, min_support_rate=min_support_rate, quiet=quiet, k=k, log_file=log_file)

# if __name__ == "__main__":
#     tr = TraceRCA('./test')
    # input_path = './original/datasets/test/anomaly_simple_name.pkl'
    # output_path = './test/test_results.pkl'
    # history_path = './original/datasets/test/history.pkl'
    # tr.tracerca(input_path=input_path, output_path=output_path, history_path=history_path)
