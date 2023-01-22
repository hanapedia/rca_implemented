from pathlib import Path
import click
import time

from query.query import MicroRCAQuery
from query._utils import load_node_dict

@click.command("Query traces from jaeger and prometheus, and TraceRCA input format")
@click.option("-o", "--output_dir", "output_dir", type=str, help="Output directory for resulting files", required=True)
@click.option("-p", "--prom_url", "prom_url", type=str, help="Output directory for resulting files", default="http://localhost:9090/api/v1/query")
@click.option("-n", "--namespace", "namespace", type=str, help="Namespace of the application", required=True)
@click.option("-e", "--end_time", "end_time", type=int, help="End time in Unix seconds", default=-1)
@click.option("-l", "--len_seconds", "len_seconds", type=int, help="Length in seconds", default=180)
@click.option("-m", "--metric_step", "metric_step", type=str, help="Prometheus query metric step", default="14s")
@click.option("-d", "--node_dict_path", "node_dict_path", type=str, help="Path node dict csv", default="")
def main(output_dir, prom_url, namespace, end_time, len_seconds, metric_step, node_dict_path):
    output_dir = Path(output_dir)

    if end_time < 0:
        end_time = time.time()

    if node_dict_path == "":
        node_dict_path = output_dir / "node_dict.csv"
    node_dict = load_node_dict(node_dict_path)

    mq = MicroRCAQuery(
            prom_url=prom_url, 
            namespace=namespace,
            fault_dir=output_dir,
            len_seconds=len_seconds,
            end_time= end_time,
            metric_step=metric_step,
            node_dict=node_dict
            )

    mq.latency_50(reporter="source")
    mq.latency_50(reporter="destination")
    mq.svc_metrics()
    mq.mpg()
    print("query complete")

if __name__ == "__main__":
    main()
