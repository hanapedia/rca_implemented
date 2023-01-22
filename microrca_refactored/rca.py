import click
from run_rca.rca import Microrca

@click.command("Query traces from jaeger and prometheus, and TraceRCA input format")
@click.option("-d", "--dataset_dir", "dataset_dir", type=str, help="Dataset directory for input files", required=True)
@click.option("-r", "--results_dir", "results_dir", type=str, help="Results directory", required=True)
def main(dataset_dir: str, results_dir: str):
    microrca = Microrca(dataset_dir=dataset_dir, results_dir=results_dir)
    microrca.run()

if __name__ == "__main__":
    main()
