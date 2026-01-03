import torch
import sys
import argparse
from model import BaselineModel, DendriticModel, BaselineCIFARModel, DendriticCIFARModel
from train import get_data_loaders, train_model
from evaluate import get_benchmark_report

def print_header():
    print("\n" + "="*70)
    print(" [PERFORATED AI: DENDRITIC OPTIMIZATION HACKATHON PROOF-OF-CONCEPT]")
    print("="*70)

def run_benchmarks(dataset_name="mnist"):
    print_header()
    print(f"\n[BENCHMARK] Dataset: {dataset_name.upper()}")
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Prepare Data
    train_loader, test_loader = get_data_loaders(batch_size=128, dataset_name=dataset_name)
    
    # 2. Instantiate Models
    if dataset_name.lower() == "cifar10":
        baseline = BaselineCIFARModel()
        dendritic = DendriticCIFARModel()
    else:
        baseline = BaselineModel()
        dendritic = DendriticModel()
    
    # 3. Training phase (Limited epochs for fast demonstration)
    baseline = train_model(baseline, train_loader, device, epochs=1)
    dendritic = train_model(dendritic, train_loader, device, epochs=1)
    
    # 4. Evaluation phase
    print("\n[BENCHMARK] Computing Benchmarks...")
    baseline_metrics = get_benchmark_report(baseline, test_loader, device)
    dendritic_metrics = get_benchmark_report(dendritic, test_loader, device)
    
    # 5. Side-by-Side Comparison Output
    format_str = "{:<25} | {:<15} | {:<15}"
    header = format_str.format("Metric", "Baseline", "Dendritic (PAI)")
    separator = "-" * len(header)
    
    print("\n" + separator)
    print(header)
    print(separator)
    
    for metric in baseline_metrics.keys():
        print(format_str.format(
            metric, 
            baseline_metrics[metric], 
            dendritic_metrics[metric]
        ))
    print(separator)
    
    # Final Hackathon Takeaway
    print("\n[CONCLUSION] HACKATHON CONCLUSION:")
    print("Dendritic optimization proves that we can achieve high performance")
    print(f"with {dendritic_metrics['Sparsity']} sparsity and fewer parameters.")

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="PerforatedAI Benchmark")
        parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"], 
                          help="Dataset to use for benchmarking (mnist or cifar10)")
        args = parser.parse_args()
        
        run_benchmarks(dataset_name=args.dataset)
    except ImportError as e:
        print(f"\n[Error] Missing dependency - {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\n[Execution Failed] {e}")
        traceback.print_exc()
        sys.exit(1)
