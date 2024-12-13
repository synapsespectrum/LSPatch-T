import os
import yaml
import csv


def extract_results(base_dir, model_name):
    results = {48: [], 96: [], 192: [], 336: [], 720: []}
    for exp_id in os.listdir(base_dir):
        exp_path = os.path.join(base_dir, exp_id)
        if not os.path.isdir(exp_path):
            continue
        meta_file = os.path.join(exp_path, 'meta.yaml')
        if not os.path.exists(meta_file):
            continue
        with open(meta_file, 'r') as f:
            meta = yaml.safe_load(f)

        if model_name.upper() not in meta.get('name', '').upper():
            continue

        input_length = int(meta['name'].split('_')[-2])
        if input_length not in results:
            continue

        for run_id in os.listdir(exp_path):
            run_path = os.path.join(exp_path, run_id)
            if not os.path.isdir(run_path):
                continue
            metrics_path = os.path.join(run_path, 'metrics')
            if not os.path.exists(metrics_path):
                continue
            mse_file = os.path.join(metrics_path, 'mse')
            mae_file = os.path.join(metrics_path, 'mae')
            if os.path.exists(mse_file) and os.path.exists(mae_file):
                with open(mse_file, 'r') as f:
                    mse = float(f.read().split()[1])
                with open(mae_file, 'r') as f:
                    mae = float(f.read().split()[1])
                results[input_length].append((mse, mae))

    # Keep only the best result (lowest MSE) for each output length
    best_results = {}
    for input_length, values in results.items():
        if values:
            best_results[input_length] = min(values, key=lambda x: x[0])

    return best_results


def print_and_save_results(results, model_name, output_file):
    print(f",{model_name},")
    print("Look back,MSE,MAE")
    for output_length in sorted(results.keys()):
        mse, mae = results[output_length]
        print(f"{output_length},{mse:.9f},{mae:.9f}")

    # with open(output_file, 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow([f",{model_name},"])
    #     writer.writerow(["Look back", "MSE", "MAE"])
    #     for output_length in sorted(results.keys()):
    #         mse, mae = results[output_length]
    #         writer.writerow([output_length, f"{mse:.9f}", f"{mae:.9f}"])


if __name__ == "__main__":
    base_dir = "../../../mlruns"
    model_name = input("Enter model name: ")
    output_file = "results.csv"
    results = extract_results(base_dir, model_name)
    print_and_save_results(results, model_name, output_file)
    print(f"Results have been saved to {output_file}")