import os
import json
import numpy as np
import argparse

def main(args):
    dir_path = args.dir

    cold_hr_values = []
    warm_hr_values = []
    cold_invalid_values = []
    warm_invalid_values = []

    print(f"Processing directory: {os.getcwd()}")
    
    # Iterate over all JSON files in the specified directory
    for filename in os.listdir(dir_path):
        if filename.endswith(".json"):
            file_path = os.path.join(dir_path, filename)
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    if "HR@1" in data:
                        if "cold" in filename:
                            cold_hr_values.append(data["HR@1"])
                        elif "warm" in filename:
                            warm_hr_values.append(data["HR@1"])

                    if "invalid_output@1" in data:
                        if "cold" in filename:
                            cold_invalid_values.append(data["invalid_output@1"])
                        elif "warm" in filename:
                            warm_invalid_values.append(data["invalid_output@1"])

            except Exception as e:
                print(f"Error reading {filename}: {e}")



    # Compute statistics for cold cases
    cold_results = {}
    if cold_hr_values:
        cold_results["average_HR@1_cold"] = np.mean(cold_hr_values)
        cold_std_hr = np.std(cold_hr_values)
        if np.isclose(cold_std_hr, 0):
            cold_std_hr = 0
        cold_results["std_HR@1_cold"] = cold_std_hr
    else:
        cold_results["average_HR@1_cold"] = None
        cold_results["std_HR@1_cold"] = None


    if cold_invalid_values:
        cold_results["invalid_output@1_cold"] = np.mean(cold_invalid_values)
        cold_std_invalid = np.std(cold_invalid_values)
        if np.isclose(cold_std_invalid, 0):
            cold_std_invalid = 0
        cold_results["std_invalid@1_cold"] = cold_std_invalid
    else:
        cold_results["invalid_output@1_cold"] = None
        cold_results["std_invalid@1_cold"] = None
    

    # Compute statistics for warm cases
    warm_results = {}
    if warm_hr_values:
        warm_results["average_HR@1_warm"] = np.mean(warm_hr_values)
        warm_std_hr = np.std(warm_hr_values)
        if np.isclose(warm_std_hr, 0):
            warm_std_hr = 0
        warm_results["std_HR@1_warm"] = warm_std_hr
    else:
        warm_results["average_HR@1_warm"] = None
        warm_results["std_HR@1_warm"] = None
    


    if warm_invalid_values:
        warm_results["invalid_output@1_warm"] = np.mean(warm_invalid_values)
        warm_std_invalid = np.std(warm_invalid_values)
        if np.isclose(warm_std_invalid, 0):
            warm_std_invalid = 0
        warm_results["std_invalid@1_warm"] = warm_std_invalid
    else:
        warm_results["invalid_output@1_warm"] = None
        warm_results["std_invalid@1_warm"] = None
    

    # Save results
    results = {**cold_results, **warm_results}
    results_file = os.path.join(dir_path, "hr1_statistics.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {results_file}")
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute average and standard deviation of HR@1 from JSON files in a directory.")
    parser.add_argument('--dir', type=str, help="Directory containing JSON files.", default="./results/ml-1m/")
    
    args = parser.parse_args()
    main(args)