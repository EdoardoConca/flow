import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse

def plot_training_metrics(progress_files, labels, save_dir=None):
    """
    Plots multiple training metrics over iterations for different models.

    Parameters:
    - progress_files (dict): Dictionary with key = model name, value = path to the progress.csv file.
    - labels (dict): Dictionary with key = model name, value = label to display in the plot.
    - save_dir (str, optional): If provided, saves the plots in this directory.

    Output:
    - Displays and optionally saves multiple matplotlib plots.
    """
    metrics = {
        "episode_reward_mean": "Episode Reward Mean",
        "custom_metrics/fuel_consumption_mean": "Fuel Consumption",
        "custom_metrics/total_collisions_mean": "Total Collisions",
        "custom_metrics/avg_speed_mean": "Average Speed",
        "custom_metrics/throughput_efficency_mean": "Throughput Efficiency",
        "custom_metrics/avg_accel_avs_mean": "Average AV Acceleration"
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for metric, title in metrics.items():
        plt.figure(figsize=(10, 6))

        for model_name, file_path in progress_files.items():
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue

            df = pd.read_csv(file_path)

            if "training_iteration" not in df.columns or metric not in df.columns:
                print(f"Missing column '{metric}' in {file_path}, skipping.")
                continue

            plt.plot(df["training_iteration"], df[metric], label=labels.get(model_name, model_name))

        plt.xlabel("Number of Iterations")
        plt.ylabel(title)
        plt.title(f"{title} Across Training Iterations")
        plt.legend()
        plt.grid(True)

        if save_dir:
            save_path = os.path.join(save_dir, f"{metric.replace('/', '_')}.png")
            plt.savefig(save_path, dpi=300)
            print(f"Graph saved at: {save_path}")

        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot multiple training metrics from Ray Tune progress.csv files.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the output plots.")
    
    args = parser.parse_args()

    # Define paths for different models (Modify paths as needed)
    progress_files = {
        "PPO": "/home/edoardo_reply/ray_results/PPO_Intersection_new_env/PPO_0_2025-02-27_03-11-15k1ygvl9h/progress.csv",
        "A3C": "/home/edoardo_reply/ray_results/A3C_Intersection_new_env/A3C_0_2025-02-27_02-28-25sujo_sjr/progress.csv",
        "DDPG": "/home/edoardo_reply/ray_results/DDPG_Intersection_new_env/DDPG_0_2025-02-27_02-48-02xzfv3wuj/progress.csv"
    }

    # Model labels for better readability
    labels = {
        "PPO": "Proximal Policy Optimization",
        "A3C": "Asynchronous Advantage Actor-Critic",
        "DDPG": "Deep Deterministic Policy Gradient"
    }

    plot_training_metrics(progress_files, labels, save_dir=args.save_dir)

if __name__ == "__main__":
    main()
