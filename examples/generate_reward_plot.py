# python script.py --results_dir /home/User/ray_results/PPO --save_dir /home/User/plots
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def find_progress_files(results_dir):
    """Find all progress.csv files in the given results directory."""
    csv_files = []
    for root, _, files in os.walk(results_dir):
        for file in files:
            if file == "progress.csv":
                csv_files.append(os.path.join(root, file))
    return csv_files

def plot_rewards(csv_files, save_dir):
    """Generate and save reward plots for each model found."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    print(f"Found {len(csv_files)} CSV files.")

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)

            if "timesteps_total" not in df.columns or "episode_reward_mean" not in df.columns:
                print(f"{csv_path} is missing required columns, skipping.")
                continue

            plt.figure(figsize=(8, 5))
            plt.plot(df["timesteps_total"], df["episode_reward_mean"], label="Mean Reward", color='b')
            plt.xlabel("Timesteps")
            plt.ylabel("Mean Reward")
            plt.title(f"Training Progress: {os.path.basename(os.path.dirname(csv_path))}")
            plt.legend()
            plt.grid()

            # Save the plot in the specified directory
            save_path = os.path.join(save_dir, f"{os.path.basename(os.path.dirname(csv_path))}_reward_plot.png")
            plt.savefig(save_path)
            print(f"Plot saved at: {save_path}")

            plt.close()  

        except Exception as e:
            print(f"Error with {csv_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Plot Training Reward from Ray Tune progress.csv files.")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing Ray Tune results.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the plots.")
    args = parser.parse_args()

    csv_files = find_progress_files(args.results_dir)
    plot_rewards(csv_files, args.save_dir)

if __name__ == "__main__":
    main()
