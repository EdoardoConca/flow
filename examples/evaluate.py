import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse

def plot_cumulative_rewards(progress_files, labels, save_path=None):
    """
    Plots the cumulative reward over training iterations for multiple models.

    Parameters:
    - progress_files (dict): Dictionary with key = model name, value = path to the progress.csv file.
    - labels (dict): Dictionary with key = model name, value = label to display in the plot.
    - save_path (str, optional): If provided, saves the plot at this location.

    Output:
    - Displays the matplotlib plot.
    """
    plt.figure(figsize=(10, 6))

    for model_name, file_path in progress_files.items():
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        df = pd.read_csv(file_path)

        # Identify the appropriate reward column
        possible_columns = ["episode_reward_mean", "cumulative_reward", "mean_reward"]
        reward_col = next((col for col in possible_columns if col in df.columns), None)

        if reward_col is None:
            print(f"No valid reward column found in {file_path}, skipping.")
            continue

        if "training_iteration" not in df.columns:
            print(f"'training_iteration' column missing in {file_path}, skipping.")
            continue

        # Plot cumulative reward
        plt.plot(df["training_iteration"], df[reward_col], label=labels.get(model_name, model_name))

    plt.xlabel("Number of Iterations")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward Comparison Across Models")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Graph saved at: {save_path}")

    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot cumulative rewards from Ray Tune progress.csv files.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the output plot.")
    
    args = parser.parse_args()

    # Define paths for different models (Modify paths as needed)
    progress_files = {
        "PPO": "/home/edoardo_reply/ray_results/PPO_Intersection/PPO_0_2025-02-24_19-55-19oio1vqy5/progress.csv",
        "A3C": "/home/edoardo_reply/ray_results/A3C_Intersection/A3C_0_2025-02-24_19-16-12288gapnb/progress.csv",
        "DDPG": "/home/edoardo_reply/ray_results/DDPG_Intersection/DDPG_0_2025-02-24_19-27-0460bqplsw/progress.csv"
    }

    # Model labels for better readability
    labels = {
        "PPO": "Proximal Policy Optimization",
        "A3C": "Asynchronous Advantage Actor-Critic",
        "DDPG": "Deep Deterministic Policy Gradient"
    }

    plot_cumulative_rewards(progress_files, labels, save_path=args.save_path)

if __name__ == "__main__":
    main()
