from ray.tune import ExperimentAnalysis

import matplotlib.pyplot as plt
import pandas as pd

import os
import pandas as pd
import matplotlib.pyplot as plt

# 🔥 Sostituisci con il percorso della cartella dei risultati di Ray Tune
results_dir = "/home/edoardo_reply/ray_results/DDPG/"

# 🔍 Trova tutti i file 'progress.csv' nella directory
csv_files = []
for root, dirs, files in os.walk(results_dir):
    for file in files:
        if file == "progress.csv":
            csv_files.append(os.path.join(root, file))

# 📢 Stampa i file trovati
print(f"📂 Trovati {len(csv_files)} file CSV")
for f in csv_files:
    print(f)


# 📊 Crea un grafico per ogni file CSV trovato
for csv_path in csv_files:
    try:
        # 🔥 Leggi il CSV
        df = pd.read_csv(csv_path)

        # 📌 Controlla che il file abbia i dati giusti
        if "timesteps_total" not in df.columns or "episode_reward_mean" not in df.columns:
            print(f"⚠️ {csv_path} non ha le colonne richieste, lo salto.")
            continue

        # 📈 Crea il grafico
        plt.figure(figsize=(8, 5))
        plt.plot(df["timesteps_total"], df["episode_reward_mean"], label="Mean Reward")
        plt.xlabel("Timesteps")
        plt.ylabel("Mean Reward")
        plt.title(f"Training Progress: {os.path.basename(os.path.dirname(csv_path))}")
        plt.legend()
        plt.grid()

        # 🔥 Salva il grafico come immagine
        save_path = os.path.join(os.path.dirname(csv_path), "reward_plot.png")
        plt.savefig(save_path)
        print(f"✅ Salvato plot in {save_path}")

        # 📊 Mostra il grafico (opzionale)
        plt.show()

    except Exception as e:
        print(f"❌ Errore con {csv_path}: {e}")



