import glob
import os
import re
import pandas as pd


targets = [
    'alpha', 'cv', 'g298',
    'gap', 'h298', 'homo',
    'lumo', 'mu', 'r2',
    'u0', 'u298', 'zpve'
]

all_df = []
for tgt in targets:
    ckpt_files = list(map(os.path.basename, glob.glob(f'./checkpoints_QM9-{tgt}_**/*.pt')))
    ckpt_files.sort()
    print()
    print(f"==============={tgt}===============")
    df_seed = []
    for f in ckpt_files:
        seed = f.split('_')[1][4:]
        epoch = f.split('_')[3].split('=')[-1]
        print()
        print(f"**Seed: {seed}, Epoch: {epoch}**")

        logs_seed = list(glob.glob(f'logs/*.log'))
        print(len(logs_seed))
        all_metrics = []
        for log in logs_seed:
            with open(log) as f:
                log_content = f.read()
                if f"Target:		 {tgt}" in log_content and f"Seed:		 {seed}" in log_content:
                    print('THIS:', log)

                    idx = log_content.find(f'=====Epoch [{epoch}/300]=====')
                    test_content = log_content[idx:idx+500]
                    test_mae = re.search(r'\[TEST\] Evaluation MAE:\s+([-+]?(?:\d+\.\d*|\d*\.\d+|\d+))', test_content).group(1)
                    print(f'Seed {seed}:')
                    print(test_mae)

                    all_metrics.append(float(test_mae))

        df = pd.DataFrame({
            tgt: all_metrics
        })
        df_seed.append(df)
    df = pd.concat(df_seed, axis=0)
    all_df.append(df.reset_index(drop=True))
    print(df)

df_results = pd.concat(all_df, axis=1)
print(df_results)
df_results.to_csv('results.csv', index=False)