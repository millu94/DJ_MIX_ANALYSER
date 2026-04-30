utf-8
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from pathlib import Path



def generate_analysis_plot(csv_path="DJ_MIX_ANALYSER/user_submission/RA.1008.csv"):

    # load the data

    path = Path(csv_path)

    if not path.exists():

        print(f"❌ File not found: {csv_path}")

        return

    

    df = pd.read_csv(path)

    

    # preprocess: convert "xx.x%" strings to floats

    def parse_pct(val):

        if pd.isna(val) or val == "": return np.nan

        return float(str(val).replace('%', ''))



    df['V1'] = df['VER 1 Confidence'].apply(parse_pct)

    df['V2'] = df['VER 2 Confidence'].apply(parse_pct)

    

    # setup the plot

    plt.figure(figsize=(18, 8))

    

    # plot ver 2 (sliding) as a continuous line

    plt.plot(df['Seconds'], df['V2'], label='VER 2 (Sliding 10s Window / 5s Hop)', 

             color='#1DB954', linewidth=2, alpha=0.8) # spotify green

    

    # plot ver 1 (fixed) as points (since it's only every 10s)

    v1_df = df.dropna(subset=['V1'])

    plt.scatter(v1_df['Seconds'], v1_df['V1'], label='VER 1 (Fixed 10s Window)', 

                color='#191414', s=30, zorder=5) # off-black

    

    # highlight the sensitivity threshold

    THRESHOLD = 35

    plt.axhline(y=THRESHOLD, color='red', linestyle='--', alpha=0.6, label=f'Sensitivity Threshold ({THRESHOLD}%)')

    

    # shade areas above threshold (critical zones)

    plt.fill_between(df['Seconds'], df['V2'], THRESHOLD, where=(df['V2'] >= THRESHOLD),

                     color='red', alpha=0.15, label='Potential Sloppy Transition')



    # annotate shift points (statistical dilution)

    df['Note'] = df['Note'].astype(str)

    shifts = df[df['Note'].str.contains('SHIFT', na=False)]

    for i, row in shifts.iterrows():

        plt.annotate('⚠️ SHIFT', (row['Seconds'], row['V2']),

                     textcoords="offset points", xytext=(0,10), ha='center',

                     fontsize=9, color='darkorange', fontweight='bold')



    # formatting

    plt.title(f"DJ Mix Analysis Timeline: {path.stem}", fontsize=16, pad=20)

    plt.xlabel("Time (Seconds)", fontsize=12)

    plt.ylabel("Confidence of 'Bad' Transition (%)", fontsize=12)

    plt.ylim(0, 100)

    plt.grid(axis='y', linestyle=':', alpha=0.5)

    

    # legend

    plt.legend(loc='upper right', frameon=True, shadow=True)

    

    # save the output

    output_dir = Path("notes/graphs")

    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / f"analysis_plot_{path.stem}.png"

    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    

    print(f"✅ Analysis plot saved to: {save_path}")



if __name__ == "__main__":

    generate_analysis_plot()

