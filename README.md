# FATIGUE_GR

FATIGUE-GR is a multimodal dataset designed for fatigue-aware gesture recognition in extended human-computer interaction. The data was collected from 41 participants using Delsys Avanti EMG and IMU sensors while interacting in a virtual reality (VR) environment. Participants played five VR games, each controlled by a distinct hand gesture: air tap, swipe, pinch, fist, and grab. Each game lasted up to 20 minutes, and participants reported their subjective fatigue every 20 seconds using the Borg CR scale (0–10).

The data collection was conducted in two sessions, with:
1. A 10-minute break between consecutive gestures, and
2. A minimum 2-hour gap between the two sessions to reduce carryover fatigue.

This setup allows researchers to analyze fatigue progression and gesture recognition over time in realistic VR interactions.

<b> How was data collected? </b> <br>
```
https://drive.google.com/drive/u/0/folders/19TKVhO4zxTrM2tD1ldGuYwbn-ayorBiR
```
Above is the link to unity environments used to collect the data

<b>How to replicate our results?</b><br>

Clone the dataset repository from Hugging Face (Make sure you have 150 gbs of storage available)
```
git clone https://huggingface.co/datasets/Kirti0111/emg-imu-fatigue-aware-gesture-dataset
```

Download the Code
```
git clone https://github.com/EternalBlissard/FATIGUE_GR.git
```

Configure Paths (All data under in the code)
```python
#TODO: Paths
```

Setup the environment
```
cd FATIGUE_GR/
conda env create -f environment.yml
```

Have a wandb key and add it to your code
```python
import os
os.environ["WANDB_API_KEY"] = "your_wandb_api_key"
```
OR <br>
Export to terminal
```
export WANDB_API_KEY=your_wandb_api_key
```
Use the hyperameters already set to reproduce the results change to see our justification
