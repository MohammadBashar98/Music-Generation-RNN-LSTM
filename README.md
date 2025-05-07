# 🎼 MusicGen: Music Generation Using RNN-LSTM with Self-Attention

**MusicGen** is an AI-driven music generation system designed using Recurrent Neural Networks (RNN), Long Short-Term Memory (LSTM) layers, and a Self-Attention mechanism. It generates coherent and stylistically rich MIDI music by learning from the Maestro dataset.

## 📘 Project Overview

This project bridges creativity and deep learning by generating piano music with:
- Bidirectional LSTM layers
- Self-Attention for enhanced sequence modeling
- Custom loss functions for step and duration
- Feature extraction via PrettyMIDI

## 📁 Dataset

- **Name**: MAESTRO v3.0.0
- **Description**: Classical piano MIDI performances
- **Source**: [Magenta Maestro Dataset](https://magenta.tensorflow.org/datasets/maestro)

## 🧠 Model Architecture

```python
Bidirectional LSTM (128 units, LeakyReLU) → Dropout
→ Bidirectional LSTM (128 units, LeakyReLU) → Dropout
→ Self-Attention → Concatenate → BatchNorm
→ Flatten → Dense outputs for pitch, step, duration, and velocity
```

## 📦 Dependencies

```bash
pip install pretty_midi tensorflow pyfluidsynth numpy matplotlib
```

## 🚀 How to Run

Run the notebook:
```bash
jupyter notebook musicgen.ipynb
```
Train the model and generate new MIDI music.

## 🎵 Music Generation

- Input: MIDI files → Feature extraction (pitch, duration, velocity)
- Output: Generated MIDI using trained model
- Playback: PyFluidSynth for rendering MIDI to sound

## 📊 Evaluation

| Model               | Validation Loss |
|--------------------|-----------------|
| Vanilla RNN        | 2.45            |
| LSTM               | 1.98            |
| BiLSTM             | 1.85            |
| LSTM + Attention   | **0.47**        |

- Human evaluation confirms melodic coherence and harmonic progression.
- Visualized using piano rolls and loss curves.

## 🔮 Future Work

- Transformer-based architectures for better long-range modeling
- Emotion and dynamics-aware generation
- Real-time music generation with user guidance
- Genre adaptation and music style transfer

## 👨‍💻 Authors

- Mohammad Bashar  
- Mahmoud Abdelalim  
- Hazem Nemer  
- Supervisor: Dr. Wisam Elmasry
