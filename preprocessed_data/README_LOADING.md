
# How to Load the Preprocessed Dataset

## Quick Load (Python)

```python
import numpy as np

# Load data
data = np.load('preprocessed_data/brain_image_dataset.npz')
eeg = data['eeg']        # Shape: (n_examples, n_channels, n_timepoints)
images = data['images']  # Shape: (n_examples, 224, 224, 3)
labels = data['labels']  # Shape: (n_examples,)

print(f"Loaded {len(eeg)} examples")
print(f"Labels: {set(labels)}")

# Split train/test
from sklearn.model_selection import train_test_split
X_eeg_train, X_eeg_test, y_train, y_test = train_test_split(
    eeg, labels, test_size=0.2, random_state=42
)
```

## Load Metadata

```python
import json

with open('preprocessed_data/dataset_info.json') as f:
    info = json.load(f)

print(f"Total examples: {info['n_examples']}")
print(f"Label distribution: {info['label_counts']}")
```

## Dataset Format

- **EEG data**: Float32, shape (n_examples, n_channels, n_timepoints)
  - Already preprocessed: bandpass filtered (0.1-40 Hz), normalized
  - Channels: 70 EEG channels
  - Timepoints: 250 (1 second at 250 Hz)
  
- **Images**: Uint8, shape (n_examples, 224, 224, 3)
  - RGB format
  - Resized to 224×224 for standard CNNs
  
- **Labels**: String array
  - Values: 'famous', 'unfamiliar', 'scrambled'
  - Balanced distribution

## Ready for ML!

This dataset is ready to use with:
- Scikit-learn
- PyTorch
- TensorFlow/Keras
- Any other ML framework

No need to deal with EEG loading or preprocessing!
