#!/usr/bin/env python3
"""
Data Preparation Script
Preprocesses EEG data and creates ready-to-use (brain, image) pairs
Saves in multiple formats for team members to use
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
from PIL import Image
import sys

def load_events_and_create_metadata(subject_id, data_dir="data/ds002718"):
    """
    Load events and create metadata for each training example
    This works even if MNE fails
    """
    events_file = Path(data_dir) / subject_id / 'eeg' / f'{subject_id}_task-FaceRecognition_events.tsv'
    events = pd.read_csv(events_file, sep='\t')
    
    # Filter to face presentations only
    face_events = events[events['event_type'] == 'faces'].copy()
    
    return face_events

def load_eeg_with_mne(eeg_path):
    """
    Load EEG using MNE-Python (now that environment is fixed)
    """
    try:
        import mne
        
        print(f"  Loading EEG with MNE...")
        raw = mne.io.read_raw_eeglab(str(eeg_path), preload=True, verbose=False)
        
        data = raw.get_data()  # Get data as numpy array (channels x timepoints)
        sfreq = raw.info['sfreq']
        ch_names = raw.ch_names
        n_channels = data.shape[0]
        n_samples = data.shape[1]
        
        print(f"  ✓ Loaded: {n_channels} channels, {n_samples} samples, {sfreq} Hz")
        
        return {
            'data': data,
            'sfreq': sfreq,
            'ch_names': ch_names,
            'n_channels': n_channels,
            'n_samples': n_samples,
            'raw': raw
        }
    
    except Exception as e:
        print(f"  ✗ Error loading with MNE: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_epoch(eeg_data, onset_sec, sfreq, tmin=0.0, tmax=1.0):
    """
    Extract a time window from EEG data
    
    Args:
        eeg_data: channels x timepoints array
        onset_sec: stimulus onset time in seconds
        sfreq: sampling frequency
        tmin: start time relative to onset (seconds)
        tmax: end time relative to onset (seconds)
    
    Returns:
        epoch: channels x timepoints_in_window
    """
    start_sample = int((onset_sec + tmin) * sfreq)
    end_sample = int((onset_sec + tmax) * sfreq)
    
    if start_sample < 0 or end_sample > eeg_data.shape[1]:
        return None
    
    epoch = eeg_data[:, start_sample:end_sample]
    return epoch

def simple_bandpass_filter(data, sfreq, lowcut=0.1, highcut=40):
    """
    Apply simple bandpass filter using scipy
    """
    try:
        from scipy.signal import butter, filtfilt
        
        nyq = 0.5 * sfreq
        low = lowcut / nyq
        high = highcut / nyq
        
        b, a = butter(4, [low, high], btype='band')
        filtered = filtfilt(b, a, data, axis=1)
        
        return filtered
    except Exception as e:
        print(f"Warning: Filtering failed: {e}")
        return data

def preprocess_epoch(epoch, sfreq):
    """
    Apply basic preprocessing to an epoch
    """
    # Apply bandpass filter
    epoch_filtered = simple_bandpass_filter(epoch, sfreq, lowcut=0.1, highcut=40)
    
    # Baseline correction (subtract mean of first 50ms if tmin < 0)
    # For now, just z-score normalization per channel
    epoch_normalized = (epoch_filtered - epoch_filtered.mean(axis=1, keepdims=True)) / (epoch_filtered.std(axis=1, keepdims=True) + 1e-10)
    
    return epoch_normalized

def load_and_resize_image(image_path, size=(224, 224)):
    """Load image and resize to standard size"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(size)
    img_array = np.array(img, dtype=np.uint8)
    return img_array

def create_preprocessed_dataset(subject_ids, data_dir="data/ds002718", output_dir="preprocessed_data"):
    """
    Main function: Create preprocessed dataset
    
    Outputs:
    - brain_image_pairs.npz: NumPy arrays of (EEG, images, labels)
    - dataset_info.json: Metadata about the dataset
    - sample_examples.pkl: A few examples for quick inspection
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("="*70)
    print("PREPROCESSING DATASET FOR TEAM")
    print("="*70)
    
    all_eeg_epochs = []
    all_images = []
    all_labels = []
    all_metadata = []
    
    stimuli_dir = Path(data_dir) / "stimuli"
    
    for subject_id in subject_ids:
        print(f"\n{'='*70}")
        print(f"Processing {subject_id}")
        print(f"{'='*70}")
        
        # Load EEG
        eeg_path = Path(data_dir) / subject_id / "eeg" / f"{subject_id}_task-FaceRecognition_eeg.set"
        
        if not eeg_path.exists():
            print(f"  ✗ EEG file not found, skipping...")
            continue
        
        eeg_dict = load_eeg_with_mne(eeg_path)
        
        if eeg_dict is None:
            print(f"  ✗ Could not load EEG, skipping...")
            continue
        
        eeg_data = eeg_dict['data']
        sfreq = eeg_dict['sfreq']
        
        # Load events
        print(f"  Loading events...")
        events = load_events_and_create_metadata(subject_id, data_dir)
        print(f"  ✓ Found {len(events)} face presentations")
        
        # Extract epochs
        print(f"  Extracting and preprocessing epochs...")
        
        for idx, (_, event) in enumerate(events.iterrows()):
            if idx % 100 == 0:
                print(f"    Progress: {idx}/{len(events)}")
            
            onset = event['onset']
            stim_file = event['stim_file']
            face_type = event['face_type']
            
            # Extract epoch (0 to 1 second after stimulus)
            epoch = extract_epoch(eeg_data, onset, sfreq, tmin=0.0, tmax=1.0)
            
            if epoch is None:
                continue
            
            # Preprocess
            epoch_preprocessed = preprocess_epoch(epoch, sfreq)
            
            # Load image
            image_path = stimuli_dir / stim_file
            if not image_path.exists():
                continue
            
            try:
                image = load_and_resize_image(image_path)
            except Exception as e:
                print(f"    Warning: Could not load {stim_file}: {e}")
                continue
            
            # Store
            all_eeg_epochs.append(epoch_preprocessed)
            all_images.append(image)
            all_labels.append(face_type)
            all_metadata.append({
                'subject': subject_id,
                'onset': onset,
                'stim_file': stim_file,
                'face_type': face_type,
                'trial_type': event['trial_type'],
                'epoch_shape': epoch_preprocessed.shape,
                'image_shape': image.shape
            })
        
        print(f"  ✓ Extracted {len([m for m in all_metadata if m['subject'] == subject_id])} valid epochs")
    
    if len(all_eeg_epochs) == 0:
        print("\n✗ No data extracted! Check your data directory.")
        return
    
    # Convert to numpy arrays
    print(f"\n{'='*70}")
    print("Converting to numpy arrays...")
    print(f"{'='*70}")
    
    eeg_array = np.array(all_eeg_epochs, dtype=np.float32)
    images_array = np.array(all_images, dtype=np.uint8)
    labels_array = np.array(all_labels)
    
    print(f"  EEG shape: {eeg_array.shape} (examples, channels, timepoints)")
    print(f"  Images shape: {images_array.shape} (examples, height, width, RGB)")
    print(f"  Labels shape: {labels_array.shape}")
    
    # Save dataset
    print(f"\n{'='*70}")
    print("Saving preprocessed dataset...")
    print(f"{'='*70}")
    
    # Save as NumPy archive (efficient, easy to load)
    npz_path = output_path / "brain_image_dataset.npz"
    np.savez_compressed(
        npz_path,
        eeg=eeg_array,
        images=images_array,
        labels=labels_array
    )
    print(f"  ✓ Saved: {npz_path} ({npz_path.stat().st_size / 1e6:.1f} MB)")
    
    # Save metadata as JSON
    metadata_path = output_path / "dataset_info.json"
    
    label_counts = pd.Series(labels_array).value_counts().to_dict()
    
    dataset_info = {
        'n_examples': len(all_eeg_epochs),
        'n_subjects': len(set(m['subject'] for m in all_metadata)),
        'subjects': list(set(m['subject'] for m in all_metadata)),
        'eeg_shape': list(eeg_array.shape),
        'images_shape': list(images_array.shape),
        'labels': list(set(labels_array)),
        'label_counts': label_counts,
        'sampling_freq': float(sfreq),
        'epoch_duration': 1.0,
        'preprocessing': [
            'Bandpass filter: 0.1-40 Hz',
            'Z-score normalization per channel',
            'Images resized to 224x224'
        ]
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    print(f"  ✓ Saved: {metadata_path}")
    
    # Save detailed metadata as pickle
    metadata_pkl = output_path / "detailed_metadata.pkl"
    with open(metadata_pkl, 'wb') as f:
        pickle.dump(all_metadata, f)
    print(f"  ✓ Saved: {metadata_pkl}")
    
    # Save a few examples for inspection
    sample_indices = np.random.choice(len(all_eeg_epochs), min(10, len(all_eeg_epochs)), replace=False)
    samples = {
        'eeg': [all_eeg_epochs[i] for i in sample_indices],
        'images': [all_images[i] for i in sample_indices],
        'labels': [all_labels[i] for i in sample_indices],
        'metadata': [all_metadata[i] for i in sample_indices]
    }
    
    samples_path = output_path / "sample_examples.pkl"
    with open(samples_path, 'wb') as f:
        pickle.dump(samples, f)
    print(f"  ✓ Saved: {samples_path}")
    
    # Create loading instructions
    instructions = """
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
"""
    
    readme_path = output_path / "README_LOADING.md"
    with open(readme_path, 'w') as f:
        f.write(instructions)
    print(f"  ✓ Saved: {readme_path}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("PREPROCESSING COMPLETE!")
    print(f"{'='*70}")
    print(f"\nDataset Summary:")
    print(f"  Total examples: {len(all_eeg_epochs)}")
    print(f"  Subjects: {len(set(m['subject'] for m in all_metadata))}")
    print(f"  EEG shape per example: {eeg_array[0].shape} (channels × timepoints)")
    print(f"  Image shape per example: {images_array[0].shape} (H × W × RGB)")
    print(f"\nLabel distribution:")
    for label, count in label_counts.items():
        pct = 100 * count / len(all_eeg_epochs)
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    print(f"\nOutput directory: {output_path.absolute()}")
    print(f"\nFiles created:")
    print(f"  • brain_image_dataset.npz - Main dataset (load with np.load)")
    print(f"  • dataset_info.json - Metadata and statistics")
    print(f"  • detailed_metadata.pkl - Per-example metadata")
    print(f"  • sample_examples.pkl - 10 examples for inspection")
    print(f"  • README_LOADING.md - Instructions for your team")
    
    print(f"\n{'='*70}")
    print("Your team can now use the preprocessed data!")
    print("Just share the 'preprocessed_data' folder")
    print(f"{'='*70}")

def main():
    """Run preprocessing"""
    
    print("\n" + "#"*70)
    print("#  DATA PREPARATION FOR TEAM")
    print("#"*70)
    
    # Check if data exists
    if not Path('data/ds002718').exists():
        print("\n✗ Error: Dataset not found!")
        print("Run: python download_boto3.py")
        return
    
    # Check MNE
    try:
        import mne
        print("✓ MNE installed")
    except ImportError:
        print("✗ MNE not installed")
        print("Install with: conda install -c conda-forge mne")
        return
    
    # Which subjects to process
    subjects = ['sub-002']  # Add more as you download them
    
    print(f"\nWill preprocess subjects: {subjects}")
    print("This will create a clean dataset file for your team")
    
    response = input("\nProceed? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Create dataset
    create_preprocessed_dataset(subjects, output_dir="preprocessed_data")

if __name__ == "__main__":
    main()

