import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from tqdm import tqdm
import pickle
from datetime import datetime

# Signal Processing
from scipy import signal
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq

warnings.filterwarnings('ignore')


class FallDetectionDataLoader:
    """Efficient data loader for fall detection dataset"""
    
    def __init__(self, data_path='data/raw'):
        self.data_path = Path(data_path)
        self.subjects = [f'sub{i}' for i in range(1, 9)]
        self.categories = ['ADLs', 'Falls', 'Near_Falls']
        self.sensors = ['r.ankle', 'l.ankle', 'r.thigh', 'l.thigh', 'head', 'sternum', 'waist']
        self.sampling_rate = 128.0
        
    def get_file_list(self):
        """Get list of all files with metadata"""
        files_info = []
        
        for subject in self.subjects:
            subject_path = self.data_path / subject
            if subject_path.exists():
                for category in self.categories:
                    category_path = subject_path / category
                    if category_path.exists():
                        files = list(category_path.glob('*.xlsx'))
                        for file_path in files:
                            files_info.append({
                                'file_path': file_path,
                                'subject': subject,
                                'category': category,
                                'trial': file_path.stem
                            })
        
        return pd.DataFrame(files_info)
    
    def load_single_file(self, file_path):
        """Load a single Excel file"""
        try:
            df = pd.read_excel(file_path)
            return df
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def load_all_data(self, max_files=None, verbose=True):
        """Load all data with progress tracking"""
        files_df = self.get_file_list()
        
        if max_files:
            files_df = files_df.head(max_files)
        
        print(f"Loading {len(files_df)} files...")
        
        all_data = []
        failed_files = []
        
        for idx, row in tqdm(files_df.iterrows(), total=len(files_df), desc="Loading files"):
            df = self.load_single_file(row['file_path'])
            
            if df is not None:
                # Add metadata
                df['subject'] = row['subject']
                df['category'] = row['category']
                df['trial'] = row['trial']
                df['file_idx'] = idx
                
                all_data.append(df)
            else:
                failed_files.append(row['file_path'])
        
        if failed_files:
            print(f"Failed to load {len(failed_files)} files")
        
        print(f"Successfully loaded {len(all_data)} files")
        return all_data, files_df


class DataPreprocessor:
    """Data preprocessing utilities for fall detection"""
    
    def __init__(self, sampling_rate=128.0):
        self.sampling_rate = sampling_rate
        self.sensors = ['r.ankle', 'l.ankle', 'r.thigh', 'l.thigh', 'head', 'sternum', 'waist']
    
    def detect_outliers(self, df, sensor_cols, threshold=3):
        """Detect outliers using Z-score method"""
        outlier_mask = pd.DataFrame(False, index=df.index, columns=sensor_cols)
        
        for col in sensor_cols:
            if col in df.columns:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_mask[col] = z_scores > threshold
        
        return outlier_mask
    
    def remove_outliers(self, df, sensor_cols, threshold=3, method='clip'):
        """Remove or clip outliers"""
        outlier_mask = self.detect_outliers(df, sensor_cols, threshold)
        
        if method == 'clip':
            # Clip outliers to threshold
            for col in sensor_cols:
                if col in df.columns:
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    df[col] = df[col].clip(
                        lower=mean_val - threshold * std_val,
                        upper=mean_val + threshold * std_val
                    )
        elif method == 'remove':
            # Remove rows with outliers
            df = df[~outlier_mask.any(axis=1)]
        
        return df
    
    def apply_filter(self, df, sensor_cols, filter_type='lowpass', cutoff=10):
        """Apply signal filtering"""
        filtered_df = df.copy()
        
        for col in sensor_cols:
            if col in df.columns:
                if filter_type == 'lowpass':
                    # Low-pass filter to remove high-frequency noise
                    b, a = signal.butter(4, cutoff / (self.sampling_rate / 2), btype='low')
                    filtered_df[col] = signal.filtfilt(b, a, df[col])
                elif filter_type == 'bandpass':
                    # Band-pass filter
                    b, a = signal.butter(4, [1, cutoff] / (self.sampling_rate / 2), btype='band')
                    filtered_df[col] = signal.filtfilt(b, a, df[col])
        
        return filtered_df


class FeatureEngineer:
    """Feature engineering for fall detection"""
    
    def __init__(self, sampling_rate=128.0):
        self.sampling_rate = sampling_rate
        self.sensors = ['r.ankle', 'l.ankle', 'r.thigh', 'l.thigh', 'head', 'sternum', 'waist']
        
    def extract_magnitude_features(self, df):
        """Extract magnitude features for each sensor"""
        features = {}
        
        for sensor in self.sensors:
            # Acceleration magnitude
            accel_cols = [col for col in df.columns if f'{sensor} Acceleration' in col]
            if accel_cols:
                accel_data = df[accel_cols].values
                accel_magnitude = np.sqrt(np.sum(accel_data**2, axis=1))
                
                features[f'{sensor}_accel_mag_mean'] = np.mean(accel_magnitude)
                features[f'{sensor}_accel_mag_std'] = np.std(accel_magnitude)
                features[f'{sensor}_accel_mag_max'] = np.max(accel_magnitude)
                features[f'{sensor}_accel_mag_min'] = np.min(accel_magnitude)
                features[f'{sensor}_accel_mag_energy'] = np.sum(accel_magnitude**2)
                features[f'{sensor}_accel_mag_rms'] = np.sqrt(np.mean(accel_magnitude**2))
                
                # Zero crossing rate
                zero_crossings = np.sum(np.diff(np.sign(accel_magnitude - np.mean(accel_magnitude))) != 0)
                features[f'{sensor}_accel_mag_zcr'] = zero_crossings / len(accel_magnitude)
            
            # Gyroscope magnitude
            gyro_cols = [col for col in df.columns if f'{sensor} Angular Velocity' in col]
            if gyro_cols:
                gyro_data = df[gyro_cols].values
                gyro_magnitude = np.sqrt(np.sum(gyro_data**2, axis=1))
                
                features[f'{sensor}_gyro_mag_mean'] = np.mean(gyro_magnitude)
                features[f'{sensor}_gyro_mag_std'] = np.std(gyro_magnitude)
                features[f'{sensor}_gyro_mag_max'] = np.max(gyro_magnitude)
                features[f'{sensor}_gyro_mag_energy'] = np.sum(gyro_magnitude**2)
        
        return features
    
    def extract_temporal_features(self, df, window_size=128):
        """Extract temporal features using sliding windows"""
        features = {}
        
        for sensor in self.sensors:
            accel_cols = [col for col in df.columns if f'{sensor} Acceleration' in col]
            if accel_cols:
                accel_data = df[accel_cols].values
                accel_magnitude = np.sqrt(np.sum(accel_data**2, axis=1))
                
                # Sliding window features
                for i in range(0, len(accel_magnitude) - window_size, window_size // 2):
                    window = accel_magnitude[i:i + window_size]
                    
                    # Statistical features
                    features[f'{sensor}_window_mean'] = np.mean(window)
                    features[f'{sensor}_window_std'] = np.std(window)
                    features[f'{sensor}_window_skew'] = skew(window)
                    features[f'{sensor}_window_kurt'] = kurtosis(window)
                    
                    # Energy features
                    features[f'{sensor}_window_energy'] = np.sum(window**2)
                    features[f'{sensor}_window_power'] = np.mean(window**2)
                    
                    # Only take first window for now
                    break
        
        return features
    
    def extract_frequency_features(self, df, n_fft=256):
        """Extract frequency domain features"""
        features = {}
        
        for sensor in self.sensors:
            accel_cols = [col for col in df.columns if f'{sensor} Acceleration' in col]
            if accel_cols:
                accel_data = df[accel_cols].values
                accel_magnitude = np.sqrt(np.sum(accel_data**2, axis=1))
                
                # FFT
                fft_data = fft(accel_magnitude, n=n_fft)
                freqs = fftfreq(n_fft, 1/self.sampling_rate)
                
                # Only positive frequencies
                positive_freqs = freqs[:n_fft//2]
                positive_fft = np.abs(fft_data[:n_fft//2])
                
                # Frequency features
                features[f'{sensor}_dominant_freq'] = positive_freqs[np.argmax(positive_fft)]
                features[f'{sensor}_spectral_centroid'] = np.sum(positive_freqs * positive_fft) / np.sum(positive_fft)
                features[f'{sensor}_spectral_energy'] = np.sum(positive_fft**2)
                features[f'{sensor}_spectral_bandwidth'] = np.sqrt(np.sum(((positive_freqs - features[f'{sensor}_spectral_centroid'])**2) * positive_fft) / np.sum(positive_fft))
                
                # Frequency band energies
                low_freq_mask = (positive_freqs >= 0) & (positive_freqs <= 5)
                mid_freq_mask = (positive_freqs > 5) & (positive_freqs <= 15)
                high_freq_mask = (positive_freqs > 15) & (positive_freqs <= 64)
                
                features[f'{sensor}_low_freq_energy'] = np.sum(positive_fft[low_freq_mask]**2)
                features[f'{sensor}_mid_freq_energy'] = np.sum(positive_fft[mid_freq_mask]**2)
                features[f'{sensor}_high_freq_energy'] = np.sum(positive_fft[high_freq_mask]**2)
        
        return features
    
    def extract_cross_sensor_features(self, df):
        """Extract features that involve multiple sensors"""
        features = {}
        
        # Get all acceleration magnitudes
        accel_magnitudes = {}
        for sensor in self.sensors:
            accel_cols = [col for col in df.columns if f'{sensor} Acceleration' in col]
            if accel_cols:
                accel_data = df[accel_cols].values
                accel_magnitudes[sensor] = np.sqrt(np.sum(accel_data**2, axis=1))
        
        # Cross-sensor correlations
        if len(accel_magnitudes) > 1:
            sensor_pairs = [(s1, s2) for s1 in accel_magnitudes.keys() for s2 in accel_magnitudes.keys() if s1 < s2]
            for s1, s2 in sensor_pairs[:5]:  # Limit to first 5 pairs
                corr = np.corrcoef(accel_magnitudes[s1], accel_magnitudes[s2])[0, 1]
                features[f'{s1}_{s2}_correlation'] = corr
        
        # Global features
        all_accel = np.concatenate(list(accel_magnitudes.values()))
        features['global_accel_mean'] = np.mean(all_accel)
        features['global_accel_std'] = np.std(all_accel)
        features['global_accel_max'] = np.max(all_accel)
        features['global_accel_energy'] = np.sum(all_accel**2)
        
        return features
    
    def extract_all_features(self, df):
        """Extract all features from a single dataframe"""
        features = {}
        
        # Magnitude features
        features.update(self.extract_magnitude_features(df))
        
        # Temporal features
        features.update(self.extract_temporal_features(df))
        
        # Frequency features
        features.update(self.extract_frequency_features(df))
        
        # Cross-sensor features
        features.update(self.extract_cross_sensor_features(df))
        
        return features


class DataSplitter:
    """Data splitting utilities for fall detection"""
    
    def __init__(self, test_size=0.2, val_size=0.2, random_state=42):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
    def subject_wise_split(self, features_df, metadata_df):
        """Split data by subject to avoid data leakage"""
        # Get unique subjects
        subjects = features_df['subject'].unique()
        n_subjects = len(subjects)
        
        print(f"Total subjects: {n_subjects}")
        print(f"Subjects: {subjects}")
        
        # Calculate split sizes
        n_test_subjects = max(1, int(n_subjects * self.test_size))
        n_val_subjects = max(1, int(n_subjects * self.val_size))
        n_train_subjects = n_subjects - n_test_subjects - n_val_subjects
        
        # Ensure we have at least 1 subject for each split
        if n_train_subjects <= 0:
            n_train_subjects = 1
            n_val_subjects = max(1, n_val_subjects - 1)
        if n_val_subjects <= 0:
            n_val_subjects = 1
            n_test_subjects = max(1, n_test_subjects - 1)
        
        # Shuffle subjects
        np.random.seed(self.random_state)
        shuffled_subjects = np.random.permutation(subjects)
        
        # Split subjects
        train_subjects = shuffled_subjects[:n_train_subjects]
        val_subjects = shuffled_subjects[n_train_subjects:n_train_subjects + n_val_subjects]
        test_subjects = shuffled_subjects[n_train_subjects + n_val_subjects:]
        
        print(f"Train subjects: {train_subjects}")
        print(f"Val subjects: {val_subjects}")
        print(f"Test subjects: {test_subjects}")
        
        # Split data
        train_mask = features_df['subject'].isin(train_subjects)
        val_mask = features_df['subject'].isin(val_subjects)
        test_mask = features_df['subject'].isin(test_subjects)
        
        X_train = features_df[train_mask].drop(['subject', 'category', 'trial', 'file_idx'], axis=1)
        y_train = features_df[train_mask]['category']
        
        X_val = features_df[val_mask].drop(['subject', 'category', 'trial', 'file_idx'], axis=1)
        y_val = features_df[val_mask]['category']
        
        X_test = features_df[test_mask].drop(['subject', 'category', 'trial', 'file_idx'], axis=1)
        y_test = features_df[test_mask]['category']
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'train_subjects': train_subjects, 'val_subjects': val_subjects, 'test_subjects': test_subjects
        }


def process_all_data(data_list, preprocessor, feature_engineer, verbose=True):
    """Process all data and extract features"""
    all_features = []
    metadata = []
    
    print(f"Processing {len(data_list)} files...")
    
    for idx, df in enumerate(tqdm(data_list, desc="Processing files")):
        try:
            # Copy data
            df_copy = df.copy()
            
            # Apply filtering
            sensor_cols = [col for col in df_copy.columns if any(sensor in col for sensor in preprocessor.sensors)]
            df_filtered = preprocessor.apply_filter(df_copy, sensor_cols, filter_type='lowpass', cutoff=20)
            
            # Extract features
            features = feature_engineer.extract_all_features(df_filtered)
            
            # Add metadata
            features['subject'] = df['subject'].iloc[0]
            features['category'] = df['category'].iloc[0]
            features['trial'] = df['trial'].iloc[0]
            features['file_idx'] = df['file_idx'].iloc[0]
            
            all_features.append(features)
            
            # Store metadata
            metadata.append({
                'file_idx': df['file_idx'].iloc[0],
                'subject': df['subject'].iloc[0],
                'category': df['category'].iloc[0],
                'trial': df['trial'].iloc[0]
            })
            
        except Exception as e:
            if verbose:
                print(f"Error processing file {idx}: {e}")
            continue
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    metadata_df = pd.DataFrame(metadata)
    
    print(f"Successfully processed {len(features_df)} files")
    print(f"Feature matrix shape: {features_df.shape}")
    
    return features_df, metadata_df


def save_processed_data(split_data, file_prefix='fall_detection'):
    """Save processed data to pickle files"""
    
    # Create data/processed folder if it doesn't exist
    data_folder = Path('data')
    processed_folder = data_folder / 'processed'
    processed_folder.mkdir(parents=True, exist_ok=True)
    
    # Save individual split datasets
    print(f"💾 SAVING PROCESSED DATA:")
    
    # Extract split data
    X_train = split_data['X_train']
    X_val = split_data['X_val'] 
    X_test = split_data['X_test']
    y_train = split_data['y_train']
    y_val = split_data['y_val']
    y_test = split_data['y_test']
    
    # Save X_train, X_val, X_test
    x_train_file = processed_folder / f'{file_prefix}_X_train.pkl'
    with open(x_train_file, 'wb') as f:
        pickle.dump(X_train, f)
    print(f"Saved X_train to: {x_train_file}")
    
    x_val_file = processed_folder / f'{file_prefix}_X_val.pkl'
    with open(x_val_file, 'wb') as f:
        pickle.dump(X_val, f)
    print(f"Saved X_val to: {x_val_file}")
    
    x_test_file = processed_folder / f'{file_prefix}_X_test.pkl'
    with open(x_test_file, 'wb') as f:
        pickle.dump(X_test, f)
    print(f"Saved X_test to: {x_test_file}")
    
    # Save y_train, y_val, y_test
    y_train_file = processed_folder / f'{file_prefix}_y_train.pkl'
    with open(y_train_file, 'wb') as f:
        pickle.dump(y_train, f)
    print(f"Saved y_train to: {y_train_file}")
    
    y_val_file = processed_folder / f'{file_prefix}_y_val.pkl'
    with open(y_val_file, 'wb') as f:
        pickle.dump(y_val, f)
    print(f"Saved y_val to: {y_val_file}")
    
    y_test_file = processed_folder / f'{file_prefix}_y_test.pkl'
    with open(y_test_file, 'wb') as f:
        pickle.dump(y_test, f)
    print(f"Saved y_test to: {y_test_file}")
    
    return {
        'x_train_file': str(x_train_file),
        'x_val_file': str(x_val_file),
        'x_test_file': str(x_test_file),
        'y_train_file': str(y_train_file),
        'y_val_file': str(y_val_file),
        'y_test_file': str(y_test_file)
    }


def main():
    """Main preprocessing pipeline"""
    print("=" * 80)
    print("FALL DETECTION DATA PREPROCESSING")
    print("=" * 80)
    
    # Initialize components
    loader = FallDetectionDataLoader()
    preprocessor = DataPreprocessor()
    feature_engineer = FeatureEngineer()
    splitter = DataSplitter()
    
    # Load data
    print("\n1. Loading data...")
    all_data, files_df = loader.load_all_data()
    
    print(f"Files per category:")
    print(files_df['category'].value_counts())
    print(f"\nFiles per subject:")
    print(files_df['subject'].value_counts().sort_index())
    
    # Process data
    print("\n2. Processing data and extracting features...")
    features_df, metadata_df = process_all_data(all_data, preprocessor, feature_engineer)
    
    print(f"Feature matrix shape: {features_df.shape}")
    print(f"Categories: {features_df['category'].value_counts().to_dict()}")
    print(f"Subjects: {features_df['subject'].value_counts().to_dict()}")
    
    # Split data
    print("\n3. Splitting data...")
    split_data = splitter.subject_wise_split(features_df, metadata_df)
    
    print(f"Train: {len(split_data['X_train'])} samples")
    print(f"Val: {len(split_data['X_val'])} samples")
    print(f"Test: {len(split_data['X_test'])} samples")
    
    # Save data
    print("\n4. Saving processed data...")
    saved_files = save_processed_data(split_data)
    
    print(f"\n✅ PREPROCESSING COMPLETED!")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    # Run the preprocessing pipeline
    main()
