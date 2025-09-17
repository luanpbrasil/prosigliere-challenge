#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) for Fall Detection Dataset

Generated visualizations:
1. Waist acceleration over time comparison
2. Head angular velocity over time comparison  
3. Acceleration magnitude distribution histogram
4. Mean acceleration by sensor (fall data)
5. Frequency spectrum analysis (waist sensor)
6. Dataset distribution by category

All plots are saved to the 'visualizations/eda' folder.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from typing import Dict, List, Optional

warnings.filterwarnings('ignore')

# Configure plotting style
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def setup_directories() -> Path:
    """Create and return the EDA visualizations directory."""
    viz_dir = Path('visualizations/eda')
    viz_dir.mkdir(parents=True, exist_ok=True)
    return viz_dir


def load_sample_data() -> Dict[str, pd.DataFrame]:
    """
    Load sample data files from each category for analysis.
    
    Returns:
        Dictionary containing sample dataframes for each category
    """
    sample_files = {
        'ADL': 'data/raw/sub1/ADLs/AXR_AS_trial1.xlsx',
        'Fall': 'data/raw/sub1/Falls/AXR_ITCS_trial1.xlsx',
        'Near_Fall': 'data/raw/sub1/Near_Falls/AXR_ITCS_trial2.xlsx'
    }
    
    sample_data = {}
    
    for category, file_path in sample_files.items():
        file_path = Path(file_path)
        if file_path.exists():
            try:
                df = pd.read_excel(file_path)
                sample_data[category] = df
                print(f"✓ Loaded {category}: {df.shape}")
            except Exception as e:
                print(f"✗ Error loading {file_path}: {e}")
        else:
            print(f"✗ File not found: {file_path}")
    
    return sample_data


def count_dataset_files() -> Dict[str, int]:
    """
    Count files per category across all subjects.
    
    Returns:
        Dictionary with file counts per category
    """
    data_path = Path('data/raw')
    subjects = [f'sub{i}' for i in range(1, 9)]
    categories = ['ADLs', 'Falls', 'Near_Falls']
    
    file_counts = {}
    
    for subject in subjects:
        subject_path = data_path / subject
        if subject_path.exists():
            for category in categories:
                category_path = subject_path / category
                if category_path.exists():
                    files = list(category_path.glob('*.xlsx'))
                    file_count = len(files)
                    
                    if category not in file_counts:
                        file_counts[category] = 0
                    file_counts[category] += file_count
    
    return file_counts


def plot_waist_acceleration(sample_data: Dict[str, pd.DataFrame], viz_dir: Path) -> None:
    """Generate waist acceleration over time comparison plot."""
    plt.figure(figsize=(12, 6))
    
    for category, df in sample_data.items():
        waist_accel_cols = [col for col in df.columns if 'waist Acceleration' in col]
        if waist_accel_cols:
            waist_accel = df[waist_accel_cols].values
            waist_accel_mag = np.sqrt(np.sum(waist_accel**2, axis=1))
            time_seconds = (df['Time'] - df['Time'].iloc[0]) / 1e6
            plt.plot(time_seconds, waist_accel_mag, label=category, alpha=0.8, linewidth=1.5)
    
    plt.title('Waist Acceleration Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s²)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = viz_dir / 'waist_acceleration_over_time.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_head_angular_velocity(sample_data: Dict[str, pd.DataFrame], viz_dir: Path) -> None:
    """Generate head angular velocity over time comparison plot."""
    plt.figure(figsize=(12, 6))
    
    for category, df in sample_data.items():
        head_gyro_cols = [col for col in df.columns if 'head Angular Velocity' in col]
        if head_gyro_cols:
            head_gyro = df[head_gyro_cols].values
            head_gyro_mag = np.sqrt(np.sum(head_gyro**2, axis=1))
            time_seconds = (df['Time'] - df['Time'].iloc[0]) / 1e6
            plt.plot(time_seconds, head_gyro_mag, label=category, alpha=0.8, linewidth=1.5)
    
    plt.title('Head Angular Velocity Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = viz_dir / 'head_angular_velocity_over_time.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_acceleration_distribution(sample_data: Dict[str, pd.DataFrame], viz_dir: Path) -> None:
    """Generate acceleration magnitude distribution histogram."""
    plt.figure(figsize=(12, 6))
    
    for category, df in sample_data.items():
        accel_cols = [col for col in df.columns if 'Acceleration' in col]
        accel_data = df[accel_cols].values
        accel_magnitude = np.sqrt(np.sum(accel_data**2, axis=1))
        plt.hist(accel_magnitude, bins=50, alpha=0.6, label=category, density=True)
    
    plt.title('Acceleration Magnitude Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Acceleration (m/s²)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = viz_dir / 'acceleration_magnitude_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_sensor_means(sample_data: Dict[str, pd.DataFrame], viz_dir: Path) -> None:
    """Generate mean acceleration by sensor bar chart using fall data."""
    plt.figure(figsize=(12, 6))
    sensors = ['r.ankle', 'l.ankle', 'r.thigh', 'l.thigh', 'head', 'sternum', 'waist']
    
    if 'Fall' not in sample_data:
        print("⚠ Fall data not available for sensor means plot")
        return
    
    fall_df = sample_data['Fall']
    sensor_means = []
    
    for sensor in sensors:
        accel_cols = [col for col in fall_df.columns if f'{sensor} Acceleration' in col]
        if accel_cols:
            accel_data = fall_df[accel_cols].values
            accel_magnitude = np.sqrt(np.sum(accel_data**2, axis=1))
            sensor_means.append(np.mean(accel_magnitude))
        else:
            sensor_means.append(0)
    
    bars = plt.bar(sensors, sensor_means, color='skyblue', alpha=0.7)
    plt.title('Mean Acceleration by Sensor (Fall)', fontsize=14, fontweight='bold')
    plt.ylabel('Acceleration (m/s²)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add values on bars
    for bar, value in zip(bars, sensor_means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    output_path = viz_dir / 'mean_acceleration_by_sensor.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_frequency_spectrum(sample_data: Dict[str, pd.DataFrame], viz_dir: Path) -> None:
    """Generate frequency spectrum analysis for waist sensor."""
    plt.figure(figsize=(12, 6))
    
    for category, df in sample_data.items():
        waist_accel_cols = [col for col in df.columns if 'waist Acceleration' in col]
        if waist_accel_cols:
            waist_accel = df[waist_accel_cols].values
            waist_accel_mag = np.sqrt(np.sum(waist_accel**2, axis=1))
            
            # FFT analysis
            fft = np.fft.fft(waist_accel_mag)
            freqs = np.fft.fftfreq(len(waist_accel_mag), 1/128.0)
            
            # Plot only positive frequencies
            positive_freqs = freqs[:len(freqs)//2]
            positive_fft = np.abs(fft[:len(fft)//2])
            
            plt.plot(positive_freqs, positive_fft, label=category, alpha=0.8)
    
    plt.title('Frequency Spectrum (Waist)', fontsize=14, fontweight='bold')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 64)  # Nyquist frequency
    plt.tight_layout()
    
    output_path = viz_dir / 'frequency_spectrum_waist.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_dataset_distribution(file_counts: Dict[str, int], viz_dir: Path) -> None:
    """Generate dataset distribution pie chart."""
    if not file_counts:
        print("⚠ No file counts available for dataset distribution plot")
        return
    
    plt.figure(figsize=(10, 8))
    categories_list = list(file_counts.keys())
    counts = list(file_counts.values())
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    
    wedges, texts, autotexts = plt.pie(counts, labels=categories_list, autopct='%1.1f%%', 
                                       colors=colors, startangle=90)
    plt.title('Dataset Distribution by Category', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = viz_dir / 'dataset_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def print_data_summary(sample_data: Dict[str, pd.DataFrame], file_counts: Dict[str, int]) -> None:
    """Print summary of the dataset."""
    print("\n" + "="*60)
    print("FALL DETECTION DATASET - EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    print(f"\nDataset Overview:")
    print(f"• Total categories: {len(file_counts)}")
    print(f"• Total files: {sum(file_counts.values())}")
    for category, count in file_counts.items():
        print(f"  - {category}: {count} files")
    
    print(f"\nSample Data Loaded:")
    for category, df in sample_data.items():
        sampling_rate = 128.0  # Hz (from notebook analysis)
        duration = len(df) / sampling_rate
        print(f"• {category}: {df.shape[0]} samples, {df.shape[1]} features, {duration:.1f}s")
    
    print(f"\nGenerated Visualizations:")
    print(f"• All plots saved to 'visualizations/eda/' folder")
    print(f"• High resolution PNG format (300 DPI)")


def main():
    """Main function to generate all EDA visualizations."""
    print("Starting Fall Detection EDA...")
    
    # Setup
    viz_dir = setup_directories()
    
    # Load data
    print("\nLoading sample data...")
    sample_data = load_sample_data()
    
    if not sample_data:
        print("✗ No sample data loaded. Please check data files exist.")
        return
    
    print("\nCounting dataset files...")
    file_counts = count_dataset_files()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    plot_waist_acceleration(sample_data, viz_dir)
    plot_head_angular_velocity(sample_data, viz_dir)
    plot_acceleration_distribution(sample_data, viz_dir)
    plot_sensor_means(sample_data, viz_dir)
    plot_frequency_spectrum(sample_data, viz_dir)
    plot_dataset_distribution(file_counts, viz_dir)
    
    # Print summary
    print_data_summary(sample_data, file_counts)
    
    print(f"\n✓ EDA completed successfully!")
    print(f"✓ All visualizations saved to: {viz_dir.absolute()}")


if __name__ == "__main__":
    main()
