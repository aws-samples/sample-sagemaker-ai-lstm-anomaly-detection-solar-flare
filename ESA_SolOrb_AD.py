/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

# Install required package
!pip install astropy

# Import libraries
from astropy.io import fits
import pandas as pd
%matplotlib inline

import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import boto3

def explore_ql_lightcurve(fits_file_path, csv_output_path="lightcurve_data.csv"):
    """
    Explore and optionally convert STIX quicklook lightcurve FITS file to CSV
    """
    try:
        with fits.open(fits_file_path) as hdul:
            print("\nHDU Information:")
            hdul.info()
            
            # Get the DATA HDU (HDU 2)
            data_hdu = hdul[2]
            
            print("\nDATA Column Names:")
            print(data_hdu.columns.names)
            
            # Create DataFrame from DATA HDU
            data_dict = {}
            
            # Handle 1D columns first
            for col_name in ['control_index', 'time', 'timedel', 'triggers', 'triggers_comp_err', 'rcr']:
                data_dict[col_name] = data_hdu.data[col_name]
            
            # Handle multi-dimensional counts data
            counts_data = data_hdu.data['counts']
            counts_err_data = data_hdu.data['counts_comp_err']
            
            # Get number of energy channels
            n_channels = counts_data.shape[1]
            
            # Add counts for each energy channel
            for i in range(n_channels):
                data_dict[f'counts_ch{i}'] = counts_data[:, i]
                data_dict[f'counts_err_ch{i}'] = counts_err_data[:, i]
            
            df = pd.DataFrame(data_dict)
            
            print("\nDataFrame Info:")
            print(df.info())
            
            print("\nSample Data (first 5 rows):")
            print(df.head())
            
            # Get energy ranges
            energy_hdu = hdul[4]
            energy_ranges = []
            for row in energy_hdu.data:
                energy_ranges.append((row[1], row[2]))  # (low, high)
            
            if csv_output_path:
                df.to_csv(csv_output_path, index=False)
                print(f"\nSaved to: {csv_output_path}")
            
            return df, energy_ranges
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

def plot_lightcurve(df, energy_ranges):
    """
    Plot the lightcurve data
    """
    plt.figure(figsize=(15, 10))
    
    # Time series for each energy band
    plt.subplot(2, 1, 1)
    
    for i, (e_low, e_high) in enumerate(energy_ranges):
        plt.plot(df['time'], df[f'counts_ch{i}'], 
                label=f'{e_low:.1f}-{e_high:.1f} keV')
    
    plt.title('STIX Quicklook Lightcurve - May 20th, 2024')
    plt.xlabel('Time (s)')
    plt.ylabel('Counts')
    plt.legend()
    plt.grid(True)
    
    # Energy-time evolution
    plt.subplot(2, 1, 2)
    counts_data = np.array([df[f'counts_ch{i}'].values for i in range(len(energy_ranges))])
    channel_energies = np.array([er[0] for er in energy_ranges])
    
    plt.pcolormesh(df['time'], channel_energies, counts_data,
                   shading='auto', cmap='magma')
    plt.colorbar(label='Counts')
    plt.title('Energy-Time Evolution')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (keV)')
    
    plt.tight_layout()
    plt.show()

def print_channel_stats(df, energy_ranges):
    """
    Print statistics for each energy channel
    """
    print("\nEnergy Channel Statistics:")
    for i, (e_low, e_high) in enumerate(energy_ranges):
        counts = df[f'counts_ch{i}']
        print(f"\nChannel {i} ({e_low:.1f}-{e_high:.1f} keV):")
        print(f"Mean counts: {counts.mean():.2f}")
        print(f"Max counts: {counts.max():.2f}")
        print(f"Standard deviation: {counts.std():.2f}")

# Usage
ql_file = 'solo_L1_stix-ql-lightcurve_20240520_V02.fits'
df, energy_ranges = explore_ql_lightcurve(ql_file)
plot_lightcurve(df, energy_ranges)
print_channel_stats(df, energy_ranges)

# Print energy ranges
print("\nEnergy Ranges:")
for i, (e_low, e_high) in enumerate(energy_ranges):
    print(f"Channel {i}: {e_low:.1f}-{e_high:.1f} keV")
    
class CrossChannelLSTM(nn.Module):
    def __init__(self, num_channels=5, hidden_size=128, num_layers=2):
        super(CrossChannelLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=num_channels,  # All channels as input
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, num_channels)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

class CrossChannelDataset(Dataset):
    def __init__(self, data, sequence_length):
        """
        data: numpy array of shape (time_steps, num_channels)
        """
        self.data = torch.FloatTensor(data)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.data) - self.sequence_length
        
    def __getitem__(self, idx):
        sequence = self.data[idx:idx + self.sequence_length]
        target = self.data[idx + self.sequence_length]
        return sequence, target

# def detect_cross_channel_anomalies(df, energy_ranges,
                                 # sequence_length=50,
                                 # hidden_size=128,
                                 # num_layers=2,
                                 # batch_size=32,
                                 # num_epochs=20):

def detect_cross_channel_anomalies(df, energy_ranges, 
                                 sequence_length=30,        
                                 hidden_size=256,     # Increased complexity
                                 num_layers=3,        # Deeper network
                                 batch_size=32,
                                 num_epochs=20,
                                 threshold_multiplier=1.5,
                                 dropout=0.2):        # Add dropout for better generalization
    """
    Cross-channel LSTM anomaly detection
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare all channels together
    channel_data = np.column_stack([df[f'counts_ch{i}'].values 
                                  for i in range(len(energy_ranges))])
    
    # Normalize data
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(channel_data)
    
    # Create dataset and dataloader
    dataset = CrossChannelDataset(normalized_data, sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = CrossChannelLSTM(
        num_channels=len(energy_ranges),
        hidden_size=hidden_size,
        num_layers=num_layers
    ).to(device)
    
    # Training
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for sequences, targets in dataloader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}')
    
    # Detect anomalies
    model.eval()
    predictions = []
    with torch.no_grad():
        for sequences, _ in dataloader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            predictions.extend(outputs.cpu().numpy())
    
    predictions = np.array(predictions)
    
    # Calculate cross-channel MSE
    mse = np.mean(np.power(normalized_data[sequence_length:] - predictions, 2), axis=1)
    # threshold = np.mean(mse)
    threshold = 0.0004
    print("Threshold:")
    print(threshold)
    # + 0.25 * np.std(mse)
    anomaly_indices = np.where(mse > threshold)[0] + sequence_length
    
    return normalized_data, predictions, anomaly_indices

# Usage

start_time = time.time()

normalized_data, predictions, anomaly_indices = detect_cross_channel_anomalies(
    df, 
    energy_ranges,
    sequence_length=50,
    hidden_size=128,
    num_layers=2
)

end_time = time.time()

print("Time:")
print(end_time - start_time)

def plot_solar_flare_anomalies(df, energy_ranges, normalized_data, predictions, anomaly_indices, bucket_name):
    """
    Plot each channel in subplots with anomaly detection and save to S3
    """
    # Create the first figure for channel plots
    fig1 = plt.figure(figsize=(15, 12))
    
    # Plot each channel in its own subplot
    for i, (e_low, e_high) in enumerate(energy_ranges):
        plt.subplot(len(energy_ranges), 1, i+1)
        
        # Plot original data
        plt.plot(df['time'], df[f'counts_ch{i}'], 
                color='C0',
                label='Observed', 
                alpha=0.6)
        
        # Mark anomalies
        anomaly_times = df['time'].iloc[anomaly_indices]
        anomaly_counts = df[f'counts_ch{i}'].iloc[anomaly_indices]
        plt.scatter(anomaly_times, anomaly_counts,
                   color='red',
                   marker='x',
                   s=100,
                   label='Anomalies')
        
        plt.title(f'Channel {i}: {e_low:.1f}-{e_high:.1f} keV')
        plt.xlabel('Time (cs)')
        plt.ylabel('Counts')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.suptitle('Multi-Channel Analysis with Anomaly Detection', fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save first figure to temporary file
    temp_file1 = '/tmp/channels_plot.png'
    plt.savefig(temp_file1, bbox_inches='tight', dpi=300)
    plt.close(fig1)
    
    # Create second figure for prediction error
    fig2 = plt.figure(figsize=(15, 4))
    
    # Calculate MSE
    actual = normalized_data[50:50+len(predictions)]
    mse = np.mean(np.power(actual - predictions, 2), axis=1)
    
    # Plot prediction error
    plt.plot(df['time'][50:50+len(predictions)], mse, 
            color='blue', alpha=0.6, label='Prediction Error')
    
    # Add threshold
    threshold = np.mean(mse) + 2*np.std(mse)
    plt.axhline(y=threshold, 
                color='red', 
                linestyle='--', 
                label=f'Anomaly Threshold ({threshold:.4f})')
    
    # Mark anomalies on error plot
    anomaly_times = df['time'].iloc[anomaly_indices]
    anomaly_mse = mse[anomaly_indices - 50]
    plt.scatter(anomaly_times, anomaly_mse,
               color='red',
               marker='x',
               s=100,
               label='Detected Anomalies')
    
    plt.title(f'LSTM Prediction Error (Anomalies: {len(anomaly_indices)})')
    plt.xlabel('Time (cs)')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save second figure to temporary file
    temp_file2 = '/tmp/prediction_error_plot.png'
    plt.savefig(temp_file2, bbox_inches='tight', dpi=300)
    plt.close(fig2)

    # Add anomaly statistics
    print("\nAnomaly Detection Statistics:")
    print(f"Total number of anomalies detected: {len(anomaly_indices)}")
    
    # Calculate anomalies per channel
    channel_anomalies = {}
    for i, (e_low, e_high) in enumerate(energy_ranges):
        channel_data = df[f'counts_ch{i}'].iloc[anomaly_indices]
        significant_anomalies = len(channel_data[channel_data > df[f'counts_ch{i}'].mean() + 2*df[f'counts_ch{i}'].std()])
        channel_anomalies[i] = significant_anomalies
        print(f"\nChannel {i} ({e_low:.1f}-{e_high:.1f} keV):")
        print(f"  Significant anomalies: {significant_anomalies}")
        print(f"  Max anomaly count: {channel_data.max():.2f}")
        print(f"  Mean anomaly count: {channel_data.mean():.2f}")
    
    # Calculate temporal distribution
    time_windows = pd.cut(df['time'].iloc[anomaly_indices], bins=5)
    print("\nTemporal Distribution of Anomalies:")
    for interval, count in time_windows.value_counts().items():
        start_time = interval.left / 100  # Convert centiseconds to seconds
        end_time = interval.right / 100
        print(f"  {start_time:.1f}s - {end_time:.1f}s: {count} anomalies")
    
    # Upload to S3
    try:
        s3_client = boto3.client('s3')
        
        # Upload channel plots
        s3_client.upload_file(
            temp_file1, 
            bucket_name, 
            'plots/channels_plot.png'
        )
        
        # Upload prediction error plot
        s3_client.upload_file(
            temp_file2, 
            bucket_name, 
            'plots/prediction_error_plot.png'
        )
        
        print(f"Plots successfully uploaded to s3://{bucket_name}/plots/")
        
    except Exception as e:
        print(f"Error uploading to S3: {str(e)}")
    
    # Clean up temporary files
    import os
    os.remove(temp_file1)
    os.remove(temp_file2)

# Usage
plot_solar_flare_anomalies(
    df, 
    energy_ranges, 
    normalized_data, 
    predictions, 
    anomaly_indices,
    bucket_name='esa-solorb-anomalies-blog'
)

# Create a new dataframe with original data and anomaly information
results_df = df.copy()

# Calculate anomalies based on prediction error
actual = normalized_data[50:50+len(predictions)]
mse = np.mean(np.power(actual - predictions, 2), axis=1)
threshold = 0.0112  # or use np.mean(mse) + 2*np.std(mse)
anomalies = (mse > threshold).astype(int)
reconstruction_error = mse

# Add anomaly-related columns
# First create full-length arrays with NaN or 0s for the sequence_length gap
results_df['is_anomaly'] = 0
results_df['reconstruction_error'] = np.nan

# Fill in the values after sequence_length
sequence_length = 50  # Make sure this matches your model's sequence length
results_df.loc[sequence_length:sequence_length + len(anomalies) - 1, 'is_anomaly'] = anomalies
results_df.loc[sequence_length:sequence_length + len(reconstruction_error) - 1, 'reconstruction_error'] = reconstruction_error

# Add threshold value as reference
results_df['anomaly_threshold'] = threshold

# Save to temporary CSV file
temp_file = '/tmp/stix_data_with_anomalies.csv'
results_df.to_csv(temp_file, index=False)

# Upload to S3
try:
    s3_client = boto3.client('s3')
    bucket_name='esa-solorb-anomalies-blog'
    
    # Upload CSV file
    output_key = 'results/stix_data_with_anomalies.csv'
    s3_client.upload_file(
        temp_file, 
        bucket_name,
        output_key
    )
    
    print(f"\nData successfully uploaded to s3://{bucket_name}/{output_key}")
    print("\nColumns in saved file:")
    for col in results_df.columns:
        print(f"- {col}")
    print(f"\nTotal rows: {len(results_df)}")
    print(f"Number of anomalies: {results_df['is_anomaly'].sum()}")
    
except Exception as e:
    print(f"Error uploading to S3: {str(e)}")

# Clean up temporary file
import os
os.remove(temp_file)