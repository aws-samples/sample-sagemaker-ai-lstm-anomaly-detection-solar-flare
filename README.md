# ESA Solar Orbiter STIX Anomaly Detection

This project implements Long Short-Term Memory (LSTM) neural networks for detecting solar flares in ESA's Solar Orbiter STIX (Spectrometer/Telescope for Imaging X-rays) data using Amazon SageMaker.

## Overview
The code analyzes multi-channel X-ray data to detect anomalies across different energy bands:
- 4.0-10.0 keV (Low energy channel)
- 10.0-15.0 keV
- 15.0-25.0 keV
- 25.0-50.0 keV
- 50.0-84.0 keV (High energy channel)

## Prerequisites
- AWS Account with appropriate permissions
- Amazon SageMaker access
- Python 3.7 or later
- Required Python packages:
  - PyTorch
  - Astropy (for FITS file handling)
  - Pandas
  - NumPy
  - Matplotlib
  - Seaborn

## Architecture
![ESA Solar Orbiter Architecture](ESA%20SolOrb%20arch.png)

## Set-up
1. Clone the repository:
```
git clone https://github.com/aws-samples/sample-sagemaker-ai-lstm-anomaly-detection-solar-flare.git

cd sample-sagemaker-ai-rcf-anomaly-detection-lunar-spacecraft
```

## Usage
- Access STIX data through ESA's Solar Orbiter Archive (SOAR)
- Follow the Jupyter Notebook for implementation details
- The system processes FITS files and generates:
  - Multi-channel anomaly detection plots
  - CSV files with anomaly flags
  - Prediction error analysis

## Results
- Successfully detects solar flare events across energy channels
- Identifies ~405 anomalous points in the dataset
- Provides visualization of events across different energy bands
- Generates comprehensive analysis of detection results

## License

This project is licensed under the MIT-0 License - see LICENSE.txt for details.
