# Algae Bloom Detection System

This project analyzes satellite imagery to detect and quantify algae blooms in lakes using computer vision and machine learning techniques.

## Overview

The system processes Sentinel-2 satellite imagery to:
1. Download and segment lake imagery
2. Detect water bodies
3. Identify potential algae blooms
4. Calculate contamination percentages

## Features

- Automated satellite imagery download using Google Earth Engine
- Water body detection and masking
- Algae bloom detection using computer vision
- Contamination level calculation
- Visual overlay generation

## Setup

1. Create a `.env` file with:
   ```
   PROJECT_ID=your_google_earth_engine_project_id
   TF_ENABLE_ONEDNN_OPTS=0
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Authenticate with Google Earth Engine

## Usage

Run the main script with:

```bash
python main.py
```

You can modify the latitude, longitude, and zoom level in `main.py` to analyze different lakes.

## Project Structure

- `main.py` - Entry point and configuration
- `datapull.py` - Satellite imagery fetching
- `datarefine.py` - Image processing and tiling
- `waterrefine.py` - Water body and algae detection
- `wateranalysis.py` - Core analysis logic
- `algae_utils/` - Utility functions for:
  - Feature extraction
  - Image segmentation
  - Classification
  - Evaluation

## Output

The system generates:
- Processed satellite imagery tiles
- Algae bloom overlay images
- Contamination level statistics
