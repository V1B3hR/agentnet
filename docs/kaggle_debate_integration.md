# Kaggle Debate Datasets Integration

This document describes the integration of Kaggle debate datasets into AgentNet's CI/CD workflow for automated training.

## Overview

The integration includes:
- GitHub Actions workflow for automated dataset download and training
- Training script that processes debate datasets and runs AgentNet simulations  
- Support for three Kaggle datasets focused on debates and international diplomacy

## Datasets

### 1. UN General Debates
- **Source**: `unitednations/un-general-debates`
- **Content**: Historical speeches and statements from UN General Assembly
- **Usage**: Provides formal diplomatic debate content and international perspectives

### 2. SOHO Forum Debate Results  
- **Source**: `martj42/the-soho-forum-debate-results`
- **Content**: Structured debate results with positions and outcomes
- **Usage**: Provides debate format examples and position analysis

### 3. UN General Debate Corpus 1946-2023
- **Source**: `namigabbasov/united-nations-general-debate-corpus-1946-2023`
- **Content**: Comprehensive corpus of UN debates spanning decades
- **Usage**: Historical context and evolution of international discourse

## GitHub Actions Workflow

### File: `.github/workflows/debate-training.yml`

The workflow:
1. **Triggers**: Runs on push to main, PRs, and manual dispatch
2. **Environment**: Ubuntu with Python 3.11
3. **Dependencies**: Installs AgentNet, Kaggle API, pandas, numpy, scikit-learn
4. **Authentication**: Uses GitHub Secrets for Kaggle credentials
5. **Download**: Fetches all three datasets using Kaggle API
6. **Training**: Executes the debate model training script
7. **Artifacts**: Uploads training results and logs
8. **Testing**: Runs existing tests to ensure no regressions

### Required GitHub Secrets

Set these in your repository settings:
- `KAGGLE_USERNAME`: Your Kaggle username
- `KAGGLE_KEY`: Your Kaggle API key

## Training Script

### File: `scripts/train_debate_model.py`

The script includes:

#### DebateDatasetProcessor
- Loads and validates Kaggle datasets
- Handles different CSV formats and structures
- Extracts debate-relevant content (arguments, positions, topics)
- Provides comprehensive preprocessing for training

#### DebateModelTrainer  
- Integrates with AgentNet's debate functionality
- Creates training scenarios from processed data
- Runs multi-agent debate simulations
- Generates training metrics and results

#### Key Features
- **Robust Error Handling**: Gracefully handles missing datasets or files
- **Flexible Data Processing**: Adapts to different CSV column structures
- **AgentNet Integration**: Uses existing debate methods from AgentNet.py
- **Comprehensive Logging**: Detailed progress and error reporting
- **Artifact Generation**: Creates training reports, results, and summaries

## Usage

### Automatic Execution
The workflow runs automatically on:
- Push to main branch
- Pull request creation/updates
- Manual workflow dispatch

### Manual Execution
```bash
# Install dependencies
pip install pandas numpy scikit-learn kaggle

# Set up Kaggle credentials
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"

# Download datasets
kaggle datasets download -d unitednations/un-general-debates -p datasets/ --unzip
kaggle datasets download -d martj42/the-soho-forum-debate-results -p datasets/ --unzip  
kaggle datasets download -d namigabbasov/united-nations-general-debate-corpus-1946-2023 -p datasets/ --unzip

# Run training
python scripts/train_debate_model.py
```

## Output Artifacts

The training process generates:

### training_output/
- `training_results.json`: Complete training results with metrics
- `training_data_summary.json`: Dataset statistics and processing summary
- `training_report.md`: Human-readable training report

### Logs
- `debate_training.log`: Detailed execution log with timestamps

## Integration with AgentNet

The training script leverages AgentNet's existing debate capabilities:
- Uses `AgentNet.debate()` method for multi-agent simulations
- Creates agents with different cognitive styles (logic, creativity, analytical)
- Processes debate scenarios derived from real diplomatic content
- Generates training data for future model improvements

## Dataset Preprocessing

### UN General Debates
- Extracts country positions and statements
- Filters for substantial content (>50 characters)
- Preserves country attribution and temporal context

### SOHO Forum Results
- Processes structured debate positions and outcomes
- Extracts topic-position-result relationships
- Maintains debate format and scoring information

### UN Corpus 1946-2023
- Handles historical speech content
- Preserves temporal evolution (1946-2023)
- Extracts long-form diplomatic arguments

## Configuration

### Workflow Configuration
The GitHub Actions workflow can be customized by modifying:
- Python version in `uses: actions/setup-python@v4`
- Dataset download commands in the download step
- Training script parameters

### Training Configuration
The training script supports various parameters:
- Dataset directory location
- Number of training scenarios
- Agent style configurations
- Output directory settings

## Error Handling

The system handles common issues:
- **Missing Datasets**: Continues with available datasets
- **Invalid CSV Format**: Logs warnings and skips problematic files
- **Kaggle API Errors**: Provides clear error messages
- **AgentNet Import Issues**: Falls back to mock training simulation

## Monitoring and Debugging

### Logs
- GitHub Actions provides workflow execution logs
- Training script generates detailed local logs
- Error messages include context and suggestions

### Artifacts
- Training artifacts are preserved for 30 days
- Results can be downloaded from GitHub Actions interface
- Local execution creates persistent output files

## Future Enhancements

Potential improvements:
- **Model Evaluation**: Add quantitative metrics for debate quality
- **Dataset Expansion**: Support for additional debate datasets
- **Real-time Training**: Continuous learning from new datasets
- **Performance Optimization**: Caching and incremental processing
- **Advanced Preprocessing**: Natural language processing for better content extraction

## Troubleshooting

### Common Issues

1. **Kaggle Authentication Failed**
   - Verify GitHub Secrets are set correctly
   - Check Kaggle API key permissions

2. **Dataset Download Timeout**
   - Large datasets may require increased timeout values
   - Consider downloading subsets for testing

3. **Training Script Errors**
   - Check Python dependencies are installed
   - Verify dataset file formats match expected structure

4. **AgentNet Import Issues**
   - Ensure AgentNet is properly installed
   - Check Python path includes project directory

### Debug Commands

```bash
# Test Kaggle API
kaggle datasets list

# Verify dataset structure
head -5 datasets/*.csv

# Run training with verbose logging
python scripts/train_debate_model.py --verbose

# Check AgentNet installation
python -c "from AgentNet import AgentNet; print('AgentNet available')"
```

## License and Compliance

- Respects Kaggle dataset licenses and terms of use
- Follows AgentNet project licensing (GPL-3.0)
- Maintains attribution for dataset sources
- Complies with GitHub Actions usage policies