# 🤖 ML Regression Dashboard

**TensorFlow Nightly Edition with Model Persistence**

A comprehensive machine learning dashboard for regression analysis with automatic model training, comparison, and persistence features.

## 🚀 Quick Start (Any Device)

This application now comes with **automatic setup** that works on any new device!

### Windows Users
1. **Double-click** `start.sh`, `start.py` or `start.bat`
2. That's it! The application will:
   - Automatically install all required packages
   - Set up the environment
   - Launch the dashboard in your browser

### Mac/Linux Users
1. **Double-click** `start.sh`, `start.py` or `start.bat` (or run `./start.sh` in terminal)
2. That's it! The application will:
   - Automatically install all required packages
   - Set up the environment
   - Launch the dashboard in your browser

### Alternative Method (All Systems)
```bash
python start.py
```

## ✨ Features

### 🤖 Machine Learning Models
- **Linear Regression** (with Cross-Validation)
- **Polynomial Regression** (with Cross-Validation)  
- **Exponential Regression** (with Cross-Validation)
- **Artificial Neural Network (ANN)** - TensorFlow Nightly optimized
- **Recurrent Neural Network (RNN)** - LSTM with sequence modeling

### 💾 Model Persistence
- **Auto-save** trained models
- **Load** previously trained models instantly
- **Manage** multiple model versions
- **Metadata** tracking (training date, performance, etc.)
- **Never retrain** - models persist across sessions

### 📊 Data Analysis
- **Automatic** column detection (Turkish/English labels)
- **Interactive** data visualization
- **Correlation** analysis
- **Statistical** summaries
- **Input-Output** relationship plots

### 🎯 Prediction & Analysis
- **Real-time** predictions with any trained model
- **Model comparison** - test all models simultaneously
- **Variable effect analysis** - see impact of individual variables
- **Performance** metrics (R², MAPE)
- **Model equations** display

### 🔬 Advanced Features
- **GPU acceleration** (automatic detection)
- **XLA JIT compilation** for faster TensorFlow execution
- **Cross-validation** for robust model evaluation
- **Batch normalization** and **dropout** for neural networks
- **Early stopping** and **learning rate scheduling**

## 📋 Supported Data Format

Upload Excel files (.xlsx) with numerical data:

| Input Column 1 | Input Column 2 | Output Column 1 | Output Column 2 |
|----------------|----------------|-----------------|-----------------|
| 55.0           | 90.5           | 10.5            | 85.3            |
| 60.0           | 91.2           | 10.8            | 86.1            |
| ...            | ...            | ...             | ...             |

- **Turkish labels** supported (Girdi/Çıktı columns)
- **Automatic preprocessing** and cleaning
- **Missing value** handling
- **Data type** conversion

## 🔧 System Requirements

- **Python 3.7+** (automatically detected)
- **Internet connection** (for initial package installation)
- **4GB RAM** minimum (8GB+ recommended for large datasets)
- **GPU optional** (CUDA-compatible for acceleration)

## 📁 File Structure

```
ml-dashboard/
├── ml.py           # Main application
├── start.py        # Cross-platform launcher (automatic setup)
├── start.bat       # Windows double-click launcher  
├── start.sh        # Mac/Linux double-click launcher
├── README.md       # This file
└── saved_models/   # Auto-created for model persistence
```

## 🛠️ What the Auto-Launcher Does

1. **Checks** your system (OS, Python version)
2. **Upgrades** pip to latest version
3. **Installs** all required packages:
   - streamlit, pandas, numpy, matplotlib, seaborn
   - scikit-learn, tensorflow (nightly), scipy
   - openpyxl, joblib, xlrd (Excel support)
4. **Creates** necessary directories
5. **Detects** GPU support
6. **Launches** the Streamlit application

## 🌐 Using the Dashboard

1. **Upload** your Excel dataset
2. **Select** input and output columns (auto-detected)
3. **Train** all models with one click
4. **Compare** model performance
5. **Make** predictions with the best model
6. **Save** models for future use
7. **Analyze** variable effects on outputs

## 🔍 Troubleshooting

### Application Won't Start
- Ensure Python 3.7+ is installed
- Check internet connection for package downloads
- Try running `python start.py` manually

### Package Installation Fails  
- The launcher will continue with available packages
- Install failed packages manually: `pip install package_name`
- Check firewall/proxy settings

### Performance Issues
- Close other applications to free memory
- Use smaller datasets for testing
- GPU acceleration will be used automatically if available

## 💡 Pro Tips

- **Model persistence** means you only train once per dataset
- **Auto-save** is enabled by default - your models are safe
- **Variable effect analysis** shows which inputs matter most
- **Cross-validation** models are more robust for predictions
- **GPU support** significantly speeds up neural network training

## 🆘 Support

If you encounter issues:
1. Check the console output for specific error messages
2. Ensure all files (ml.py, start.py) are in the same directory
3. Try running `pip install --upgrade pip` manually
4. For Windows: Run as Administrator if needed

---

**🎉 Ready to explore your data with machine learning!**
