import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_percentage_error, make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.optimize import curve_fit
import tensorflow as tf
from tensorflow import keras
import warnings
import pickle
import joblib
import os
from datetime import datetime
import json
warnings.filterwarnings('ignore')

tf.config.optimizer.set_jit(True)
tf.config.run_functions_eagerly(False)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Configured {len(gpus)} GPU(s) with memory growth enabled")
    except RuntimeError as e:
        print(f"GPU memory configuration error: {e}")
        print("Note: GPU memory configuration must be set at program startup")
    except Exception as e:
        print(f"GPU configuration warning: {e}")
        pass

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)

st.set_page_config(page_title="ML Regression Dashboard", layout="wide")

class MLModels:
    def __init__(self):
        self.X_scaler = MinMaxScaler(feature_range=(0, 1))
        self.y_scaler = MinMaxScaler(feature_range=(0, 1))
        self.models = {}
        self.model_scalers = {} 
        self.results = {}
        self.input_cols = []
        self.output_cols = []
        self.model_metadata = {}
        
        # Create models directory if it doesn't exist
        if not os.path.exists('saved_models'):
            os.makedirs('saved_models')
    
    def save_models(self, filename=None):
        """Save all trained models and associated data to disk"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"ml_models_{timestamp}"
            
            # Prepare data to save
            save_data = {
                'results': self.results,
                'input_cols': self.input_cols,
                'output_cols': self.output_cols,
                'model_metadata': self.model_metadata,
                'X_scaler': self.X_scaler,
                'y_scaler': self.y_scaler,
                'model_scalers': self.model_scalers
            }
            
            # Save sklearn models and scalers
            sklearn_models = {}
            tensorflow_models = {}
            
            for model_name, model in self.models.items():
                if model_name in ['ANN', 'RNN']:
                    # Save TensorFlow models separately with .keras extension
                    tf_model_path = f'saved_models/{filename}_{model_name}.keras'
                    model.save(tf_model_path)
                    tensorflow_models[model_name] = tf_model_path
                else:
                    # Save sklearn models with pickle
                    sklearn_models[model_name] = model
            
            save_data['sklearn_models'] = sklearn_models
            save_data['tensorflow_models'] = tensorflow_models
            
            # Save everything except TF models with joblib
            joblib.dump(save_data, f'saved_models/{filename}.pkl')
            
            return filename
            
        except Exception as e:
            st.error(f"Error saving models: {str(e)}")
            return None
    
    def load_models(self, filename):
        """Load previously saved models from disk"""
        try:
            # Load main data
            save_data = joblib.load(f'saved_models/{filename}.pkl')
            
            self.results = save_data['results']
            self.input_cols = save_data['input_cols']
            self.output_cols = save_data['output_cols']
            self.model_metadata = save_data.get('model_metadata', {})
            self.X_scaler = save_data['X_scaler']
            self.y_scaler = save_data['y_scaler']
            self.model_scalers = save_data['model_scalers']
            
            # Load sklearn models
            self.models.update(save_data['sklearn_models'])
            
            # Load TensorFlow models
            for model_name, model_path in save_data['tensorflow_models'].items():
                if os.path.exists(model_path):
                    self.models[model_name] = keras.models.load_model(model_path)
            
            return True
            
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return False
    
    def get_saved_models_list(self):
        """Get list of saved model files"""
        try:
            files = [f for f in os.listdir('saved_models') if f.endswith('.pkl')]
            return [f.replace('.pkl', '') for f in files]
        except:
            return []
    
    def delete_saved_model(self, filename):
        """Delete a saved model file"""
        try:
            # Delete main pickle file
            if os.path.exists(f'saved_models/{filename}.pkl'):
                os.remove(f'saved_models/{filename}.pkl')
            
            # Delete TensorFlow model files (.keras extension)
            for model_type in ['ANN', 'RNN']:
                tf_model_path = f'saved_models/{filename}_{model_type}.keras'
                if os.path.exists(tf_model_path):
                    os.remove(tf_model_path)
                
                # Also check for old directory-style saves (for backward compatibility)
                tf_model_dir = f'saved_models/{filename}_{model_type}'
                if os.path.exists(tf_model_dir) and os.path.isdir(tf_model_dir):
                    import shutil
                    shutil.rmtree(tf_model_dir)
            
            return True
        except Exception as e:
            st.error(f"Error deleting model: {str(e)}")
            return False

    def preprocess_data(self, df):
        try:
            df_processed = df.copy()
            
            first_row = df_processed.iloc[0]
            column_names = df_processed.columns.tolist()
            has_turkish_labels = any(str(col).startswith(("Girdi", "Çıktı")) for col in column_names)
            
            if has_turkish_labels:
                df_processed.columns = df_processed.iloc[0]  
                df_processed = df_processed[1:].reset_index(drop=True)
                
                if 'Tarih' in df_processed.columns:
                    df_processed = df_processed.drop(columns=['Tarih'])
            
            for col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                df_processed[col] = df_processed[col].astype('float64')
            
            df_processed = df_processed.dropna(how='all', axis=1)
            df_processed = df_processed.dropna(how='all', axis=0)
            df_processed = df_processed.dropna()
            for col in df_processed.columns:
                if df_processed[col].dtype == 'object':
                    try:
                        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').astype('float64')
                    except:
                        df_processed = df_processed.drop(columns=[col])
                        st.warning(f"Dropped column {col} due to type conversion issues")
            
            df_processed = df_processed.dropna()
            if df_processed.empty:
                st.error("No valid numeric data found after preprocessing!")
                return df
            
            non_numeric_cols = df_processed.select_dtypes(exclude=['number']).columns
            if len(non_numeric_cols) > 0:
                st.warning(f"Dropping non-numeric columns: {list(non_numeric_cols)}")
                df_processed = df_processed.select_dtypes(include=['number'])
            return df_processed
            
        except Exception as e:
            st.error(f"Error in preprocessing: {str(e)}")
            df_fallback = df.copy()
            for col in df_fallback.columns:
                try:
                    df_fallback[col] = pd.to_numeric(df_fallback[col], errors='coerce').astype('float64')
                except:
                    df_fallback = df_fallback.drop(columns=[col])
            
            df_fallback = df_fallback.select_dtypes(include=['number']).dropna()
            return df_fallback
    
    def identify_columns(self, df):
        try:
            first_row = df.iloc[0]
            
            girdi_labels = [label for label in first_row.index if str(label).startswith("Girdi")]
            cikti_labels = [label for label in first_row.index if str(label).startswith("Çıktı")]
            
            if girdi_labels and cikti_labels:
                girdi_column_indices = [df.columns.get_loc(col) for col in girdi_labels]
                cikti_column_indices = [df.columns.get_loc(col) for col in cikti_labels]
                
                df_temp = df.copy()
                df_temp.columns = df_temp.iloc[0]
                df_temp = df_temp[1:]
                
                all_columns = df_temp.columns.tolist()
                input_cols = [all_columns[i] for i in girdi_column_indices]
                output_cols = [all_columns[i] for i in cikti_column_indices]
                
                return input_cols, output_cols
            
        except Exception as e:
            st.warning(f"Could not identify Turkish labels: {str(e)}")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            st.error("No numeric columns found in the dataset!")
            return [], []
        
        mid_point = len(numeric_cols) // 2
        input_cols = numeric_cols[:mid_point] if mid_point > 0 else numeric_cols[:1]
        output_cols = numeric_cols[mid_point:] if mid_point < len(numeric_cols) else numeric_cols[-1:]
        
        st.info("Using fallback column assignment - please manually select columns if needed.")
        return input_cols, output_cols
    
    def linear_regression(self, X, y):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=344)
        
        models = []
        r2_scores = []
        mape_scores = []
        
        for out_idx in range(y.shape[1]):
            model = LinearRegression()
            model.fit(X_train, y_train[:, out_idx])
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test[:, out_idx], y_pred)
            mape = mean_absolute_percentage_error(y_test[:, out_idx], y_pred)
            
            models.append(model)
            r2_scores.append(r2)
            mape_scores.append(mape)
        
        avg_r2 = np.mean(r2_scores)
        avg_mape = np.mean(mape_scores)
        
        self.models['Linear Regression'] = (models, scaler)
        return {"R²": avg_r2, "MAPE": avg_mape, "details": (r2_scores, mape_scores)}
    
    def polynomial_regression(self, X, y, degree=2):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        poly_model.fit(X_train, y_train)
        
        y_pred = poly_model.predict(X_test)
        
        r2_scores = []
        mape_scores = []
        
        for i in range(y.shape[1]):
            r2 = r2_score(y_test[:, i], y_pred[:, i])
            mape = mean_absolute_percentage_error(y_test[:, i], y_pred[:, i])
            r2_scores.append(r2)
            mape_scores.append(mape)
        
        avg_r2 = np.mean(r2_scores)
        avg_mape = np.mean(mape_scores)
        
        self.models['Polynomial Regression'] = poly_model
        return {"R²": avg_r2, "MAPE": avg_mape, "details": (r2_scores, mape_scores)}
    
    def exponential_regression(self, X, y):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=344)
        
        models = []
        r2_scores = []
        mape_scores = []
        
        for out_idx in range(y.shape[1]):
            if np.any(y_train[:, out_idx] <= 0) or np.any(y_test[:, out_idx] <= 0):
                models.append(None)
                r2_scores.append(0.0)
                mape_scores.append(1.0)
                continue
                
            y_train_log = np.log(y_train[:, out_idx])
            model = LinearRegression()
            model.fit(X_train, y_train_log)
            y_pred_log = model.predict(X_test)
            y_pred_exp = np.exp(y_pred_log)
            r2 = r2_score(y_test[:, out_idx], y_pred_exp)
            mape = mean_absolute_percentage_error(y_test[:, out_idx], y_pred_exp)
            
            models.append(model)
            r2_scores.append(r2)
            mape_scores.append(mape)
        
        avg_r2 = np.mean(r2_scores)
        avg_mape = np.mean(mape_scores)
        
        self.models['Exponential Regression'] = (models, scaler)
        return {"R²": avg_r2, "MAPE": avg_mape, "details": (r2_scores, mape_scores)}
    
    def linear_regression_cv(self, X, y, cv_folds=5):
        """Linear regression with cross-validation for more robust evaluation"""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create custom scorer for MAPE
        def mape_scorer(y_true, y_pred):
            return -mean_absolute_percentage_error(y_true, y_pred)  # Negative because sklearn maximizes
        
        mape_custom_scorer = make_scorer(mape_scorer)
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        r2_scores_all = []
        mape_scores_all = []
        models = []
        
        for out_idx in range(y.shape[1]):
            model = LinearRegression()
            
            # Cross-validation for R²
            r2_cv_scores = cross_val_score(model, X_scaled, y[:, out_idx], cv=kf, scoring='r2')
            r2_mean = np.mean(r2_cv_scores)
            
            # Cross-validation for MAPE
            mape_cv_scores = cross_val_score(model, X_scaled, y[:, out_idx], cv=kf, scoring=mape_custom_scorer)
            mape_mean = -np.mean(mape_cv_scores)  # Convert back to positive
            
            # Train final model on full dataset
            model.fit(X_scaled, y[:, out_idx])
            models.append(model)
            
            r2_scores_all.append(r2_mean)
            mape_scores_all.append(mape_mean)
        
        avg_r2 = np.mean(r2_scores_all)
        avg_mape = np.mean(mape_scores_all)
        
        self.models['Linear Regression (CV)'] = (models, scaler)
        return {"R²": avg_r2, "MAPE": avg_mape, "details": (r2_scores_all, mape_scores_all)}
    
    def polynomial_regression_cv(self, X, y, degree=2, cv_folds=5):
        """Polynomial regression with cross-validation for more robust evaluation"""
        # Create custom scorer for MAPE
        def mape_scorer(y_true, y_pred):
            # Handle multi-output case
            if y_pred.ndim > 1:
                mape_values = []
                for i in range(y_pred.shape[1]):
                    mape_values.append(mean_absolute_percentage_error(y_true[:, i], y_pred[:, i]))
                return -np.mean(mape_values)
            else:
                return -mean_absolute_percentage_error(y_true, y_pred)
        
        mape_custom_scorer = make_scorer(mape_scorer)
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        
        # Cross-validation for R²
        r2_cv_scores = cross_val_score(poly_model, X, y, cv=kf, scoring='r2')
        r2_mean = np.mean(r2_cv_scores)
        
        # Cross-validation for MAPE  
        mape_cv_scores = cross_val_score(poly_model, X, y, cv=kf, scoring=mape_custom_scorer)
        mape_mean = -np.mean(mape_cv_scores)
        
        # Train final model on full dataset
        poly_model.fit(X, y)
        
        # Calculate individual output scores for details
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        poly_model_detail = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        poly_model_detail.fit(X_train, y_train)
        y_pred = poly_model_detail.predict(X_test)
        
        r2_scores = []
        mape_scores = []
        for i in range(y.shape[1]):
            r2 = r2_score(y_test[:, i], y_pred[:, i])
            mape = mean_absolute_percentage_error(y_test[:, i], y_pred[:, i])
            r2_scores.append(r2)
            mape_scores.append(mape)
        
        self.models['Polynomial Regression (CV)'] = poly_model
        return {"R²": r2_mean, "MAPE": mape_mean, "details": (r2_scores, mape_scores)}
    
    def exponential_regression_cv(self, X, y, cv_folds=5):
        """Exponential regression with cross-validation for more robust evaluation"""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        models = []
        r2_scores_all = []
        mape_scores_all = []
        
        for out_idx in range(y.shape[1]):
            if np.any(y[:, out_idx] <= 0):
                models.append(None)
                r2_scores_all.append(0.0)
                mape_scores_all.append(1.0)
                continue
            
            y_log = np.log(y[:, out_idx])
            
            # Cross-validation scores
            r2_cv_scores = []
            mape_cv_scores = []
            
            for train_idx, test_idx in kf.split(X_scaled):
                X_train_fold, X_test_fold = X_scaled[train_idx], X_scaled[test_idx]
                y_train_fold, y_test_fold = y_log[train_idx], y[:, out_idx][test_idx]
                
                model_fold = LinearRegression()
                model_fold.fit(X_train_fold, y_train_fold)
                
                y_pred_log = model_fold.predict(X_test_fold)
                y_pred_exp = np.exp(y_pred_log)
                
                r2_fold = r2_score(y_test_fold, y_pred_exp)
                mape_fold = mean_absolute_percentage_error(y_test_fold, y_pred_exp)
                
                r2_cv_scores.append(r2_fold)
                mape_cv_scores.append(mape_fold)
            
            r2_mean = np.mean(r2_cv_scores)
            mape_mean = np.mean(mape_cv_scores)
            
            # Train final model on full dataset
            model = LinearRegression()
            model.fit(X_scaled, y_log)
            models.append(model)
            
            r2_scores_all.append(r2_mean)
            mape_scores_all.append(mape_mean)
        
        avg_r2 = np.mean(r2_scores_all)
        avg_mape = np.mean(mape_scores_all)
        
        self.models['Exponential Regression (CV)'] = (models, scaler)
        return {"R²": avg_r2, "MAPE": avg_mape, "details": (r2_scores_all, mape_scores_all)}
    
    def ann_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train)
        model = keras.Sequential([
            keras.layers.Input(shape=(X.shape[1],), name='input_layer'),
            keras.layers.Dense(64, activation='relu', name='dense_1'),
            keras.layers.BatchNormalization(name='batch_norm_1'),
            keras.layers.Dropout(0.4, name='dropout_1'),

            keras.layers.Dense(32, activation='relu', name='dense_2'),
            keras.layers.BatchNormalization(name='batch_norm_2'),
            keras.layers.Dropout(0.3, name='dropout_2'),

            keras.layers.Dense(y.shape[1], activation='linear', name='output_layer')
        ])
        
        optimizer = keras.optimizers.AdamW(
            learning_rate=0.001,
            weight_decay=0.0001
        )
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=8,  # reduced from 15
                restore_best_weights=True,
                verbose=0
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6,
                verbose=0
            )
        ]
        
        history = model.fit(
            X_train_scaled, y_train_scaled,
            epochs=50,  # reduced from 100
            batch_size=32,
            validation_split=0.2,
            verbose=0,
            callbacks=callbacks
        )
        
        y_pred_scaled = model.predict(X_test_scaled, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        
        r2_scores = []
        mape_scores = []
        
        for i in range(y.shape[1]):
            r2 = r2_score(y_test[:, i], y_pred[:, i])
            mape = mean_absolute_percentage_error(y_test[:, i], y_pred[:, i])
            r2_scores.append(r2)
            mape_scores.append(mape)
        
        avg_r2 = np.mean(r2_scores)
        avg_mape = np.mean(mape_scores)
        
        self.models['ANN'] = model
        self.model_scalers['ANN'] = {'X_scaler': scaler_X, 'y_scaler': scaler_y}
        return {"R²": avg_r2, "MAPE": avg_mape, "details": (r2_scores, mape_scores)}
    
    def create_sequences(self, X, y, time_steps=10):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            Xs.append(X[i:(i + time_steps)])
            ys.append(y[i + time_steps])
        return np.array(Xs), np.array(ys)
    
    def rnn_model(self, X, y, time_steps=10):
        y_log = np.log(y)
        X_scaler = MinMaxScaler(feature_range=(0, 1))
        y_scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled = X_scaler.fit_transform(X)
        Y_scaled = y_scaler.fit_transform(y_log)
        X_seq, y_seq = self.create_sequences(X_scaled, Y_scaled, time_steps)
        
        if len(X_seq) == 0:
            return {"R²": 0.0, "MAPE": 1.0, "details": ([0], [1])}
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=0.2, random_state=42
        )
        
        model = keras.Sequential([
            keras.layers.LSTM(50, return_sequences=True, input_shape=(time_steps, X.shape[1])),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(50),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(y.shape[1])
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        y_pred_scaled = model.predict(X_test, verbose=0)
        y_pred_log = y_scaler.inverse_transform(y_pred_scaled)
        y_pred = np.exp(y_pred_log)
        y_test_log = y_scaler.inverse_transform(y_test)
        y_test_actual = np.exp(y_test_log)
        r2_scores = []
        mape_scores = []
        
        for i in range(y_test_actual.shape[1]):
            r2 = r2_score(y_test_actual[:, i], y_pred[:, i])
            mape = mean_absolute_percentage_error(y_test_actual[:, i], y_pred[:, i])
            r2_scores.append(r2)
            mape_scores.append(mape)
        
        avg_r2 = np.mean(r2_scores)
        avg_mape = np.mean(mape_scores)
        self.models['RNN'] = model
        self.model_scalers['RNN'] = {'X_scaler': X_scaler, 'y_scaler': y_scaler}
        return {"R²": avg_r2, "MAPE": avg_mape, "details": (r2_scores, mape_scores)}
    
    def run_all_models(self, X, y, input_cols=None, output_cols=None):
        results = {}
        
        # Store column information
        if input_cols:
            self.input_cols = input_cols
        if output_cols:
            self.output_cols = output_cols
        
        # Store metadata
        self.model_metadata = {
            'training_timestamp': datetime.now().isoformat(),
            'data_shape': X.shape,
            'output_shape': y.shape,
            'n_input_features': X.shape[1],
            'n_output_features': y.shape[1]
        }
        
        with st.spinner("Running Linear Regression..."):
            results["Linear Regression"] = self.linear_regression(X, y)
        
        with st.spinner("Running Linear Regression (CV)..."):
            results["Linear Regression (CV)"] = self.linear_regression_cv(X, y)
        
        with st.spinner("Running Polynomial Regression..."):
            results["Polynomial Regression"] = self.polynomial_regression(X, y)
        
        with st.spinner("Running Polynomial Regression (CV)..."):
            results["Polynomial Regression (CV)"] = self.polynomial_regression_cv(X, y)
        
        with st.spinner("Running Exponential Regression..."):
            results["Exponential Regression"] = self.exponential_regression(X, y)
        
        with st.spinner("Running Exponential Regression (CV)..."):
            results["Exponential Regression (CV)"] = self.exponential_regression_cv(X, y)
        
        with st.spinner("Running Enhanced ANN (TF Nightly)..."):
            results["ANN"] = self.ann_model(X, y)
        
        with st.spinner("Running Enhanced RNN (TF Nightly)..."):
            results["RNN"] = self.rnn_model(X, y)
        
        # Store results
        self.results = results
        
        return results
    
    def _sensitivity_analysis(self, results, input_cols, output_cols, df_processed, ml_models):
        """Dataset-wide delta analysis - replace one input column with constant value"""
        st.write("**📈 Dataset Delta Analysis**")
        st.write("Replace one input column with a constant value across ALL dataset rows and see the effect of that variable being held constant.")
        st.info("💡 **Delta Calculation**: delta = original_output - modified_output (shows the effect of the variable)")
        
        if not results or not input_cols or not output_cols:
            st.error("No model results available for variable effect analysis.")
            return
        
        # Model selection
        results_df = pd.DataFrame({
            model: {"R²": float(results[model]["R²"]), "MAPE": float(results[model]["MAPE"])} 
            for model in results.keys()
        }).T.sort_values("R²", ascending=False)
        
        best_model = results_df.index[0]
        model_options = list(results.keys())
        default_model_idx = model_options.index(best_model)
        
        selected_model = st.selectbox(
            "Select Model for Analysis:",
            options=model_options,
            index=default_model_idx,
            help=f"Best model ({best_model}) is selected by default"
        )
        
        # Select which input to replace with constant
        st.write("**📋 Select Input to Replace:**")
        col1, col2 = st.columns(2)
        
        with col1:
            selected_input = st.selectbox(
                "Choose input variable to replace:",
                options=input_cols,
                help="This input will be replaced with a constant value across all dataset rows"
            )
        
        with col2:
            # Show info about selected input
            col_data = df_processed[selected_input]
            min_val = float(col_data.min())
            max_val = float(col_data.max())
            mean_val = float(col_data.mean())
            
            st.info(f"""
            **{selected_input} in Dataset:**
            - Mean: {mean_val:.2f}
            - Min: {min_val:.2f}  
            - Max: {max_val:.2f}
            - Rows: {len(df_processed):,}
            """)
        
        # Interface for setting the constant value
        st.write(f"**🔧 Set Constant Value for {selected_input}:**")
        
        constant_value = st.number_input(
            f"Constant value to replace {selected_input} across all {len(df_processed):,} rows:",
            value=mean_val,
            min_value=min_val * 0.5,
            max_value=max_val * 1.5,
            step=(max_val - min_val) / 100,
            format="%.2f",
            key=f"constant_{selected_input}",
            help=f"This value will replace all {len(df_processed):,} values in the {selected_input} column"
        )
        
        # Show the replacement info
        original_values = col_data.values
        unique_original = len(col_data.unique())
        
        st.info(f"""
        **📊 Replacement Summary:**
        - **Column**: {selected_input}
        - **Original unique values**: {unique_original:,}
        - **Will be replaced with**: {constant_value:.2f}
        - **Rows affected**: {len(df_processed):,}
        """)
        
        # Show sample of original vs modified
        with st.expander("👁️ Sample: Original vs Modified"):
            sample_size = min(10, len(df_processed))
            sample_original = df_processed[input_cols].head(sample_size).copy()
            sample_modified = sample_original.copy()
            sample_modified[selected_input] = constant_value
            
            st.write("**Original (first 10 rows):**")
            st.dataframe(sample_original, use_container_width=True)
            
            st.write("**Modified (first 10 rows):**")
            st.dataframe(sample_modified, use_container_width=True)
        
        # Run analysis button
        if st.button("🔍 Calculate Variable Effect (Deltas)", type="primary"):
            try:
                with st.spinner(f"Calculating deltas for {len(df_processed):,} rows..."):
                    # Prepare original dataset
                    X_original = df_processed[input_cols].values
                    
                    # Prepare modified dataset (replace selected column with constant)
                    X_modified = X_original.copy()
                    selected_col_idx = input_cols.index(selected_input)
                    X_modified[:, selected_col_idx] = constant_value
                    
                    # Get predictions for original dataset
                    original_predictions = []
                    for i in range(len(X_original)):
                        pred = self._get_model_prediction(selected_model, X_original[i:i+1], ml_models)
                        original_predictions.append(pred)
                    
                    # Get predictions for modified dataset
                    modified_predictions = []
                    for i in range(len(X_modified)):
                        pred = self._get_model_prediction(selected_model, X_modified[i:i+1], ml_models)
                        modified_predictions.append(pred)
                    
                    # Convert to numpy arrays for easier calculation
                    original_predictions = np.array(original_predictions)
                    modified_predictions = np.array(modified_predictions)
                    
                    # Calculate deltas (original - modified to show the effect of the variable)
                    deltas = original_predictions - modified_predictions
                    
                    # Calculate percentage changes
                    percentage_changes = np.zeros_like(deltas)
                    non_zero_mask = original_predictions != 0
                    percentage_changes[non_zero_mask] = (deltas[non_zero_mask] / original_predictions[non_zero_mask]) * 100
                    
                    # Store results in session state to prevent loss on rerun
                    analysis_key = f"variable_effect_{selected_model}_{selected_input}_{constant_value}"
                    st.session_state[f"{analysis_key}_deltas"] = deltas
                    st.session_state[f"{analysis_key}_percentage_changes"] = percentage_changes
                    st.session_state[f"{analysis_key}_original_predictions"] = original_predictions
                    st.session_state[f"{analysis_key}_modified_predictions"] = modified_predictions
                    st.session_state[f"{analysis_key}_selected_input"] = selected_input
                    st.session_state[f"{analysis_key}_constant_value"] = constant_value
                    st.session_state[f"{analysis_key}_selected_model"] = selected_model
                    st.session_state["current_analysis_key"] = analysis_key
                
            except Exception as e:
                st.error(f"Error in variable effect analysis: {str(e)}")
                st.write("Please check your model selection and try again.")
        
        # Check for existing results in session state and display them
        if "current_analysis_key" in st.session_state and st.session_state["current_analysis_key"]:
            analysis_key = st.session_state["current_analysis_key"]
            
            # Check if we have all the required data
            if (f"{analysis_key}_deltas" in st.session_state and 
                f"{analysis_key}_percentage_changes" in st.session_state):
                
                # Retrieve stored results
                deltas = st.session_state[f"{analysis_key}_deltas"]
                percentage_changes = st.session_state[f"{analysis_key}_percentage_changes"]
                original_predictions = st.session_state[f"{analysis_key}_original_predictions"]
                modified_predictions = st.session_state[f"{analysis_key}_modified_predictions"]
                stored_selected_input = st.session_state[f"{analysis_key}_selected_input"]
                stored_constant_value = st.session_state[f"{analysis_key}_constant_value"]
                stored_selected_model = st.session_state[f"{analysis_key}_selected_model"]
                
                # Display results
                st.success(f"✅ Variable Effect Analysis Complete for {len(df_processed):,} rows!")
                st.info(f"📊 **Analysis**: {stored_selected_model} model with {stored_selected_input} = {stored_constant_value}")
                
                # Summary statistics
                st.write("**📊 Variable Effect Summary (Original - Modified):**")
                summary_cols = st.columns(len(output_cols))
                
                for i, out_col in enumerate(output_cols):
                    with summary_cols[i]:
                        col_deltas = deltas[:, i]
                        mean_delta = np.mean(col_deltas)
                        std_delta = np.std(col_deltas)
                        min_delta = np.min(col_deltas)
                        max_delta = np.max(col_deltas)
                        
                        st.metric(
                            label=f"**{out_col}**",
                            value=f"{mean_delta:+.2f}",
                            delta=f"σ={std_delta:.2f}",
                            help=f"Mean delta: {mean_delta:+.2f}\nStd: {std_delta:.2f}\nRange: [{min_delta:+.2f}, {max_delta:+.2f}]"
                        )
                
                # Detailed results table
                st.write("**📋 Detailed Results:**")
                
                # Create results dataframe
                results_data = {
                    'Row': range(1, len(df_processed) + 1),
                    f'Original_{stored_selected_input}': df_processed[stored_selected_input].values,
                    f'Modified_{stored_selected_input}': [stored_constant_value] * len(df_processed)
                }
                
                # Add original and modified predictions
                for i, out_col in enumerate(output_cols):
                    results_data[f'{out_col}_Original'] = original_predictions[:, i]
                    results_data[f'{out_col}_Modified'] = modified_predictions[:, i]
                    results_data[f'{out_col}_Delta'] = deltas[:, i]
                    results_data[f'{out_col}_Delta_%'] = percentage_changes[:, i]
                
                results_df = pd.DataFrame(results_data)
                
                # Display options with key to prevent rerun
                display_option = st.radio(
                    "Choose display option:",
                    ["Summary (10 rows)", "Full table", "Statistics only"],
                    horizontal=True,
                    key=f"display_option_{analysis_key}"
                )
                
                if display_option == "Summary (10 rows)":
                    st.dataframe(results_df.head(10), use_container_width=True)
                elif display_option == "Full table":
                    st.dataframe(results_df, use_container_width=True)
                elif display_option == "Statistics only":
                    st.info("📊 **Statistics Only View** - Detailed table hidden. Statistical summary is shown above, and visualizations are available below.")
                    
                    # Show condensed statistics table instead of row-by-row data
                    stats_summary = {}
                    for i, out_col in enumerate(output_cols):
                        col_deltas = deltas[:, i]
                        stats_summary[out_col] = {
                            "Mean Delta": f"{np.mean(col_deltas):+.2f}",
                            "Std Delta": f"{np.std(col_deltas):.2f}",
                            "Min Delta": f"{np.min(col_deltas):+.2f}",
                            "Max Delta": f"{np.max(col_deltas):+.2f}",
                            "Mean Original": f"{np.mean(original_predictions[:, i]):.2f}",
                            "Mean Modified": f"{np.mean(modified_predictions[:, i]):.2f}"
                        }
                    
                    stats_df = pd.DataFrame(stats_summary).T
                    st.dataframe(stats_df, use_container_width=True)
                
                # Download option
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label=f"📥 Download Full Results ({len(df_processed):,} rows) as CSV",
                    data=csv,
                    file_name=f"variable_effect_analysis_{stored_selected_model}_{stored_selected_input}.csv",
                    mime="text/csv"
                )
                
                # Visualizations
                st.write("**📈 Variable Effect Visualizations:**")
                
                for i, out_col in enumerate(output_cols):
                    with st.expander(f"📊 {out_col} Analysis"):
                        col_deltas = deltas[:, i]
                        
                        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
                        
                        # Delta distribution
                        ax1.hist(col_deltas, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                        ax1.set_title(f'{out_col} Delta Distribution')
                        ax1.set_xlabel('Delta Value')
                        ax1.set_ylabel('Frequency')
                        ax1.axvline(np.mean(col_deltas), color='red', linestyle='--', label=f'Mean: {np.mean(col_deltas):.2f}')
                        ax1.legend()
                        ax1.grid(True, alpha=0.3)
                        
                        # Original vs Modified scatter
                        ax2.scatter(original_predictions[:, i], modified_predictions[:, i], alpha=0.6, s=20)
                        ax2.plot([original_predictions[:, i].min(), original_predictions[:, i].max()], 
                                [original_predictions[:, i].min(), original_predictions[:, i].max()], 'r--', label='No Change Line')
                        ax2.set_xlabel(f'{out_col} Original')
                        ax2.set_ylabel(f'{out_col} Modified')
                        ax2.set_title(f'{out_col}: Original vs Modified')
                        ax2.legend()
                        ax2.grid(True, alpha=0.3)
                        
                        # Delta vs original values
                        ax3.scatter(original_predictions[:, i], col_deltas, alpha=0.6, s=20, color='orange')
                        ax3.set_xlabel(f'{out_col} Original Value')
                        ax3.set_ylabel(f'{out_col} Delta')
                        ax3.set_title(f'{out_col}: Delta vs Original Value')
                        ax3.axhline(0, color='black', linestyle='-', alpha=0.3)
                        ax3.grid(True, alpha=0.3)
                        
                        # Delta vs row index
                        ax4.plot(range(len(col_deltas)), col_deltas, alpha=0.7, color='green', linewidth=1)
                        ax4.set_xlabel('Row Index')
                        ax4.set_ylabel(f'{out_col} Delta')
                        ax4.set_title(f'{out_col}: Delta Across Dataset')
                        ax4.axhline(0, color='black', linestyle='-', alpha=0.3)
                        ax4.axhline(np.mean(col_deltas), color='red', linestyle='--', alpha=0.7, label=f'Mean: {np.mean(col_deltas):.2f}')
                        ax4.legend()
                        ax4.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Statistics for this output
                        st.write(f"**{out_col} Statistics:**")
                        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                        
                        with stats_col1:
                            st.metric("Mean Delta", f"{np.mean(col_deltas):+.2f}")
                        with stats_col2:
                            st.metric("Std Delta", f"{np.std(col_deltas):.2f}")
                        with stats_col3:
                            st.metric("Min Delta", f"{np.min(col_deltas):+.2f}")
                        with stats_col4:
                            st.metric("Max Delta", f"{np.max(col_deltas):+.2f}")
                
                # Clear results button
                if st.button("🗑️ Clear Analysis Results", key=f"clear_{analysis_key}"):
                    # Clear all session state related to this analysis
                    keys_to_clear = [k for k in st.session_state.keys() if analysis_key in k or k == "current_analysis_key"]
                    for key in keys_to_clear:
                        del st.session_state[key]
                    st.rerun()
    
    def _get_format_for_column(self, col_name):
        """Get appropriate decimal formatting for each column"""
        return ".2f"  # All values now use 2 decimal precision
    
    def _get_model_prediction(self, model_name, input_array, ml_models):
        """Helper method to get predictions from a specific model"""
        try:
            if model_name not in ml_models.models or ml_models.models[model_name] is None:
                return [0.0] * len(ml_models.output_cols if hasattr(ml_models, 'output_cols') else [1])
            
            model = ml_models.models[model_name]
            
            if model_name == "RNN":
                if model_name in ml_models.model_scalers:
                    rnn_scalers = ml_models.model_scalers[model_name]
                    time_steps = 10
                    input_scaled = rnn_scalers['X_scaler'].transform(input_array)
                    input_sequence = np.repeat(input_scaled, time_steps, axis=0).reshape(1, time_steps, -1)
                    pred_scaled = model.predict(input_sequence, verbose=0)
                    pred_log = rnn_scalers['y_scaler'].inverse_transform(pred_scaled)
                    predictions = np.exp(pred_log)[0]
                else:
                    time_steps = 10
                    input_scaled = ml_models.X_scaler.transform(input_array)
                    input_sequence = np.repeat(input_scaled, time_steps, axis=0).reshape(1, time_steps, -1)
                    pred_scaled = model.predict(input_sequence, verbose=0)
                    pred_log = ml_models.y_scaler.inverse_transform(pred_scaled)
                    predictions = np.exp(pred_log)[0]
            
            elif model_name == "ANN":
                if model_name in ml_models.model_scalers:
                    ann_scalers = ml_models.model_scalers[model_name]
                    input_scaled = ann_scalers['X_scaler'].transform(input_array)
                    pred_scaled = model.predict(input_scaled, verbose=0)
                    predictions = ann_scalers['y_scaler'].inverse_transform(pred_scaled)[0]
                else:
                    input_scaled = ml_models.X_scaler.transform(input_array)
                    pred_scaled = model.predict(input_scaled, verbose=0)
                    predictions = pred_scaled[0]
            
            elif model_name == "Linear Regression" or model_name == "Linear Regression (CV)":
                models, scaler = model
                input_scaled = scaler.transform(input_array)
                predictions = []
                for i, lin_model in enumerate(models):
                    pred = lin_model.predict(input_scaled)[0]
                    predictions.append(pred)
            
            elif model_name == "Polynomial Regression" or model_name == "Polynomial Regression (CV)":
                predictions = model.predict(input_array)[0]
            
            elif model_name == "Exponential Regression" or model_name == "Exponential Regression (CV)":
                models, scaler = model
                input_scaled = scaler.transform(input_array)
                predictions = []
                for i, exp_model in enumerate(models):
                    if exp_model is None:
                        predictions.append(0.0)
                    else:
                        try:
                            log_pred = exp_model.predict(input_scaled)[0]
                            pred = np.exp(log_pred)
                            predictions.append(pred)
                        except:
                            predictions.append(0.0)
            
            else:
                predictions = [0.0] * len(ml_models.output_cols if hasattr(ml_models, 'output_cols') else [1])
            
            return predictions
            
        except Exception as e:
            return [0.0] * len(ml_models.output_cols if hasattr(ml_models, 'output_cols') else [1])

    def _display_learned_formulas(self, input_cols, output_cols, ml_models):
        """Display the actual learned equations with coefficients"""
        st.subheader("📐 Learned Model Equations")
        st.write("Here are the actual fitted equations with learned parameters:")
        
        formula_cols = st.columns(2)
        col_idx = 0
        
        for model_name, model in ml_models.models.items():
            if model is None:
                continue
                
            with formula_cols[col_idx % 2]:
                st.write(f"**{model_name}:**")
                
                try:
                    if "Linear Regression" in model_name:
                        models, scaler = model
                        for out_idx, output_col in enumerate(output_cols):
                            if out_idx < len(models) and models[out_idx] is not None:
                                lin_model = models[out_idx]
                                coeffs = lin_model.coef_
                                intercept = lin_model.intercept_
                                
                                # Build equation string
                                equation_parts = [f"{intercept:.2f}"]
                                for i, (coeff, input_col) in enumerate(zip(coeffs, input_cols)):
                                    sign = "+" if coeff >= 0 else ""
                                    equation_parts.append(f"{sign}{coeff:.2f}×{input_col}")
                                
                                equation = f"{output_col} = " + "".join(equation_parts)
                                st.code(equation, language=None)
                    
                    elif "Polynomial Regression" in model_name:
                        poly_model = model
                        # Get the polynomial features and linear regression from pipeline
                        if hasattr(poly_model, 'steps'):
                            poly_features = poly_model.steps[0][1]
                            linear_reg = poly_model.steps[1][1]
                            
                            for out_idx, output_col in enumerate(output_cols):
                                if out_idx < linear_reg.coef_.shape[0]:
                                    coeffs = linear_reg.coef_[out_idx] if linear_reg.coef_.ndim > 1 else linear_reg.coef_
                                    intercept = linear_reg.intercept_[out_idx] if hasattr(linear_reg.intercept_, '__len__') else linear_reg.intercept_
                                    
                                    # For polynomial, show simplified form (assuming degree 2)
                                    if len(coeffs) >= 3:  # intercept + linear + quadratic terms
                                        equation = f"{output_col} = {intercept:.2f}"
                                        # Linear terms
                                        for i, input_col in enumerate(input_cols):
                                            if i + 1 < len(coeffs):
                                                coeff = coeffs[i + 1]
                                                sign = "+" if coeff >= 0 else ""
                                                equation += f"{sign}{coeff:.2f}×{input_col}"
                                        
                                        # Quadratic terms (simplified representation)
                                        quad_start = len(input_cols) + 1
                                        if quad_start < len(coeffs):
                                            equation += " + [quadratic terms]"
                                        
                                        st.code(equation, language=None)
                                    else:
                                        st.code(f"{output_col} = [polynomial with {len(coeffs)} terms]", language=None)
                    
                    elif "Exponential Regression" in model_name:
                        models, scaler = model
                        for out_idx, output_col in enumerate(output_cols):
                            if out_idx < len(models) and models[out_idx] is not None:
                                exp_model = models[out_idx]
                                coeffs = exp_model.coef_
                                intercept = exp_model.intercept_
                                
                                # Show both linearized and exponential form
                                if len(coeffs) == 1:
                                    linear_eq = f"ln({output_col}) = {intercept:.2f} + {coeffs[0]:.2f}×{input_cols[0]}"
                                    exp_eq = f"{output_col} = {np.exp(intercept):.2f} × e^({coeffs[0]:.2f}×{input_cols[0]})"
                                    st.code(linear_eq, language=None)
                                    st.code(exp_eq, language=None)
                                else:
                                    # Multiple inputs
                                    linear_parts = [f"{intercept:.2f}"]
                                    exp_parts = []
                                    for i, (coeff, input_col) in enumerate(zip(coeffs, input_cols)):
                                        sign = "+" if coeff >= 0 else ""
                                        linear_parts.append(f"{sign}{coeff:.2f}×{input_col}")
                                        exp_parts.append(f"{coeff:.2f}×{input_col}")
                                    
                                    linear_eq = f"ln({output_col}) = " + "".join(linear_parts)
                                    exp_eq = f"{output_col} = {np.exp(intercept):.2f} × e^({'+'.join(exp_parts)})"
                                    st.code(linear_eq, language=None)
                                    st.code(exp_eq, language=None)
                    
                    elif model_name == "ANN":
                        # For neural networks, show architecture summary
                        if hasattr(model, 'layers'):
                            layer_info = []
                            for i, layer in enumerate(model.layers):
                                if hasattr(layer, 'units'):
                                    layer_info.append(f"Layer {i+1}: {layer.units} units ({layer.activation.__name__})")
                            
                            st.code(f"Neural Network Architecture:\n" + "\n".join(layer_info), language=None)
                            st.code(f"Total Parameters: {model.count_params():,}", language=None)
                    
                    elif model_name == "RNN":
                        # For RNN, show sequence structure
                        if hasattr(model, 'layers'):
                            st.code(f"RNN with {len(model.layers)} layers", language=None)
                            st.code(f"Sequence length: 10 timesteps", language=None)
                            st.code(f"Total Parameters: {model.count_params():,}", language=None)
                    
                    else:
                        st.code(f"Complex model - parameters not displayed", language=None)
                        
                except Exception as e:
                    st.code(f"Could not extract formula: {str(e)}", language=None)
                
                col_idx += 1
                
        st.divider()
    
    def _display_results(self, results, input_cols, output_cols, df_processed, ml_models):
        if results is None or input_cols is None or output_cols is None:
            st.error("No model results available. Please run the models first.")
            return
        
        # Prepare common data
        results_df = pd.DataFrame({
            model: {"R²": float(results[model]["R²"]), "MAPE": float(results[model]["MAPE"])} 
            for model in results.keys()
        }).T
        
        results_df["R²"] = results_df["R²"].astype('float64')
        results_df["MAPE"] = results_df["MAPE"].astype('float64')
        results_df = results_df.sort_values("R²", ascending=False)
        
        best_model = results_df.index[0]
        best_r2 = results_df.loc[best_model, "R²"]
        best_mape = results_df.loc[best_model, "MAPE"]
        
        # Show best model prominently
        st.success(f"🏆 **Best Model: {best_model}** with R² = {best_r2:.2f} and MAPE = {best_mape*100:.2f}%")
        
        # Create tabs for different model result views
        header_col1, header_col2 = st.columns([5,1])
        with header_col1:
            st.subheader("🤖 Model Results & Analysis")
        with header_col2:
            if st.button("🗑️ Clear Results", help="Clear model results and start fresh", key="clear_results_analysis_header"):
                st.session_state.model_results = None
                st.session_state.trained_models = None
                st.session_state.model_input_cols = None
                st.session_state.model_output_cols = None
                st.session_state.processed_data = None
                st.rerun()
        
        calc_tab1, calc_tab2, calc_tab3, calc_tab4, calc_tab5 = st.tabs(["🎯 Make Predictions", "🧪 Variable Impact Test", "⚖️ Compare All Models", "📊 Model Comparison", "📐 Model Equations"])
            
        with calc_tab1:
            st.write("Enter input values to get predictions from your trained models:")
                
            with st.form("prediction_form"):
                st.write("**Input Values:**")
                    
                input_values = {}
                cols_per_row = 3
                input_rows = [input_cols[i:i + cols_per_row] for i in range(0, len(input_cols), cols_per_row)]
                
                for row in input_rows:
                    cols = st.columns(len(row))
                    for i, col_name in enumerate(row):
                        with cols[i]:
                            col_data = df_processed[col_name]
                            min_val = float(col_data.min())
                            max_val = float(col_data.max())
                            mean_val = float(col_data.mean())
                            
                            display_name = str(col_name)
                            input_values[col_name] = st.number_input(
                                f"{display_name}",
                                min_value=min_val * 0.5,
                                max_value=max_val * 1.5,
                                value=mean_val,
                                step=(max_val - min_val) / 100,
                                format="%.2f"
                            )
                
                st.write("**Model Selection:**")
                model_options = list(results.keys())
                default_model_idx = model_options.index(best_model)
                
                selected_model = st.selectbox(
                    "Choose model for prediction:",
                    options=model_options,
                    index=default_model_idx,
                    help=f"Best model ({best_model}) is selected by default"
                )
                model_info = {
                    "Linear Regression": "Direct mathematical relationship",
                    "Linear Regression (CV)": "Direct mathematical relationship with cross-validation",
                    "Polynomial Regression": "Non-linear relationships with polynomial features",
                    "Polynomial Regression (CV)": "Non-linear relationships with polynomial features and cross-validation",
                    "Exponential Regression": "Log-linear transformation for exponential relationships",
                    "Exponential Regression (CV)": "Log-linear transformation for exponential relationships with cross-validation",
                    "ANN": "Deep learning with batch normalization (TF Nightly optimized)",
                    "RNN": "Sequence-based predictions with bidirectional GRU (TF Nightly optimized)"
                }
                
                if selected_model in model_info:
                    st.info(f"**{selected_model}**: {model_info[selected_model]}")
                
                selected_r2 = results[selected_model]["R²"]
                selected_mape = results[selected_model]["MAPE"] * 100  # percent
                st.write(f"**Selected Model Performance**: R² = {selected_r2:.2f}, MAPE = {selected_mape:.2f}%")
                
                predict_button = st.form_submit_button("🚀 Predict", type="primary")
                
                if predict_button:
                    try:
                        input_array = np.array([[input_values[col] for col in input_cols]])
                        if selected_model in ml_models.models and ml_models.models[selected_model] is not None:
                            model = ml_models.models[selected_model]
                            
                            if selected_model == "RNN":
                                if selected_model in ml_models.model_scalers:
                                    rnn_scalers = ml_models.model_scalers[selected_model]
                                    time_steps = 10
                                    input_scaled = rnn_scalers['X_scaler'].transform(input_array)
                                    
                                    input_sequence = np.repeat(input_scaled, time_steps, axis=0).reshape(1, time_steps, -1)
                                    
                                    pred_scaled = model.predict(input_sequence, verbose=0)
                                    pred_log = rnn_scalers['y_scaler'].inverse_transform(pred_scaled)
                                    predictions = np.exp(pred_log)[0]
                                else:
                                    time_steps = 10
                                    input_scaled = ml_models.X_scaler.transform(input_array)
                                    input_sequence = np.repeat(input_scaled, time_steps, axis=0).reshape(1, time_steps, -1)
                                    pred_scaled = model.predict(input_sequence, verbose=0)
                                    pred_log = ml_models.y_scaler.inverse_transform(pred_scaled)
                                    predictions = np.exp(pred_log)[0]
                                
                            elif selected_model == "ANN":
                                if selected_model in ml_models.model_scalers:
                                    ann_scalers = ml_models.model_scalers[selected_model]
                                    input_scaled = ann_scalers['X_scaler'].transform(input_array)
                                    pred_scaled = model.predict(input_scaled, verbose=0)
                                    predictions = ann_scalers['y_scaler'].inverse_transform(pred_scaled)[0]
                                else:
                                    input_scaled = ml_models.X_scaler.transform(input_array)
                                    pred_scaled = model.predict(input_scaled, verbose=0)
                                    predictions = pred_scaled[0]
                                
                            elif selected_model == "Linear Regression" or selected_model == "Linear Regression (CV)":
                                models, scaler = model
                                input_scaled = scaler.transform(input_array)
                                predictions = []
                                for i, lin_model in enumerate(models):
                                    pred = lin_model.predict(input_scaled)[0]
                                    predictions.append(pred)
                                
                            elif selected_model == "Polynomial Regression" or selected_model == "Polynomial Regression (CV)":
                                predictions = model.predict(input_array)[0]
                                
                            elif selected_model == "Exponential Regression" or selected_model == "Exponential Regression (CV)":
                                models, scaler = model
                                input_scaled = scaler.transform(input_array)
                                predictions = []
                                for i, exp_model in enumerate(models):
                                    if exp_model is None:
                                        predictions.append(0.0)
                                    else:
                                        try:
                                            log_pred = exp_model.predict(input_scaled)[0]
                                            pred = np.exp(log_pred)
                                            predictions.append(pred)
                                        except:
                                            predictions.append(0.0)
                                
                            else: 
                                predictions = [0.0] * len(output_cols)
                                                 
                                st.success(f"✅ Predictions using **{selected_model}**:")
                            
                            pred_cols = st.columns(len(output_cols))
                            for i, (col_name, pred_value) in enumerate(zip(output_cols, predictions)):
                                with pred_cols[i]:
                                    format_str = self._get_format_for_column(col_name)
                                    st.metric(
                                        label=f"**{col_name}**",
                                        value=f"{float(pred_value):{format_str}}",
                                        help=f"Predicted value for {col_name}"
                                    )
                            
                            with st.expander("📋 Input Summary"):
                                input_df = pd.DataFrame([input_values])
                                st.dataframe(input_df)
                                
                            model_r2 = results[selected_model]["R²"]
                            model_mape = results[selected_model]["MAPE"] * 100  # percent
                            confidence_color = "green" if model_r2 > 0.9 else "orange" if model_r2 > 0.7 else "red"
                            st.markdown(f"""
                            **Model Performance:**
                            - R² Score: ::{confidence_color}[{model_r2:.2f}]
                            - MAPE: {model_mape:.2f}%
                            """)
                            
                        else:
                            st.error(f"Model {selected_model} not available for prediction. Please retrain the models.")
                            
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
                        st.write("Please check your input values and try again.")
                    
            with st.expander("💡 Example Input Values"):
                st.write("Here are some example values from your dataset:")
                example_data = df_processed[input_cols].sample(n=min(5, len(df_processed)), random_state=42)
                
                display_example = example_data.copy()
                for col in display_example.columns:
                    display_example[col] = display_example[col].astype('float64')
                display_example = display_example.reset_index(drop=True)
                
                st.dataframe(display_example.round(2))
                st.write("💡 **Tip**: You can copy these values to test the prediction functionality.")
                
                if st.button("🔄 Load Random Example", help="Load a random example into the input fields above"):
                    st.rerun()
        
        with calc_tab2:
            st.write("Test how changing one variable affects your predictions:")
            
            # Move variable selection outside the form for dynamic updates
            variable_to_change = st.selectbox(
                "Choose variable to modify:",
                options=input_cols,
                key="variable_impact_variable",
                help="This variable will be tested with a different value"
            )
            
            with st.form("variable_impact_form"):
                st.write("**🎯 Baseline Input Values:**")
                
                baseline_values = {}
                cols_per_row = 3
                input_rows = [input_cols[i:i + cols_per_row] for i in range(0, len(input_cols), cols_per_row)]
                
                for row in input_rows:
                    cols = st.columns(len(row))
                    for i, col_name in enumerate(row):
                        with cols[i]:
                            col_data = df_processed[col_name]
                            min_val = float(col_data.min())
                            max_val = float(col_data.max())
                            mean_val = float(col_data.mean())
                            
                            display_name = str(col_name)
                            baseline_values[col_name] = st.number_input(
                                f"{display_name} (baseline)",
                                min_value=min_val * 0.5,
                                max_value=max_val * 1.5,
                                value=mean_val,
                                step=(max_val - min_val) / 100,
                                format="%.2f",
                                key=f"baseline_{col_name}"
                            )
                
                st.divider()
                # Show selected variable and its info at the top
                col_data = df_processed[variable_to_change]
                min_val = float(col_data.min())
                max_val = float(col_data.max())
                mean_val = float(col_data.mean())
                baseline_val = baseline_values[variable_to_change]
                dtype_val = col_data.dtype
                
                st.write(f"**🎛️ Alternative Value for {variable_to_change}:**")
                
                modified_value = st.number_input(
                    f"New value for {variable_to_change}:",
                    min_value=min_val * 0.5,
                    max_value=max_val * 1.5,
                    value=baseline_val * 1.1,  # Default to 10% increase
                    step=(max_val - min_val) / 100,
                    format="%.2f",
                    key=f"modified_{variable_to_change}",
                    help=f"This will replace the baseline value of {baseline_val:.2f}"
                )
                
                st.write("**📊 Comparison Summary:**")
                comparison_col1, comparison_col2, comparison_col3 = st.columns(3)
                
                with comparison_col1:
                    st.metric("Variable", variable_to_change)
                with comparison_col2:
                    st.metric("Baseline Value", f"{baseline_val:.2f}")
                with comparison_col3:
                    st.metric("Modified Value", f"{modified_value:.2f}", 
                             delta=f"{modified_value - baseline_val:+.2f}")
                
                st.write("**🤖 Model Selection:**")
                model_options = list(results.keys())
                results_df = pd.DataFrame({
                    model: {"R²": float(results[model]["R²"]), "MAPE": float(results[model]["MAPE"])} 
                    for model in results.keys()
                }).T.sort_values("R²", ascending=False)
                best_model = results_df.index[0]
                default_model_idx = model_options.index(best_model)
                
                selected_model = st.selectbox(
                    "Choose model for variable impact test:",
                    options=model_options,
                    index=default_model_idx,
                    help=f"Best model ({best_model}) is selected by default",
                    key="variable_impact_model"
                )
                
                test_button = st.form_submit_button("🧪 Run Variable Impact Test", type="primary")
                
                if test_button:
                    try:
                        # Prepare baseline input array
                        baseline_array = np.array([[baseline_values[col] for col in input_cols]])
                        
                        # Prepare modified input array
                        modified_values = baseline_values.copy()
                        modified_values[variable_to_change] = modified_value
                        modified_array = np.array([[modified_values[col] for col in input_cols]])
                        
                        if selected_model in ml_models.models and ml_models.models[selected_model] is not None:
                            # Get baseline predictions
                            baseline_predictions = self._get_model_prediction(selected_model, baseline_array, ml_models)
                            
                            # Get modified predictions
                            modified_predictions = self._get_model_prediction(selected_model, modified_array, ml_models)
                            
                            # Calculate deltas
                            deltas = [mod - base for mod, base in zip(modified_predictions, baseline_predictions)]
                            percentage_changes = []
                            for i, (base, delta) in enumerate(zip(baseline_predictions, deltas)):
                                if base != 0:
                                    pct = (delta / base) * 100
                                else:
                                    pct = 0.0
                                percentage_changes.append(pct)
                            
                            st.success(f"✅ Variable Impact Test Complete using **{selected_model}**")
                            
                            # Display baseline results
                            st.write("**📊 Baseline Results:**")
                            baseline_cols = st.columns(len(output_cols))
                            for i, (col_name, pred_value) in enumerate(zip(output_cols, baseline_predictions)):
                                with baseline_cols[i]:
                                    format_str = self._get_format_for_column(col_name)
                                    st.metric(
                                        label=f"**{col_name}**",
                                        value=f"{float(pred_value):{format_str}}",
                                        help="Prediction with original values"
                                    )
                            
                            # Display modified results
                            st.write("**🔄 Modified Results:**")
                            modified_cols = st.columns(len(output_cols))
                            for i, (col_name, pred_value) in enumerate(zip(output_cols, modified_predictions)):
                                with modified_cols[i]:
                                    format_str = self._get_format_for_column(col_name)
                                    st.metric(
                                        label=f"**{col_name}**",
                                        value=f"{float(pred_value):{format_str}}",
                                        help=f"Prediction with {variable_to_change} = {modified_value:.2f}"
                                    )
                            
                            # Display delta results
                            st.write("**📈 Delta Results:**")
                            delta_cols = st.columns(len(output_cols))
                            for i, (col_name, delta_value) in enumerate(zip(output_cols, deltas)):
                                with delta_cols[i]:
                                    delta_color = "normal"
                                    if abs(delta_value) > abs(baseline_predictions[i]) * 0.05:  # >5% change
                                        delta_color = "inverse"
                                    
                                    format_str = self._get_format_for_column(col_name)
                                    st.metric(
                                        label=f"**{col_name}**",
                                        value=f"{delta_value:+{format_str[1:]}}",
                                        delta=f"{percentage_changes[i]:+.2f}%",
                                        delta_color=delta_color,
                                        help="Difference (Modified - Baseline)"
                                    )
                            
                            # Summary table
                            st.write("**📋 Summary Table:**")
                            summary_data = {
                                'Output': output_cols,
                                'Baseline': [f"{pred:{self._get_format_for_column(col)}}" for pred, col in zip(baseline_predictions, output_cols)],
                                'Modified': [f"{pred:{self._get_format_for_column(col)}}" for pred, col in zip(modified_predictions, output_cols)],
                                'Delta': [f"{delta:+{self._get_format_for_column(col)[1:]}}" for delta, col in zip(deltas, output_cols)],
                                'Change %': [f"{pct:+.2f}%" for pct in percentage_changes]
                            }
                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(summary_df, use_container_width=True)
                            
                            # Visualization
                            st.write("**📈 Visual Comparison:**")
                            
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                            
                            # Bar chart comparing baseline vs modified
                            x_pos = np.arange(len(output_cols))
                            width = 0.35
                            
                            ax1.bar(x_pos - width/2, baseline_predictions, width, label='Baseline', alpha=0.8, color='skyblue')
                            ax1.bar(x_pos + width/2, modified_predictions, width, label='Modified', alpha=0.8, color='lightcoral')
                            
                            ax1.set_xlabel('Output Variables')
                            ax1.set_ylabel('Predicted Values')
                            ax1.set_title('Baseline vs Modified Predictions')
                            ax1.set_xticks(x_pos)
                            ax1.set_xticklabels(output_cols, rotation=45)
                            ax1.legend()
                            ax1.grid(True, alpha=0.3)
                            
                            # Delta chart
                            colors = ['green' if d >= 0 else 'red' for d in deltas]
                            bars = ax2.bar(output_cols, deltas, color=colors, alpha=0.7)
                            ax2.set_xlabel('Output Variables')
                            ax2.set_ylabel('Delta (Modified - Baseline)')
                            ax2.set_title(f'Impact of Changing {variable_to_change}')
                            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                            ax2.grid(True, alpha=0.3)
                            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
                            
                            # Add value labels on bars
                            for bar, delta, pct in zip(bars, deltas, percentage_changes):
                                height = bar.get_height()
                                ax2.text(bar.get_x() + bar.get_width()/2., height + (max(deltas) - min(deltas))*0.01,
                                        f'{delta:+.2f}\n({pct:+.1f}%)', ha='center', va='bottom', fontsize=9)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                        else:
                            st.error(f"Model {selected_model} not available. Please retrain the models.")
                            
                    except Exception as e:
                        st.error(f"Error in variable impact test: {str(e)}")
                        st.write("Please check your input values and model selection.")
        
        with calc_tab3:
            st.write("Get predictions from all trained models at once:")
            
            with st.form("compare_models_form"):
                st.write("**Quick Input for Model Comparison:**")
                comparison_cols = st.columns(min(3, len(input_cols)))
                comparison_values = {}
                
                for i, col_name in enumerate(input_cols):
                    col_idx = i % len(comparison_cols)
                    with comparison_cols[col_idx]:
                        col_data = df_processed[col_name]
                        mean_val = float(col_data.mean())
                        display_name = str(col_name)
                        comparison_values[col_name] = st.number_input(
                            f"{display_name}",
                            value=mean_val,
                            key=f"compare_{col_name}",
                            format="%.2f"
                        )
                
                compare_button = st.form_submit_button("🔍 Compare All Models", type="secondary")
                
                if compare_button:
                    st.write("**Predictions from all models:**")
                    
                    comparison_input = np.array([[comparison_values[col] for col in input_cols]])
                    comparison_results = {}
                    
                    for model_name in results.keys():
                        try:
                            if model_name in ml_models.models and ml_models.models[model_name] is not None:
                                model = ml_models.models[model_name]
                                
                                if model_name == "RNN" and model_name in ml_models.model_scalers:
                                    rnn_scalers = ml_models.model_scalers[model_name]
                                    input_scaled = rnn_scalers['X_scaler'].transform(comparison_input)
                                    input_sequence = np.repeat(input_scaled, 10, axis=0).reshape(1, 10, -1)
                                    pred_scaled = model.predict(input_sequence, verbose=0)
                                    pred_log = rnn_scalers['y_scaler'].inverse_transform(pred_scaled)
                                    preds = np.exp(pred_log)[0]
                                elif model_name == "ANN" and model_name in ml_models.model_scalers:
                                    ann_scalers = ml_models.model_scalers[model_name]
                                    input_scaled = ann_scalers['X_scaler'].transform(comparison_input)
                                    pred_scaled = model.predict(input_scaled, verbose=0)
                                    preds = ann_scalers['y_scaler'].inverse_transform(pred_scaled)[0]
                                elif model_name == "Linear Regression" or model_name == "Linear Regression (CV)":
                                    models, scaler = model
                                    comparison_scaled = scaler.transform(comparison_input)
                                    preds = []
                                    for lin_model in models:
                                        pred = lin_model.predict(comparison_scaled)[0]
                                        preds.append(pred)
                                elif model_name == "Polynomial Regression" or model_name == "Polynomial Regression (CV)":
                                    preds = model.predict(comparison_input)[0]
                                elif model_name == "Exponential Regression" or model_name == "Exponential Regression (CV)":
                                    models, scaler = model
                                    comparison_scaled = scaler.transform(comparison_input)
                                    preds = []
                                    for exp_model in models:
                                        if exp_model is None:
                                            preds.append(0.0)
                                        else:
                                            try:
                                                log_pred = exp_model.predict(comparison_scaled)[0] 
                                                pred = np.exp(log_pred)
                                                preds.append(pred)
                                            except:
                                                preds.append(0.0)
                                else:
                                    preds = [0.0] * len(output_cols)
                                
                                comparison_results[model_name] = preds
                            else:
                                comparison_results[model_name] = ["N/A"] * len(output_cols)
                        except Exception as e:
                            comparison_results[model_name] = ["Error"] * len(output_cols)
                    
                    comparison_df = pd.DataFrame(comparison_results, index=output_cols).T
                    comparison_df = comparison_df.astype(str)
                    
                    for col in comparison_df.columns:
                        try:
                            comparison_df[col] = pd.to_numeric(comparison_df[col], errors='coerce')
                            # Apply uniform 2 decimal rounding
                            comparison_df[col] = comparison_df[col].round(2)
                        except:
                            pass
                        
                    st.dataframe(comparison_df)
        
        with calc_tab4:
            st.write("**Performance Comparison Table:**")
            try:
                styled_df = results_df.style.highlight_max(subset=["R²"]).highlight_min(subset=["MAPE"])
                st.dataframe(styled_df.format({"R²": "{:.2f}", "MAPE": lambda x: f"{x*100:.2f}%"}))
            except Exception as e:
                try:
                    clean_results = results_df.copy().reset_index()
                    for col in clean_results.columns:
                        if clean_results[col].dtype == 'object':
                            clean_results[col] = clean_results[col].astype(str)
                        elif np.issubdtype(clean_results[col].dtype, np.number):
                            clean_results[col] = clean_results[col].astype('float64')
                    st.dataframe(clean_results.style.format({"R²": "{:.2f}", "MAPE": lambda x: f"{x*100:.2f}%"}))
                except Exception as e2:
                    st.error(f"Could not display results table: {str(e2)}")
                    st.write("Results:", results_df.to_dict())
            
            st.write("**Performance Charts:**")
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                models = list(results_df.index)
                r2_scores = results_df["R²"].values
                colors = ['gold' if model == best_model else 'skyblue' for model in models]
                bars = ax.bar(models, r2_scores, color=colors)
                ax.set_ylabel('R² Score')
                ax.set_title('Model Performance (R² Score)')
                ax.set_ylim(0, 1)
                plt.xticks(rotation=45)
                
                for bar, score in zip(bars, r2_scores):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{score:.2f}', ha='center', va='bottom')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                mape_scores = results_df["MAPE"].values * 100  # Convert to percent for chart
                colors = ['gold' if model == best_model else 'lightcoral' for model in models]
                bars = ax.bar(models, mape_scores, color=colors)
                ax.set_ylabel('MAPE Score')
                ax.set_title('Model Performance (MAPE Score - Lower is Better)')
                plt.xticks(rotation=45)
                
                for bar, score in zip(bars, mape_scores):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mape_scores)*0.01,
                           f'{score:.2f}%', ha='center', va='bottom')
                
                plt.tight_layout()
                st.pyplot(fig)
        
        with calc_tab5:
            # Display learned formulas
            self._display_learned_formulas(input_cols, output_cols, ml_models)

st.title("🤖 ML Regression Dashboard - TF Nightly Edition")
st.write("Upload an Excel file to run regression models (Linear, Polynomial, Exponential + CV versions, ANN, RNN) and view results.")

st.sidebar.header("📊 Data Setup")

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload your dataset (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    st.sidebar.success("✅ File uploaded successfully!")
    
    # Show basic file info
    st.sidebar.write(f"**File name:** {uploaded_file.name}")
    file_size = len(uploaded_file.getvalue()) / 1024
    st.sidebar.write(f"**File size:** {file_size:.1f} KB")
    
else:
    st.sidebar.info("👆 Please upload an Excel file to get started!")

# System Information at bottom of sidebar
st.sidebar.divider()
with st.sidebar.expander("🔧 System Information"):
    st.write(f"**TensorFlow:** {tf.__version__}")
    gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
    st.write(f"**GPU:** {'✅ Yes' if gpu_available else '❌ No'}")
    st.write(f"**XLA JIT:** {'✅ Enabled' if tf.config.optimizer.get_jit() else '❌ Disabled'}")


if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        
        ml_models = MLModels()
        input_cols, output_cols = ml_models.identify_columns(df)
        df_processed = ml_models.preprocess_data(df)
        
        # Add debug information to sidebar
        st.sidebar.divider()
        with st.sidebar.expander("🔍 Debug Information"):
            st.write(f"**Processed data shape:** {df_processed.shape}")
            st.write(f"**Raw column names from file:** {list(df.columns)}")
            st.write(f"**Processed column names:** {list(df_processed.columns)}")
            st.write(f"**All column types:**")
            for col, dtype in df_processed.dtypes.items():
                st.write(f"  - {col}: {dtype}")
            st.write(f"**Numeric columns found:** {list(df_processed.select_dtypes(include=[np.number]).columns)}")
            st.write(f"**Any NaN values:** {df_processed.isnull().sum().sum()}")
            st.write(f"**Index type:** {type(df_processed.index)}")
            st.write("**First row of processed data:**")
            if not df_processed.empty:
                st.dataframe(df_processed.head(1).round(2))
        
        # Quick status (always visible)
        if input_cols and output_cols:
            st.success(f"✅ Ready to train! {len(input_cols)} input(s) → {len(output_cols)} output(s)")
        else:
            st.warning("⚠️ Please select both input and output columns before training models")
        
        # Column Selection (Minimizable) - 1/2 width
        
        with st.expander("🎯 Column Selection", expanded=False):
            st.write("**Input Columns:**")
            if input_cols:
                st.success(f"✅ {len(input_cols)} input columns")
            else:
                st.warning("⚠️ No input columns")
            
            available_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
            selected_inputs = st.multiselect("Select Input Columns", available_cols, default=input_cols)
            
            st.write("")  # Spacing
            
            st.write("**Output Columns:**")  
            if output_cols:
                st.success(f"✅ {len(output_cols)} output columns")
            else:
                st.warning("⚠️ No output columns")
           
            selected_outputs = st.multiselect("Select Output Columns", available_cols, default=output_cols)

            if selected_inputs:
                input_cols = selected_inputs
            if selected_outputs:
                output_cols = selected_outputs
        
        st.divider()
        st.subheader("📊 Data Analysis & Information")
        
        data_tab1, data_tab2, data_tab3, data_tab4 = st.tabs(["📈 Data Relationships", "📊 Data Statistics" , "📋 Raw Data", "🔍 Column Details"])
        
        with data_tab1:
            if input_cols and output_cols:
                try:
                    analysis_cols = input_cols + output_cols
                    if analysis_cols:
                        # Side by side layout
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            if len(input_cols) > 0 and len(output_cols) > 0:
                                st.write("**Input-Output Correlation Matrix:**")
                                
                                # Create cross-correlation matrix (inputs vs outputs only)
                                input_data = df_processed[input_cols]
                                output_data = df_processed[output_cols]
                                
                                # Calculate correlations between inputs and outputs
                                cross_corr = input_data.corrwith(output_data.iloc[:, 0] if len(output_cols) == 1 
                                                               else output_data, axis=0)
                                
                                if len(output_cols) == 1:
                                    # Single output - show as vertical heatmap
                                    cross_corr_df = pd.DataFrame({output_cols[0]: cross_corr})
                                    fig, ax = plt.subplots(figsize=(4, max(6, len(input_cols) * 0.4)))
                                    sns.heatmap(cross_corr_df, annot=True, cmap='coolwarm', center=0, 
                                              ax=ax, cbar_kws={'label': 'Correlation'})
                                    ax.set_title(f'Input Features → {output_cols[0]} Correlations')
                                else:
                                    # Multiple outputs - show full cross-correlation
                                    cross_corr_matrix = pd.DataFrame(index=input_cols, columns=output_cols)
                                    for input_col in input_cols:
                                        for output_col in output_cols:
                                            cross_corr_matrix.loc[input_col, output_col] = df_processed[input_col].corr(df_processed[output_col])
                                    
                                    cross_corr_matrix = cross_corr_matrix.astype(float)
                                    fig, ax = plt.subplots(figsize=(max(6, len(output_cols) * 1.2), max(6, len(input_cols) * 0.4)))
                                    sns.heatmap(cross_corr_matrix, annot=True, cmap='coolwarm', center=0, 
                                              ax=ax, cbar_kws={'label': 'Correlation'})
                                    ax.set_title('Input Features → Output Targets Correlations')
                                    ax.set_xlabel('Output Targets')
                                    ax.set_ylabel('Input Features')
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close(fig)
                        
                        with col2:
                            # Input-Output Relationships first
                            st.write("**Input-Output Relationships:**")
                            fig, axes = plt.subplots(len(output_cols), len(input_cols), figsize=(12, 8))
                            plt.subplots_adjust(hspace=0.5)
                            
                            if len(output_cols) == 1 and len(input_cols) == 1:
                                axes.scatter(df_processed[input_cols[0]], df_processed[output_cols[0]], alpha=0.5, s=10)
                                axes.set_xlabel(input_cols[0])
                                axes.set_ylabel(output_cols[0])
                                axes.set_title(f"{output_cols[0]} vs {input_cols[0]}")
                            elif len(output_cols) == 1:
                                for j, in_col in enumerate(input_cols):
                                    axes[j].scatter(df_processed[in_col], df_processed[output_cols[0]], alpha=0.5, s=10)
                                    axes[j].set_xlabel(in_col)
                                    axes[j].set_ylabel(output_cols[0])
                                    axes[j].set_title(f"{output_cols[0]} vs {in_col}")
                            elif len(input_cols) == 1:
                                for i, out_col in enumerate(output_cols):
                                    axes[i].scatter(df_processed[input_cols[0]], df_processed[out_col], alpha=0.5, s=10)
                                    axes[i].set_xlabel(input_cols[0])
                                    axes[i].set_ylabel(out_col)
                                    axes[i].set_title(f"{out_col} vs {input_cols[0]}")
                            else:
                                for i, out_col in enumerate(output_cols):
                                    for j, in_col in enumerate(input_cols):
                                        axes[i, j].scatter(df_processed[in_col], df_processed[out_col], alpha=0.5, s=10)
                                        axes[i, j].set_xlabel(in_col)
                                        axes[i, j].set_ylabel(out_col)
                                        axes[i, j].set_title(f"{out_col} vs {in_col}")
                            
                            st.pyplot(fig)
                            plt.close(fig)
                            
                    else:
                        st.warning("No columns available for analysis.")
                except Exception as e:
                    st.error(f"Error in data analysis: {str(e)}")
            else:
                st.info("Select input and output columns to see statistics and correlation.")

        with st.expander("🔧 Model Management", expanded=False):
            # Create tabs for different model operations
            tab1, tab2, tab3 = st.tabs(["🚀 Train New Models", "💾 Save/Load Models", "📁 Manage Saved Models"])
            
            with tab1:
                # Auto-save option
                auto_save = st.checkbox("💾 Auto-save models after training", value=True, help="Automatically save models for future use")
                
                if st.button("🚀 Run All Models", type="primary"):
                    if len(input_cols) > 0 and len(output_cols) > 0:
                        try:
                            X = df_processed[input_cols].values
                            y = df_processed[output_cols].values
                            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y).any(axis=1))
                            X = X[mask]
                            y = y[mask]
                            if X.shape[0] < 10:
                                st.error("Not enough data points after cleaning. Need at least 10 rows.")
                            else:
                                st.write("🔄 Running all models... It might take a while!")
                                results = ml_models.run_all_models(X, y, input_cols, output_cols)
                                st.session_state.model_results = results
                                st.session_state.trained_models = ml_models
                                st.session_state.model_input_cols = input_cols
                                st.session_state.model_output_cols = output_cols
                                st.session_state.processed_data = df_processed
                                
                                # Auto-save if enabled
                                if auto_save:
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    auto_save_name = f"auto_save_{timestamp}"
                                    saved_name = ml_models.save_models(auto_save_name)
                                    if saved_name:
                                        st.info(f"🔄 Models auto-saved as: {saved_name}")
                                
                                st.success("✅ Models trained successfully! Results are now persistent across page interactions.")
                                st.balloons()
                        except Exception as e:
                            st.error(f"Error running models: {str(e)}")
                            st.write("Please check your data format and try again.")
                    else:
                        st.warning("Please select at least one input and one output column.")
            
            with tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**💾 Save Current Models**")
                    if st.session_state.get('trained_models') is not None:
                        custom_name = st.text_input("Model Name (optional)", help="Leave empty for auto-generated timestamp name")
                        if st.button("💾 Save Models", type="secondary"):
                            saved_name = st.session_state.trained_models.save_models(custom_name if custom_name else None)
                            if saved_name:
                                st.success(f"✅ Models saved as: {saved_name}")
                            else:
                                st.error("❌ Failed to save models")
                    else:
                        st.info("ℹ️ Train models first before saving")
                
                with col2:
                    st.write("**📂 Load Saved Models**")
                    saved_models = ml_models.get_saved_models_list()
                    if saved_models:
                        selected_model = st.selectbox("Choose saved model:", saved_models)
                        if st.button("📂 Load Models", type="secondary"):
                            temp_ml_models = MLModels()
                            if temp_ml_models.load_models(selected_model):
                                st.session_state.model_results = temp_ml_models.results
                                st.session_state.trained_models = temp_ml_models
                                st.session_state.model_input_cols = temp_ml_models.input_cols
                                st.session_state.model_output_cols = temp_ml_models.output_cols
                                st.session_state.processed_data = df_processed  # Keep current data
                                st.success(f"✅ Models loaded from: {selected_model}")
                                st.rerun()
                            else:
                                st.error("❌ Failed to load models")
                    else:
                        st.info("ℹ️ No saved models found")
                
                with tab3:
                    st.write("**🗂️ Saved Models**")
                    saved_models = ml_models.get_saved_models_list()
                    if saved_models:
                        for model_name in saved_models:
                            with st.expander(f"📁 {model_name}", expanded=False):
                                # Buttons at the top in equal columns
                                col1, col2 = st.columns([1, 1])
                                with col1:
                                    if st.button("📂 Load", key=f"load_{model_name}", help=f"Load {model_name}"):
                                        temp_ml_models = MLModels()
                                        if temp_ml_models.load_models(model_name):
                                            st.session_state.model_results = temp_ml_models.results
                                            st.session_state.trained_models = temp_ml_models
                                            st.session_state.model_input_cols = temp_ml_models.input_cols
                                            st.session_state.model_output_cols = temp_ml_models.output_cols
                                            st.session_state.processed_data = df_processed
                                            st.success(f"✅ Loaded: {model_name}")
                                            st.rerun()
                                with col2:
                                    if st.button("🗑️ Delete", key=f"delete_{model_name}", help=f"Delete {model_name}"):
                                        if ml_models.delete_saved_model(model_name):
                                            st.success(f"✅ Deleted: {model_name}")
                                            st.rerun()
                                        else:
                                            st.error(f"❌ Failed to delete: {model_name}")
                                
                                # Model metadata info below
                                try:
                                    temp_ml = MLModels()
                                    save_data = joblib.load(f'saved_models/{model_name}.pkl')
                                    metadata = save_data.get('model_metadata', {})
                                    
                                    if metadata:
                                        st.write(f"**Training Date:** {metadata.get('training_timestamp', 'Unknown')[:19]}")
                                        st.write(f"**Input Features:** {metadata.get('n_input_features', 'Unknown')}")
                                        st.write(f"**Output Features:** {metadata.get('n_output_features', 'Unknown')}")
                                        st.write(f"**Data Shape:** {metadata.get('data_shape', 'Unknown')}")
                                        
                                        if 'input_cols' in save_data and save_data['input_cols']:
                                            st.write(f"**Input Columns:** {', '.join(save_data['input_cols'])}")
                                        if 'output_cols' in save_data and save_data['output_cols']:
                                            st.write(f"**Output Columns:** {', '.join(save_data['output_cols'])}")
                                    
                                    # Show model results if available
                                    if 'results' in save_data and save_data['results']:
                                        st.write("**Model Performance:**")
                                        results_df = pd.DataFrame({
                                            model: {"R²": float(save_data['results'][model]["R²"]), "MAPE": float(save_data['results'][model]["MAPE"])} 
                                            for model in save_data['results'].keys()
                                        }).T.sort_values("R²", ascending=False)
                                        st.dataframe(results_df.head(3))  # Show top 3 models
                                except:
                                    st.write("*Model info unavailable*")
                    else:
                        st.info("ℹ️ No saved models found")
        
        with data_tab2:
            st.write("**Data Statistics:**")
            st.dataframe(df_processed[analysis_cols].describe().round(2))

        with data_tab3:
            st.write("**Original data preview:**")
            st.dataframe(df.head())
            st.write("**After preprocessing:**")
            st.dataframe(df_processed.head().round(2))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", df_processed.shape[0])
            with col2:
                st.metric("Columns", df_processed.shape[1])
            with col3:
                st.metric("Missing Values", df_processed.isnull().sum().sum())
        
        with data_tab4:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Input Columns:**")
                if input_cols:
                    st.success(f"Found {len(input_cols)} input columns:")
                    for col in input_cols:
                        st.write(f"- {col}")
                else:
                    st.error("No input columns identified!")
            with col2:
                st.write("**Output Columns:**")  
                if output_cols:
                    st.success(f"Found {len(output_cols)} output columns:")
                    for col in output_cols:
                        st.write(f"- {col}")
                else:
                    st.error("No output columns identified!")
        
        def make_arrow_compatible(dataframe):
            df_clean = dataframe.copy()
            for col in df_clean.columns:
                try:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce', downcast=None)
                    df_clean[col] = df_clean[col].astype('float64')
                except:
                    df_clean = df_clean.drop(columns=[col])
            
            df_clean = df_clean.dropna()
            df_clean = df_clean.reset_index(drop=True)
            
            for col in df_clean.columns:
                if df_clean[col].dtype not in ['float64', 'int64', 'float32', 'int32']:
                    df_clean[col] = df_clean[col].astype('float64')
            
            return df_clean
        

        
        df_processed = make_arrow_compatible(df_processed)
        
        def safe_display_dataframe(dataframe, title):
            st.write(f"**{title}:**")
            try:
                display_df = dataframe.head().copy()
                for col in display_df.columns:
                    display_df[col] = pd.to_numeric(display_df[col], errors='coerce').astype('float64')

                display_df = display_df.dropna()
                display_df = display_df.reset_index(drop=True)
                st.dataframe(display_df)
                
            except Exception as e:
                st.error(f"Could not display {title.lower()}: {str(e)}")
                st.write(f"Data shape: {dataframe.shape}")
                if hasattr(dataframe, 'dtypes'):
                    st.write(f"Column types: {dataframe.dtypes.to_dict()}")
        

        if 'model_results' not in st.session_state:
            st.session_state.model_results = None
        if 'trained_models' not in st.session_state:
            st.session_state.trained_models = None
        if 'model_input_cols' not in st.session_state:
            st.session_state.model_input_cols = None
        if 'model_output_cols' not in st.session_state:
            st.session_state.model_output_cols = None
        if 'processed_data' not in st.session_state:
            st.session_state.processed_data = None

        
        if st.session_state.model_results is not None:            
            display_ml_models = MLModels()
            display_ml_models.models = st.session_state.trained_models.models
            display_ml_models.model_scalers = st.session_state.trained_models.model_scalers
            display_ml_models.X_scaler = st.session_state.trained_models.X_scaler
            display_ml_models.y_scaler = st.session_state.trained_models.y_scaler
            
            display_ml_models._display_results(
                st.session_state.model_results, 
                st.session_state.model_input_cols, 
                st.session_state.model_output_cols, 
                st.session_state.processed_data, 
                display_ml_models
            )

    except Exception as e:
        st.error(f"Error loading file: {str(e)}")

else:
    # Instructions for new UI flow
    st.subheader("🚀 How It Works")
    st.write("""
    **New Streamlined Workflow:**
    1. **Upload** your Excel dataset in the sidebar 📊
    2. **Select** input and output columns in the sidebar 🎯
    3. **Train** or **Load** your models instantly 🤖
    4. **View** data analysis in organized tabs 📈
    5. **Make** predictions with your best model 🔮
    """)
    
    st.subheader("📋 Example Data Format")
    example_data = pd.DataFrame({
        'Car_Speed': [55, 60, 62, 58, 65, 61],
        'Engine_Temp': [90.5, 91.2, 92.1, 90.8, 91.5, 91.9],
        'Fuel_Level': [45, 44, 43, 46, 42, 41],
        'Tire_Pressure_Front': [32.1, 32.3, 32.0, 31.8, 32.2, 31.9],
        'Tire_Pressure_Rear': [30.5, 30.7, 30.4, 30.6, 30.8, 30.3],
        'Battery_Voltage': [12.5, 12.6, 12.7, 12.5, 12.6, 12.7],
        'Oil_Temp': [85.3, 86.1, 87.0, 85.9, 86.4, 86.8],
        'Odometer': [15000, 15020, 15040, 15010, 15030, 15050],
        'Fuel_Consumption': [10.5, 10.8, 11.2, 10.7, 11.0, 10.9]
    })
    st.dataframe(example_data.round(2))
    st.write("Your Excel file should contain numerical data with input and output columns.")
    
    st.info("💡 **Pro Tip**: With model persistence, you only need to train once per dataset!")
