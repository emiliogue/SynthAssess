import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, mean_absolute_error
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from sklearn.utils import resample
import numpy as np
from imblearn.over_sampling import SMOTE

class MLEfficacyMetrics:
    def __init__(self, original_data, synthetic_data, holdout_data, target_column, target_type):
        self.original_data = original_data
        self.synthetic_data = synthetic_data
        self.holdout_data = holdout_data
        self.target_column = target_column
        self.target_type = target_type
        self.preprocess_data()
        
    def preprocess_data(self):
        '''
        Preprocess the data for training the XGBoost model.
        
        - Separate features and target
        - Identify categorical features
        - Encode categorical features
        - Align columns across all datasets
        - Encode target if classification
        '''
        # Separate features and target
        self.X_original = self.original_data.drop(columns=[self.target_column])
        self.y_original = self.original_data[self.target_column]
        
        self.X_synthetic = self.synthetic_data.drop(columns=[self.target_column])
        self.y_synthetic = self.synthetic_data[self.target_column]
        
        self.X_holdout = self.holdout_data.drop(columns=[self.target_column])
        self.y_holdout = self.holdout_data[self.target_column]
        
        # Identify categorical features
        self.categorical_features = self.X_original.select_dtypes(include=['object', 'category']).columns
        
        # Encode categorical features
        self.X_original = self.encode_categorical_features(self.X_original)
        self.X_synthetic = self.encode_categorical_features(self.X_synthetic)
        self.X_holdout = self.encode_categorical_features(self.X_holdout)
        
        # Align columns across all datasets
        self.align_columns()
        
        # Encode target if classification
        if self.target_type == 'classification' and self.y_original.dtype == 'object':
            le = LabelEncoder()
            self.y_original = le.fit_transform(self.y_original)
            self.y_synthetic = le.transform(self.y_synthetic)
            self.y_holdout = le.transform(self.y_holdout)
        
    def encode_categorical_features(self, df):
        '''
        One-hot encode categorical features.
        :param df: DataFrame containing categorical features.
        :return: DataFrame with one-hot encoded categorical features.
        '''
        for col in self.categorical_features:
            df = pd.get_dummies(df, columns=[col], prefix=[col])
        return df
    
    def align_columns(self):
        '''
        Align columns across all datasets.
        '''

        # Ensure all dataframes have the same columns
        all_columns = set(self.X_original.columns) | set(self.X_synthetic.columns) | set(self.X_holdout.columns)
        
        for col in all_columns:
            if col not in self.X_original:
                self.X_original[col] = 0
            if col not in self.X_synthetic:
                self.X_synthetic[col] = 0
            if col not in self.X_holdout:
                self.X_holdout[col] = 0
                
        self.X_original = self.X_original[sorted(all_columns)]
        self.X_synthetic = self.X_synthetic[sorted(all_columns)]
        self.X_holdout = self.X_holdout[sorted(all_columns)]
        
    def train_model(self, data_type='original'):
        '''
        Train an XGBoost model on the specified data type.
        :param data_type: str, 'original' or 'synthetic'
        :return: trained XGBoost model
        '''
        if data_type == 'original':
            X = self.X_original
            y = self.y_original
        elif data_type == 'synthetic':
            X = self.X_synthetic
            y = self.y_synthetic
        else:
            raise ValueError("data_type should be either 'original' or 'synthetic'")
        
        # Check data for class imbalance and handle if necessary
        if self.target_type == 'classification':
            if len(y.unique()) > 2:
                X, y = self.handle_class_imbalance(X, y)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train XGBoost model
        if self.target_type == 'classification':

            # check if the target is binary or multiclass
            if len(y_train.unique()) == 2:
                model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            else:
                model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

        elif self.target_type == 'regression':
            model = xgb.XGBRegressor()
        else:
            raise ValueError("target_type should be either 'classification' or 'regression'")
        model.set_params(early_stopping_rounds=10)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        return model
    
    # def plot_roc_curve(self, model_1, model_2, X, y):
    #     # Calculate ROC curve for both models
    #     fpr_1, tpr_1, _ = roc_curve(y, model_1.predict_proba(X)[:, 1])
    #     fpr_2, tpr_2, _ = roc_curve(y, model_2.predict_proba(X)[:, 1])

    #     # Create Plotly figure
    #     fig = make_subplots()
        
    #     # Add traces for both models
    #     fig.add_trace(go.Scatter(x=fpr_1, y=tpr_1, mode='lines', name='Original Data Model'))
    #     fig.add_trace(go.Scatter(x=fpr_2, y=tpr_2, mode='lines', name='Synthetic Data Model'))
        
    #     # Update layout
    #     fig.update_layout(
    #         title='ROC Curve',
    #         xaxis_title='False Positive Rate',
    #         yaxis_title='True Positive Rate',
    #         legend=dict(x=0.8, y=0.2)
    #     )
        
    #     return fig


    def plot_roc_curve(self, model_1, model_2, X, y, n_bootstrap=1000, alpha=0.05):
        '''
        Plot ROC curve with confidence intervals for both models.
        :param model_1: trained model on original data
        :param model_2: trained model on synthetic data
        :param X: features
        :param y: target
        :param n_bootstrap: number of bootstrap iterations
        :param alpha: significance level for confidence intervals
        :return: Plotly figure
        '''
        
        def bootstrap_roc(model, X, y):
            '''
            Calculate ROC curve for a model using bootstrapping.
            :param model: trained model
            :param X: features
            :param y: target
            :return: list of false positive rates and true positive rates
            '''
            fprs, tprs = [], []
            for _ in range(n_bootstrap):
                X_resampled, y_resampled = resample(X, y)
                fpr, tpr, _ = roc_curve(y_resampled, model.predict_proba(X_resampled)[:, 1])
                fprs.append(fpr)
                tprs.append(tpr)
            return fprs, tprs

        def mean_ci(fprs, tprs):
            '''
            Calculate mean ROC curve and confidence intervals.
            :param fprs: list of false positive rates
            :param tprs: list of true positive rates
            :return: mean false positive rate, mean true positive rate, lower bound, upper bound
            '''
            mean_fpr = np.linspace(0, 1, 100)
            tprs_interpolated = []
            for i in range(len(fprs)):
                tprs_interpolated.append(np.interp(mean_fpr, fprs[i], tprs[i]))
            mean_tpr = np.mean(tprs_interpolated, axis=0)
            std_tpr = np.std(tprs_interpolated, axis=0)
            tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
            tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
            return mean_fpr, mean_tpr, tpr_lower, tpr_upper

        # Calculate ROC curve and confidence intervals for both models
        fprs_1, tprs_1 = bootstrap_roc(model_1, X, y)
        fprs_2, tprs_2 = bootstrap_roc(model_2, X, y)
        
        mean_fpr_1, mean_tpr_1, tpr_lower_1, tpr_upper_1 = mean_ci(fprs_1, tprs_1)
        mean_fpr_2, mean_tpr_2, tpr_lower_2, tpr_upper_2 = mean_ci(fprs_2, tprs_2)

        # Calculate AUROC for both models
        auroc_1 = np.trapz(mean_tpr_1, mean_fpr_1)
        auroc_2 = np.trapz(mean_tpr_2, mean_fpr_2)

        # Create Plotly figure
        fig = make_subplots()

        # Add traces for both models
        fig.add_trace(go.Scatter(x=mean_fpr_1, y=mean_tpr_1, mode='lines', name='Original Data Model', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=mean_fpr_2, y=mean_tpr_2, mode='lines', name='Synthetic Data Model', line=dict(color='red')))
        
        # Add confidence interval for both models
        fig.add_trace(go.Scatter(x=np.concatenate([mean_fpr_1, mean_fpr_1[::-1]]),
                                 y=np.concatenate([tpr_upper_1, tpr_lower_1[::-1]]),
                                 fill='toself', fillcolor='rgba(0, 0, 255, 0.2)', line=dict(color='rgba(255, 255, 255, 0)'),
                                 showlegend=False, name='Original Data Model CI'))
        fig.add_trace(go.Scatter(x=np.concatenate([mean_fpr_2, mean_fpr_2[::-1]]),
                                 y=np.concatenate([tpr_upper_2, tpr_lower_2[::-1]]),
                                 fill='toself', fillcolor='rgba(255, 0, 0, 0.2)', line=dict(color='rgba(255, 255, 255, 0)'),
                                 showlegend=False, name='Synthetic Data Model CI'))
        
        # Add AUROC values
        fig.add_annotation(x=0.5, y=0.5, xref='paper', yref='paper', text=f'Original Data AUROC: {auroc_1:.2f}', showarrow=False)
        fig.add_annotation(x=0.5, y=0.4, xref='paper', yref='paper', text=f'Synthetic Data AUROC: {auroc_2:.2f}', showarrow=False)

        # Update layout
        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            legend=dict(x=0.8, y=0.2)
        )
        
        return fig
    
    # def assess_efficacy(self):
    #     # Train models
    #     model_original = self.train_model(data_type='original')
    #     model_synthetic = self.train_model(data_type='synthetic')
        

    #     # Predictions on holdout data
    #     y_pred_original = model_original.predict(self.X_holdout)
    #     y_pred_synthetic = model_synthetic.predict(self.X_holdout)
        
    #     if self.target_type == 'classification':
    #         accuracy_original = accuracy_score(self.y_holdout, y_pred_original)
    #         accuracy_synthetic = accuracy_score(self.y_holdout, y_pred_synthetic)
    #         report_original = classification_report(self.y_holdout, y_pred_original)
    #         report_synthetic = classification_report(self.y_holdout, y_pred_synthetic)
            
    #         print("Original Data Model Accuracy:", accuracy_original)
    #         print("Synthetic Data Model Accuracy:", accuracy_synthetic)

    #         print("Original Data Classification Report:")
    #         print(report_original)
    #         print("Synthetic Data Classification Report:")
    #         print(report_synthetic)

    #         # add report to dataframe
    #         report_original = pd.DataFrame(classification_report(self.y_holdout, y_pred_original, output_dict=True)).T
    #         report_synthetic = pd.DataFrame(classification_report(self.y_holdout, y_pred_synthetic, output_dict=True)).T
    #         # round to 2 decimal places
    #         report_original = report_original.round(2)
    #         report_synthetic = report_synthetic.round(2)
    #         report_original.reset_index(inplace=True)
    #         report_synthetic.reset_index(inplace=True)

    #         # plot ROC curve and confusion matrix
    #         fig = self.plot_roc_curve(model_original, model_synthetic, self.X_holdout, self.y_holdout)


            
    #     elif self.target_type == 'regression':
    #         mse_original = mean_squared_error(self.y_holdout, y_pred_original)
    #         mse_synthetic = mean_squared_error(self.y_holdout, y_pred_synthetic)
    #         mae_original = mean_absolute_error(self.y_holdout, y_pred_original)
    #         mae_synthetic = mean_absolute_error(self.y_holdout, y_pred_synthetic)
            
    #         print("Original Data Model MSE:", mse_original)
    #         print("Synthetic Data Model MSE:", mse_synthetic)
    #         print("Original Data Model MAE:", mae_original)
    #         print("Synthetic Data Model MAE:", mae_synthetic)
    #     return report_original, report_synthetic, fig
    
    def bootstrap_confidence_interval(self, metric_function, y_true, y_pred, n_bootstrap=1000, alpha=0.05):
        '''
        Calculate bootstrap confidence interval for a given metric.
        :param metric_function: metric function
        :param y_true: true target values
        :param y_pred: predicted target values
        :param n_bootstrap: number of bootstrap iterations
        :param alpha: significance level for confidence intervals
        :return: lower bound, upper bound
        '''
        bootstrap_metrics = []
        for _ in range(n_bootstrap):
            y_true_resampled, y_pred_resampled = resample(y_true, y_pred)
            bootstrap_metrics.append(metric_function(y_true_resampled, y_pred_resampled))
        lower_bound = np.percentile(bootstrap_metrics, 100 * alpha / 2)
        upper_bound = np.percentile(bootstrap_metrics, 100 * (1 - alpha / 2))
        return lower_bound, upper_bound

    def handle_class_imbalance(self, X, y):
        '''
        Handle class imbalance using SMOTE.
        :param X: features
        :param y: target
        :return: resampled features and target
        '''
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled
    

    def assess_efficacy(self):
        '''
        Assess the efficacy of the models trained on original and synthetic data.
        :return: classification reports for both models, Plotly figure
        '''
        # Train model
        model_original = self.train_model(data_type='original')
        model_synthetic = self.train_model(data_type='synthetic')

        # Predictions on holdout data
        y_pred_original = model_original.predict(self.X_holdout)
        y_pred_synthetic = model_synthetic.predict(self.X_holdout)
        
        if self.target_type == 'classification':
            accuracy_original = accuracy_score(self.y_holdout, y_pred_original)
            accuracy_synthetic = accuracy_score(self.y_holdout, y_pred_synthetic)
            report_original = classification_report(self.y_holdout, y_pred_original)
            report_synthetic = classification_report(self.y_holdout, y_pred_synthetic)

            # Bootstrap confidence intervals for accuracy
            acc_original_ci = self.bootstrap_confidence_interval(accuracy_score, self.y_holdout, y_pred_original)
            acc_synthetic_ci = self.bootstrap_confidence_interval(accuracy_score, self.y_holdout, y_pred_synthetic)

            # Add report to dataframe
            report_original = pd.DataFrame(classification_report(self.y_holdout, y_pred_original, output_dict=True)).T
            report_synthetic = pd.DataFrame(classification_report(self.y_holdout, y_pred_synthetic, output_dict=True)).T
            # Round to 2 decimal places
            report_original = report_original.round(2)
            report_synthetic = report_synthetic.round(2)
            report_original.reset_index(inplace=True)
            report_synthetic.reset_index(inplace=True)

            # Plot ROC curve and confusion matrix
            fig = self.plot_roc_curve(model_original, model_synthetic, self.X_holdout, self.y_holdout)
        
        elif self.target_type == 'regression':
            mse_original = mean_squared_error(self.y_holdout, y_pred_original)
            mse_synthetic = mean_squared_error(self.y_holdout, y_pred_synthetic)
            mae_original = mean_absolute_error(self.y_holdout, y_pred_original)
            mae_synthetic = mean_absolute_error(self.y_holdout, y_pred_synthetic)
            mape_original = np.mean(np.abs((self.y_holdout - y_pred_original) / self.y_holdout)) * 100
            mape_synthetic = np.mean(np.abs((self.y_holdout - y_pred_synthetic) / self.y_holdout)) * 100
            

            # Bootstrap confidence intervals for MSE and MAE
            mse_original_ci = self.bootstrap_confidence_interval(mean_squared_error, self.y_holdout, y_pred_original)
            mse_synthetic_ci = self.bootstrap_confidence_interval(mean_squared_error, self.y_holdout, y_pred_synthetic)
            mae_original_ci = self.bootstrap_confidence_interval(mean_absolute_error, self.y_holdout, y_pred_original)
            mae_synthetic_ci = self.bootstrap_confidence_interval(mean_absolute_error, self.y_holdout, y_pred_synthetic)
            mape_original_ci = self.bootstrap_confidence_interval(lambda x, y: np.mean(np.abs((x - y) / x) * 100), self.y_holdout, y_pred_original)
            mape_synthetic_ci = self.bootstrap_confidence_interval(lambda x, y: np.mean(np.abs((x - y) / x) * 100), self.y_holdout, y_pred_synthetic)

            # Regression Report with confidence intervals
            report_original = pd.DataFrame({'Metric': ['Mean Squared Error', 'Mean Absolute Error', 'Mean Absolute Percentage Error'],
                                            'Value': [mse_original, mae_original, mape_original],
                                            '95% CI Lower Bound': [mse_original_ci[0], mae_original_ci[0], mape_original_ci[0]],
                                            '95% CI Upper Bound': [mse_original_ci[1], mae_original_ci[1], mape_original_ci[1]]})
            report_synthetic = pd.DataFrame({'Metric': ['Mean Squared Error', 'Mean Absolute Error', 'Mean Absolute Percentage Error'],
                                            'Value': [mse_synthetic, mae_synthetic, mape_synthetic],
                                            '95% CI Lower Bound': [mse_synthetic_ci[0], mae_synthetic_ci[0], mape_synthetic_ci[0]],
                                            '95% CI Upper Bound': [mse_synthetic_ci[1], mae_synthetic_ci[1], mape_synthetic_ci[1]]})
            # Round to 2 decimal places
            report_original = report_original.round(2)
            report_synthetic = report_synthetic.round(2)
 

            fig = self.plot_residuals(model_original, model_synthetic, self.X_holdout, self.y_holdout)

        return report_original, report_synthetic, fig
    
    def plot_residuals(self, model_1, model_2, X, y):
        '''
        Plot residuals for both models.
        :param model_1: trained model on original data
        :param model_2: trained model on synthetic data
        :param X: features
        :param y: target
        :return: Plotly figure
        '''
        # Calculate residuals for both models
        residuals_1 = y - model_1.predict(X)
        residuals_2 = y - model_2.predict(X)

        # Create Plotly figure
        fig = make_subplots(rows=1, cols=1, subplot_titles=('Original Data Model', 'Synthetic Data Model'))

        # Add traces for both models
        fig.add_trace(go.Scatter(x=y, y=residuals_1, mode='markers', name='Original Data Model'), row=1, col=1)
        fig.add_trace(go.Scatter
                        (x=y, y=residuals_2, mode='markers', name='Synthetic Data Model'), row=1, col=1)
        
        # Update layout
        fig.update_layout(title='Residual Plot', xaxis_title='True Values', yaxis_title='Residuals')

        return fig
    
    def train_test_discriminator(self, anchor, comparison):
        '''
        Train a discriminator model to distinguish between anchor and comparison data and assess performance.
        :param anchor: df, anchor data
        :param comparison: df, comparison data
        :return: classification report, Plotly figure
        '''
        # Check if the datasets are the same size and resample if necessary
        if len(anchor) != len(comparison):
            min_len = min(len(anchor), len(comparison))
            self.anchor = resample(anchor, n_samples=min_len)
            self.comparison = resample(comparison, n_samples=min_len)
        else:
            self.anchor = anchor
            self.comparison = comparison

        self.anchor['data_label'] = 0
        self.comparison['data_label'] = 1
        # Combine anchor and comparison data
        X = pd.concat([self.anchor, self.comparison])

        # Test-train split
        X_train, X_test, y_train, y_test = train_test_split(X.drop(columns='data_label'), X['data_label'], test_size=0.2, random_state=42)

        # Train XGBoost model
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # Predictions on test data
        y_pred = model.predict(X_test)

        # Classification report
        report = classification_report(y_test, y_pred)
        # Add report to dataframe
        report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T
        # Round to 2 decimal places
        report = report.round(2)
        report.reset_index(inplace=True)

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines'))
        fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')

        # calculate feature importance
        feature_importance = model.feature_importances_
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=feature_importance[sorted_idx], y=pos, orientation='h'))
        fig2.update_layout(title='Feature Importance', xaxis_title='Relative Importance', yaxis_title='Features')
        # add feature names
        fig2.update_yaxes(tickvals=pos, ticktext=X.columns[sorted_idx])


        return report, fig, fig2

    