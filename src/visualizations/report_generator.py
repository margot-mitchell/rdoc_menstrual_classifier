"""
Report generator module for creating comprehensive analysis reports.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
from src.visualizations.plotter import DataPlotter


class ReportGenerator:
    """Report generator for creating comprehensive analysis reports."""
    
    def __init__(self, output_dir: str):
        """
        Initialize report generator.
        
        Args:
            output_dir (str): Output directory for saving reports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.plotter = DataPlotter(output_dir)
    
    def generate_simulation_report(self, hormone_df: pd.DataFrame, period_df: pd.DataFrame,
                                 survey_df: pd.DataFrame, pattern_df: pd.DataFrame,
                                 config: Dict[str, Any], output_path: Optional[str] = None) -> None:
        """
        Generate a comprehensive simulation report.
        
        Args:
            hormone_df (pd.DataFrame): Hormone data
            period_df (pd.DataFrame): Period data
            survey_df (pd.DataFrame): Survey data
            pattern_df (pd.DataFrame): Pattern data
            config (dict): Simulation configuration
            output_path (str): Output path for the report
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'simulation_report.html')
        
        # Generate plots
        self.plotter.plot_hormone_cycles(hormone_df)
        self.plotter.plot_hormone_correlations(hormone_df)
        self.plotter.plot_cycle_length_distribution(survey_df)
        self.plotter.plot_pattern_distribution(pattern_df)
        
        # Calculate statistics
        stats = self._calculate_simulation_statistics(hormone_df, period_df, survey_df, pattern_df)
        
        # Generate HTML report
        html_content = self._create_simulation_html_report(stats, config)
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"Simulation report saved to {output_path}")
    
    def generate_classification_report(self, results: Dict[str, Dict[str, float]],
                                     config: Dict[str, Any], output_path: Optional[str] = None) -> None:
        """
        Generate a comprehensive classification report.
        
        Args:
            results (dict): Classification results
            config (dict): Classification configuration
            output_path (str): Output path for the report
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'classification_report.html')
        
        # Generate model comparison plot
        self.plotter.plot_model_comparison(results, 'accuracy')
        
        # Generate HTML report
        html_content = self._create_classification_html_report(results, config)
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"Classification report saved to {output_path}")
    
    def generate_cross_validation_report(self, cv_results: Dict[str, Dict[str, Any]],
                                       config: Dict[str, Any], output_path: Optional[str] = None) -> None:
        """
        Generate a comprehensive cross-validation report.
        
        Args:
            cv_results (dict): Cross-validation results
            config (dict): Cross-validation configuration
            output_path (str): Output path for the report
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'cross_validation_report.html')
        
        # Generate HTML report
        html_content = self._create_cv_html_report(cv_results, config)
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"Cross-validation report saved to {output_path}")
    
    def generate_summary_report(self, hormone_df: pd.DataFrame, survey_df: pd.DataFrame,
                              pattern_df: pd.DataFrame, results: Optional[Dict[str, Dict[str, float]]] = None,
                              output_path: Optional[str] = None) -> None:
        """
        Generate a summary report with all key findings.
        
        Args:
            hormone_df (pd.DataFrame): Hormone data
            survey_df (pd.DataFrame): Survey data
            pattern_df (pd.DataFrame): Pattern data
            results (dict): Model results (optional)
            output_path (str): Output path for the report
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'summary_report.html')
        
        # Generate dashboard
        self.plotter.create_summary_dashboard(hormone_df, survey_df, pattern_df, results)
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics(hormone_df, survey_df, pattern_df, results)
        
        # Generate HTML report
        html_content = self._create_summary_html_report(summary_stats)
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"Summary report saved to {output_path}")
    
    def _calculate_simulation_statistics(self, hormone_df: pd.DataFrame, period_df: pd.DataFrame,
                                       survey_df: pd.DataFrame, pattern_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate simulation statistics."""
        stats = {}
        
        # Hormone statistics
        hormone_cols = ['estradiol', 'progesterone', 'testosterone']
        for hormone in hormone_cols:
            if hormone in hormone_df.columns:
                stats[f'{hormone}_mean'] = hormone_df[hormone].mean()
                stats[f'{hormone}_std'] = hormone_df[hormone].std()
                stats[f'{hormone}_min'] = hormone_df[hormone].min()
                stats[f'{hormone}_max'] = hormone_df[hormone].max()
        
        # Cycle length statistics
        if 'cycle_length' in survey_df.columns:
            stats['cycle_length_mean'] = survey_df['cycle_length'].mean()
            stats['cycle_length_std'] = survey_df['cycle_length'].std()
            stats['cycle_length_min'] = survey_df['cycle_length'].min()
            stats['cycle_length_max'] = survey_df['cycle_length'].max()
        
        # Pattern distribution
        if 'menstrual_pattern' in pattern_df.columns:
            pattern_counts = pattern_df['menstrual_pattern'].value_counts()
            stats['pattern_distribution'] = pattern_counts.to_dict()
        
        # Subject statistics
        stats['n_subjects'] = hormone_df['subject_id'].nunique()
        stats['n_samples'] = len(hormone_df)
        
        return stats
    
    def _calculate_summary_statistics(self, hormone_df: pd.DataFrame, survey_df: pd.DataFrame,
                                    pattern_df: pd.DataFrame, results: Optional[Dict[str, Dict[str, float]]]) -> Dict[str, Any]:
        """Calculate summary statistics."""
        stats = self._calculate_simulation_statistics(hormone_df, None, survey_df, pattern_df)
        
        # Add model performance if available
        if results:
            stats['best_model'] = max(results.keys(), key=lambda x: results[x]['accuracy'])
            stats['best_accuracy'] = results[stats['best_model']]['accuracy']
            stats['model_results'] = results
        
        return stats
    
    def _create_simulation_html_report(self, stats: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Create HTML report for simulation results."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Simulation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .stat {{ margin: 10px 0; }}
                .plot {{ text-align: center; margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Menstrual Cycle Simulation Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Simulation Configuration</h2>
                <table>
                    <tr><th>Parameter</th><th>Value</th></tr>
                    <tr><td>Number of Subjects</td><td>{config['simulation']['n_subjects']}</td></tr>
                    <tr><td>Hormone Samples</td><td>{config['simulation']['n_hormone_samples']}</td></tr>
                    <tr><td>Period Days</td><td>{config['simulation']['n_period_days']}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Data Summary</h2>
                <div class="stat">Total Subjects: {stats['n_subjects']}</div>
                <div class="stat">Total Samples: {stats['n_samples']}</div>
            </div>
            
            <div class="section">
                <h2>Hormone Statistics</h2>
                <table>
                    <tr><th>Hormone</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr>
        """
        
        hormone_cols = ['estradiol', 'progesterone', 'testosterone']
        for hormone in hormone_cols:
            if f'{hormone}_mean' in stats:
                html += f"""
                    <tr>
                        <td>{hormone.capitalize()}</td>
                        <td>{stats[f'{hormone}_mean']:.3f}</td>
                        <td>{stats[f'{hormone}_std']:.3f}</td>
                        <td>{stats[f'{hormone}_min']:.3f}</td>
                        <td>{stats[f'{hormone}_max']:.3f}</td>
                    </tr>
                """
        
        html += """
                </table>
            </div>
        """
        
        if 'cycle_length_mean' in stats:
            html += f"""
            <div class="section">
                <h2>Cycle Length Statistics</h2>
                <div class="stat">Mean: {stats['cycle_length_mean']:.1f} days</div>
                <div class="stat">Standard Deviation: {stats['cycle_length_std']:.1f} days</div>
                <div class="stat">Range: {stats['cycle_length_min']:.0f} - {stats['cycle_length_max']:.0f} days</div>
            </div>
            """
        
        if 'pattern_distribution' in stats:
            html += """
            <div class="section">
                <h2>Pattern Distribution</h2>
                <table>
                    <tr><th>Pattern</th><th>Count</th></tr>
            """
            for pattern, count in stats['pattern_distribution'].items():
                html += f"<tr><td>{pattern}</td><td>{count}</td></tr>"
            html += "</table></div>"
        
        html += """
            <div class="section">
                <h2>Visualizations</h2>
                <div class="plot">
                    <img src="hormone_cycles.png" alt="Hormone Cycles" style="max-width: 100%;">
                    <p>Hormone Cycles</p>
                </div>
                <div class="plot">
                    <img src="hormone_correlations.png" alt="Hormone Correlations" style="max-width: 100%;">
                    <p>Hormone Correlations</p>
                </div>
                <div class="plot">
                    <img src="cycle_length_distribution.png" alt="Cycle Length Distribution" style="max-width: 100%;">
                    <p>Cycle Length Distribution</p>
                </div>
                <div class="plot">
                    <img src="pattern_distribution.png" alt="Pattern Distribution" style="max-width: 100%;">
                    <p>Pattern Distribution</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _create_classification_html_report(self, results: Dict[str, Dict[str, float]], config: Dict[str, Any]) -> str:
        """Create HTML report for classification results."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Classification Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .plot {{ text-align: center; margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .best {{ background-color: #d4edda; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Classification Results Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Model Performance Comparison</h2>
                <table>
                    <tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-Score</th></tr>
        """
        
        best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
        
        for model, metrics in results.items():
            row_class = "best" if model == best_model else ""
            html += f"""
                <tr class="{row_class}">
                    <td>{model}</td>
                    <td>{metrics['accuracy']:.4f}</td>
                    <td>{metrics['precision']:.4f}</td>
                    <td>{metrics['recall']:.4f}</td>
                    <td>{metrics['f1']:.4f}</td>
                </tr>
            """
        
        html += f"""
                </table>
                <p><strong>Best Model:</strong> {best_model} (Accuracy: {results[best_model]['accuracy']:.4f})</p>
            </div>
            
            <div class="section">
                <h2>Model Comparison Visualization</h2>
                <div class="plot">
                    <img src="model_comparison_accuracy.png" alt="Model Comparison" style="max-width: 100%;">
                    <p>Model Performance Comparison</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _create_cv_html_report(self, cv_results: Dict[str, Dict[str, Any]], config: Dict[str, Any]) -> str:
        """Create HTML report for cross-validation results."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cross-Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .best {{ background-color: #d4edda; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Cross-Validation Results Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Cross-Validation Configuration</h2>
                <p>Folds: {config['cross_validation']['cv_folds']}</p>
                <p>Repeats: {config['cross_validation'].get('cv_repeats', 1)}</p>
                <p>Scoring: {config['cross_validation']['scoring']}</p>
            </div>
            
            <div class="section">
                <h2>Cross-Validation Results</h2>
                <table>
                    <tr><th>Model</th><th>Mean CV Score</th><th>Std CV Score</th><th>Mean Accuracy</th><th>Mean F1</th></tr>
        """
        
        best_model = max(cv_results.keys(), key=lambda x: cv_results[x]['mean_cv_score'])
        
        for model, results in cv_results.items():
            row_class = "best" if model == best_model else ""
            html += f"""
                <tr class="{row_class}">
                    <td>{model}</td>
                    <td>{results['mean_cv_score']:.4f}</td>
                    <td>{results['std_cv_score']:.4f}</td>
                    <td>{results['mean_accuracy']:.4f}</td>
                    <td>{results['mean_f1']:.4f}</td>
                </tr>
            """
        
        html += f"""
                </table>
                <p><strong>Best Model:</strong> {best_model} (CV Score: {cv_results[best_model]['mean_cv_score']:.4f} ± {cv_results[best_model]['std_cv_score']*2:.4f})</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _create_summary_html_report(self, stats: Dict[str, Any]) -> str:
        """Create HTML report for summary results."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Summary Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .plot {{ text-align: center; margin: 20px 0; }}
                .highlight {{ background-color: #fff3cd; padding: 10px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Menstrual Cycle Analysis Summary Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Key Findings</h2>
                <div class="highlight">
                    <p><strong>Data Summary:</strong> {stats['n_subjects']} subjects, {stats['n_samples']} samples</p>
        """
        
        if 'cycle_length_mean' in stats:
            html += f"<p><strong>Average Cycle Length:</strong> {stats['cycle_length_mean']:.1f} days</p>"
        
        if 'best_model' in stats:
            html += f"<p><strong>Best Performing Model:</strong> {stats['best_model']} (Accuracy: {stats['best_accuracy']:.4f})</p>"
        
        html += """
                </div>
            </div>
            
            <div class="section">
                <h2>Summary Dashboard</h2>
                <div class="plot">
                    <img src="summary_dashboard.png" alt="Summary Dashboard" style="max-width: 100%;">
                    <p>Comprehensive Analysis Dashboard</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html 