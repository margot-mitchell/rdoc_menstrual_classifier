"""
Plotter module for generating various plots and visualizations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List
from datetime import datetime


class DataPlotter:
    """Data plotter for generating various visualizations."""
    
    def __init__(self, output_dir: str):
        """
        Initialize data plotter.
        
        Args:
            output_dir (str): Output directory for saving plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_hormone_cycles(self, hormone_df: pd.DataFrame, subject_ids: Optional[List[int]] = None,
                           output_path: Optional[str] = None) -> None:
        """
        Plot hormone cycles for specified subjects.
        
        Args:
            hormone_df (pd.DataFrame): Hormone data
            subject_ids (list): List of subject IDs to plot (if None, plot first 5)
            output_path (str): Output path for the plot
        """
        if subject_ids is None:
            subject_ids = hormone_df['subject_id'].unique()[:5]
        
        n_subjects = len(subject_ids)
        fig, axes = plt.subplots(n_subjects, 1, figsize=(12, 4 * n_subjects))
        if n_subjects == 1:
            axes = [axes]
        
        for i, subject_id in enumerate(subject_ids):
            subject_data = hormone_df[hormone_df['subject_id'] == subject_id].copy()
            
            if 'cycle_day' in subject_data.columns:
                x_col = 'cycle_day'
                x_label = 'Cycle Day'
            else:
                x_col = 'date'
                x_label = 'Date'
            
            # Plot hormone levels
            for hormone in ['estradiol', 'progesterone', 'testosterone']:
                if hormone in subject_data.columns:
                    axes[i].plot(subject_data[x_col], subject_data[hormone], 
                               label=hormone.capitalize(), marker='o', markersize=3)
            
            axes[i].set_xlabel(x_label)
            axes[i].set_ylabel('Hormone Level')
            axes[i].set_title(f'Subject {subject_id} - Hormone Cycles')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'hormone_cycles.png')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_phase_distribution(self, hormone_df: pd.DataFrame, output_path: Optional[str] = None) -> None:
        """
        Plot distribution of menstrual cycle phases.
        
        Args:
            hormone_df (pd.DataFrame): Hormone data with phase information
            output_path (str): Output path for the plot
        """
        if 'phase' not in hormone_df.columns:
            print("No phase information available in the data")
            return
        
        plt.figure(figsize=(10, 6))
        phase_counts = hormone_df['phase'].value_counts()
        
        plt.pie(phase_counts.values, labels=phase_counts.index, autopct='%1.1f%%')
        plt.title('Distribution of Menstrual Cycle Phases')
        
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'phase_distribution.png')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_hormone_correlations(self, hormone_df: pd.DataFrame, output_path: Optional[str] = None) -> None:
        """
        Plot correlation matrix of hormone levels.
        
        Args:
            hormone_df (pd.DataFrame): Hormone data
            output_path (str): Output path for the plot
        """
        hormone_cols = ['estradiol', 'progesterone', 'testosterone']
        available_cols = [col for col in hormone_cols if col in hormone_df.columns]
        
        if len(available_cols) < 2:
            print("Need at least 2 hormone columns for correlation plot")
            return
        
        correlation_matrix = hormone_df[available_cols].corr()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Hormone Level Correlations')
        
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'hormone_correlations.png')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_cycle_length_distribution(self, survey_df: pd.DataFrame, output_path: Optional[str] = None) -> None:
        """
        Plot distribution of cycle lengths.
        
        Args:
            survey_df (pd.DataFrame): Survey data with cycle length information
            output_path (str): Output path for the plot
        """
        if 'cycle_length' not in survey_df.columns:
            print("No cycle length information available in the data")
            return
        
        plt.figure(figsize=(10, 6))
        plt.hist(survey_df['cycle_length'], bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Cycle Length (days)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Menstrual Cycle Lengths')
        plt.grid(True, alpha=0.3)
        
        # Add mean line
        mean_cycle = survey_df['cycle_length'].mean()
        plt.axvline(mean_cycle, color='red', linestyle='--', 
                   label=f'Mean: {mean_cycle:.1f} days')
        plt.legend()
        
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'cycle_length_distribution.png')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_pattern_distribution(self, pattern_df: pd.DataFrame, output_path: Optional[str] = None) -> None:
        """
        Plot distribution of menstrual patterns.
        
        Args:
            pattern_df (pd.DataFrame): Pattern data
            output_path (str): Output path for the plot
        """
        if 'menstrual_pattern' not in pattern_df.columns:
            print("No menstrual pattern information available in the data")
            return
        
        plt.figure(figsize=(10, 6))
        pattern_counts = pattern_df['menstrual_pattern'].value_counts()
        
        plt.bar(pattern_counts.index, pattern_counts.values)
        plt.xlabel('Menstrual Pattern')
        plt.ylabel('Count')
        plt.title('Distribution of Menstrual Patterns')
        plt.xticks(rotation=45)
        
        # Add count labels on bars
        for i, v in enumerate(pattern_counts.values):
            plt.text(i, v + 0.5, str(v), ha='center')
        
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'pattern_distribution.png')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_model_comparison(self, results: Dict[str, Dict[str, float]], 
                            metric: str = 'accuracy', output_path: Optional[str] = None) -> None:
        """
        Plot model comparison for a specific metric.
        
        Args:
            results (dict): Dictionary of model results
            metric (str): Metric to compare
            output_path (str): Output path for the plot
        """
        models = list(results.keys())
        values = [results[model][metric] for model in models]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, values)
        plt.xlabel('Model')
        plt.ylabel(metric.capitalize())
        plt.title(f'Model Comparison - {metric.capitalize()}')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if output_path is None:
            output_path = os.path.join(self.output_dir, f'model_comparison_{metric}.png')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_training_history(self, history: Dict[str, List[float]], model_name: str,
                            output_path: Optional[str] = None) -> None:
        """
        Plot training history for neural network models.
        
        Args:
            history (dict): Training history dictionary
            model_name (str): Name of the model
            output_path (str): Output path for the plot
        """
        if not history:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        if 'loss' in history and 'val_loss' in history:
            ax1.plot(history['loss'], label='Training Loss')
            ax1.plot(history['val_loss'], label='Validation Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Model Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        if 'accuracy' in history and 'val_accuracy' in history:
            ax2.plot(history['accuracy'], label='Training Accuracy')
            ax2.plot(history['val_accuracy'], label='Validation Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Model Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Training History - {model_name}')
        plt.tight_layout()
        
        if output_path is None:
            output_path = os.path.join(self.output_dir, f'{model_name.lower().replace(" ", "_")}_training_history.png')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_summary_dashboard(self, hormone_df: pd.DataFrame, survey_df: pd.DataFrame,
                               pattern_df: pd.DataFrame, results: Optional[Dict[str, Dict[str, float]]] = None,
                               output_path: Optional[str] = None) -> None:
        """
        Create a summary dashboard with multiple plots.
        
        Args:
            hormone_df (pd.DataFrame): Hormone data
            survey_df (pd.DataFrame): Survey data
            pattern_df (pd.DataFrame): Pattern data
            results (dict): Model results (optional)
            output_path (str): Output path for the dashboard
        """
        fig = plt.figure(figsize=(20, 15))
        
        # Create subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Hormone correlations
        ax1 = fig.add_subplot(gs[0, 0])
        hormone_cols = ['estradiol', 'progesterone', 'testosterone']
        available_cols = [col for col in hormone_cols if col in hormone_df.columns]
        if len(available_cols) >= 2:
            correlation_matrix = hormone_df[available_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax1)
            ax1.set_title('Hormone Correlations')
        
        # Plot 2: Cycle length distribution
        ax2 = fig.add_subplot(gs[0, 1])
        if 'cycle_length' in survey_df.columns:
            ax2.hist(survey_df['cycle_length'], bins=15, alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Cycle Length (days)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Cycle Length Distribution')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Pattern distribution
        ax3 = fig.add_subplot(gs[0, 2])
        if 'menstrual_pattern' in pattern_df.columns:
            pattern_counts = pattern_df['menstrual_pattern'].value_counts()
            ax3.bar(pattern_counts.index, pattern_counts.values)
            ax3.set_xlabel('Pattern')
            ax3.set_ylabel('Count')
            ax3.set_title('Pattern Distribution')
            ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Hormone levels over time (sample subject)
        ax4 = fig.add_subplot(gs[1, :])
        if 'subject_id' in hormone_df.columns:
            sample_subject = hormone_df['subject_id'].iloc[0]
            subject_data = hormone_df[hormone_df['subject_id'] == sample_subject]
            if 'cycle_day' in subject_data.columns:
                for hormone in available_cols:
                    ax4.plot(subject_data['cycle_day'], subject_data[hormone], 
                            label=hormone.capitalize(), marker='o')
                ax4.set_xlabel('Cycle Day')
                ax4.set_ylabel('Hormone Level')
                ax4.set_title(f'Hormone Levels - Subject {sample_subject}')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
        
        # Plot 5: Model comparison (if results available)
        if results:
            ax5 = fig.add_subplot(gs[2, :])
            models = list(results.keys())
            accuracies = [results[model]['accuracy'] for model in models]
            ax5.bar(models, accuracies)
            ax5.set_xlabel('Model')
            ax5.set_ylabel('Accuracy')
            ax5.set_title('Model Performance Comparison')
            ax5.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for i, v in enumerate(accuracies):
                ax5.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.suptitle('Menstrual Cycle Analysis Dashboard', fontsize=16)
        
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'summary_dashboard.png')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close() 