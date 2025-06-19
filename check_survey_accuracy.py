import pandas as pd
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_survey_accuracy():
    """Check the accuracy of date_of_last_period and cycle_length from survey data."""
    
    # Load data
    survey_df = pd.read_csv('output/survey_responses.csv')
    period_df = pd.read_csv('output/period_sleep_data.csv')
    
    # Convert dates to datetime
    survey_df['date_of_last_period'] = pd.to_datetime(survey_df['date_of_last_period'])
    survey_df['date_of_response'] = pd.to_datetime(survey_df['date_of_response'])
    period_df['date'] = pd.to_datetime(period_df['date'])
    
    logger.info(f"Loaded {len(survey_df)} survey responses")
    logger.info(f"Loaded {len(period_df)} period records")
    
    # Get the date range of period data
    min_period_date = period_df['date'].min()
    max_period_date = period_df['date'].max()
    logger.info(f"Period data date range: {min_period_date} to {max_period_date}")
    
    # Analyze each subject
    accuracy_results = []
    
    for _, survey_row in survey_df.iterrows():
        subject_id = survey_row['subject_id']
        reported_last_period = survey_row['date_of_last_period']
        reported_cycle_length = survey_row['cycle_length']
        response_date = survey_row['date_of_response']
        
        # Get actual period data for this subject
        subject_periods = period_df[period_df['subject_id'] == subject_id].copy()
        subject_periods = subject_periods.sort_values('date')
        
        # Find actual periods (where period == 'Yes')
        actual_periods = subject_periods[subject_periods['period'] == 'Yes']['date'].tolist()
        
        if not actual_periods:
            logger.warning(f"Subject {subject_id}: No actual periods found in period data")
            continue
        
        # Check if reported last period is within THIS SUBJECT'S period data range
        subject_min_date = subject_periods['date'].min()
        subject_max_date = subject_periods['date'].max()
        reported_in_range = subject_min_date <= reported_last_period <= subject_max_date
        
        # Check if reported last period matches any actual period (only if in range)
        period_match = False
        closest_period = None
        days_diff = None
        
        if reported_in_range:
            period_match = reported_last_period in actual_periods
            
            # Find the closest actual period to the reported date
            if actual_periods:
                closest_period = min(actual_periods, key=lambda x: abs((x - reported_last_period).days))
                days_diff = abs((closest_period - reported_last_period).days)
        
        # Calculate actual cycle length from period data
        if len(actual_periods) >= 2:
            # Find consecutive period starts (first day of each period)
            # Group consecutive period days and take the first day of each period
            period_starts = []
            current_period_start = None
            
            for period_date in actual_periods:
                if current_period_start is None:
                    current_period_start = period_date
                elif (period_date - current_period_start).days > 7:  # Gap of more than 7 days indicates new period
                    period_starts.append(current_period_start)
                    current_period_start = period_date
            
            # Add the last period start
            if current_period_start is not None:
                period_starts.append(current_period_start)
            
            # Calculate intervals between consecutive period starts
            if len(period_starts) >= 2:
                intervals = []
                for i in range(1, len(period_starts)):
                    interval = (period_starts[i] - period_starts[i-1]).days
                    intervals.append(interval)
                
                actual_cycle_length = np.mean(intervals)
                cycle_length_diff = abs(reported_cycle_length - actual_cycle_length)
            else:
                actual_cycle_length = None
                cycle_length_diff = None
        else:
            actual_cycle_length = None
            cycle_length_diff = None
        
        # Store results
        result = {
            'subject_id': subject_id,
            'reported_last_period': reported_last_period,
            'reported_cycle_length': reported_cycle_length,
            'response_date': response_date,
            'reported_in_range': reported_in_range,
            'period_match': period_match,
            'closest_period': closest_period,
            'days_diff': days_diff,
            'actual_cycle_length': actual_cycle_length,
            'cycle_length_diff': cycle_length_diff,
            'num_actual_periods': len(actual_periods)
        }
        accuracy_results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(accuracy_results)
    
    # Print summary statistics
    logger.info("\n=== SURVEY DATA ACCURACY ANALYSIS ===")
    
    # Count how many reported dates are outside the range
    out_of_range = results_df['reported_in_range'].sum()
    total_subjects = len(results_df)
    logger.info(f"Reported dates within period data range: {out_of_range}/{total_subjects} ({out_of_range/total_subjects*100:.1f}%)")
    
    # Period date accuracy (only for dates within range)
    in_range_results = results_df[results_df['reported_in_range']]
    if len(in_range_results) > 0:
        period_matches = in_range_results['period_match'].sum()
        logger.info(f"Reported dates that match actual period days: {period_matches}/{len(in_range_results)} ({period_matches/len(in_range_results)*100:.1f}%)")
        
        # Days difference statistics (only for dates within range)
        valid_days_diff = in_range_results['days_diff'].dropna()
        if len(valid_days_diff) > 0:
            logger.info(f"Average days difference from closest period: {valid_days_diff.mean():.1f} days")
            logger.info(f"Median days difference: {valid_days_diff.median():.1f} days")
            logger.info(f"Max days difference: {valid_days_diff.max():.0f} days")
    
    # Cycle length accuracy (all subjects)
    valid_cycle_diff = results_df['cycle_length_diff'].dropna()
    if len(valid_cycle_diff) > 0:
        logger.info(f"Average cycle length difference: {valid_cycle_diff.mean():.1f} days")
        logger.info(f"Median cycle length difference: {valid_cycle_diff.median():.1f} days")
        logger.info(f"Max cycle length difference: {valid_cycle_diff.max():.0f} days")
    
    # Show problematic cases
    logger.info("\n=== PROBLEMATIC CASES ===")
    
    # Cases where reported period doesn't match any actual period (only within range)
    if len(in_range_results) > 0:
        mismatched = in_range_results[~in_range_results['period_match']].copy()
        # Only show cases with large differences (>5 days) as these are truly problematic
        large_mismatches = mismatched[mismatched['days_diff'] > 5].copy()
        # Exclude cases where reported_last_period is exactly cycle_length days before the first actual period
        filtered_mismatches = []
        for _, row in large_mismatches.iterrows():
            subject_id = row['subject_id']
            reported_last_period = row['reported_last_period']
            cycle_length = row['reported_cycle_length']
            # Get actual period data for this subject
            subject_periods = period_df[period_df['subject_id'] == subject_id].copy()
            subject_periods = subject_periods.sort_values('date')
            actual_periods = subject_periods[subject_periods['period'] == 'Yes']['date'].tolist()
            if actual_periods:
                first_period = min(actual_periods)
                synthetic_date = first_period - pd.Timedelta(days=cycle_length)
                # If reported_last_period is not the synthetic date, keep as problematic
                if not pd.Timestamp(reported_last_period) == pd.Timestamp(synthetic_date):
                    filtered_mismatches.append(row)
            else:
                filtered_mismatches.append(row)
        if len(filtered_mismatches) > 0:
            logger.info(f"Subjects with large period date mismatches (within range, not synthetic) ({len(filtered_mismatches)}):")
            for row in filtered_mismatches[:10]:
                logger.info(f"  Subject {row['subject_id']}: reported {row['reported_last_period']}, closest actual {row['closest_period']} (diff: {row['days_diff']} days)")
        else:
            logger.info("No subjects with large period date mismatches found (excluding synthetic cases).")
    
    # Cases with large cycle length differences
    large_cycle_diff = results_df[results_df['cycle_length_diff'] > 5]
    if len(large_cycle_diff) > 0:
        logger.info(f"\nSubjects with large cycle length differences ({len(large_cycle_diff)}):")
        for _, row in large_cycle_diff.head(10).iterrows():
            logger.info(f"  Subject {row['subject_id']}: reported {row['reported_cycle_length']}, actual {row['actual_cycle_length']:.1f} (diff: {row['cycle_length_diff']:.1f} days)")
    else:
        logger.info("No subjects with large cycle length differences found.")
    
    # Save detailed results
    results_df.to_csv('output/survey_accuracy_analysis.csv', index=False)
    logger.info(f"\nDetailed results saved to output/survey_accuracy_analysis.csv")
    
    return results_df

if __name__ == "__main__":
    check_survey_accuracy() 