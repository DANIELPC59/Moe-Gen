import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
import logging
def get_logger(filename):
    # Create logger
    logger = logging.getLogger("Experiment_logger")
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate handlers (common issue in Jupyter)
    if logger.hasHandlers():
        logger.handlers.clear()

    # 1. Formatter: includes time and detailed information
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # 2. File Handler: writes to file
    file_handler = logging.FileHandler(filename, mode='a', encoding='utf-8') # mode='a' means append mode
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# Init logger
log_file_name='/home/xuke/zpc/Code/Moe_Gen/CenterInject_Explore/Distribution/AWF_CV.log'
logger = get_logger(log_file_name)

def get_valid_upstream_data(x_seq, adv_seq):
    """
    Helper function: Extract all valid upstream bursts (>0) from a single sample
    Returns: Original upstream sequence, perturbed upstream sequence, perturbation residual (delta)
    """
    valid_mask = x_seq > 0  
    u_x = x_seq[valid_mask]
    u_adv = adv_seq[valid_mask]
    u_delta = u_adv - u_x
    return u_x, u_adv, u_delta

def evaluate_scheme_a_micro(data_x, data_adv, label_y):
    """
    Calculate CV (Coefficient of Variation) and Gini (Gini Coefficient)
    """
    X = data_x.squeeze()
    X_adv = data_adv.squeeze()
    unique_labels = np.unique(label_y)
    results = []

    for label in unique_labels:
        idx = (label_y == label)
        class_x, class_adv = X[idx], X_adv[idx]
        
        cv_list, gini_list = [], []
        
        for i in range(len(class_x)):
            _, _, delta = get_valid_upstream_data(class_x[i], class_adv[i])
            
            if len(delta) > 0 and np.mean(delta) > 0:
                
                cv = np.std(delta) / np.mean(delta)
                cv_list.append(cv)
               
                gini_list.append(calculate_gini(delta))
        
        results.append({
            'Label': label,
            'Avg_CV': np.mean(cv_list) if cv_list else 0,
            'Avg_Gini': np.mean(gini_list) if gini_list else 0
        })
    
    return pd.DataFrame(results)

def calculate_gini(x):
    x = np.sort(np.abs(x))
    n = len(x)
    if n == 0 or np.sum(x) == 0: return 0
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * x)) / (n * np.sum(x))

def evaluate_scheme_b_macro(data_x, data_adv, label_y):
    """
    Calculate Wasserstein distance (EMD)
    """
    X = data_x.squeeze()
    X_adv = data_adv.squeeze()
    unique_labels = np.unique(label_y)
    results = []

    for label in unique_labels:
        idx = (label_y == label)
        class_x, class_adv = X[idx], X_adv[idx]
        
      
        pool_x, pool_adv = [], []
        
        for i in range(len(class_x)):
            ux, uadv, _ = get_valid_upstream_data(class_x[i], class_adv[i])
            pool_x.extend(ux)
            pool_adv.extend(uadv)
            
        # Calculate Wasserstein 
        if pool_x and pool_adv:
            wd_score = wasserstein_distance(pool_x, pool_adv)
        else:
            wd_score = 0
            
        results.append({
            'Label': label,
            'EMD_Score': wd_score
        })
        
    return pd.DataFrame(results)

def run_full_evaluation(data_x, data_adv, label_y, method_name):
    print(f'=======>{method_name}<======')
    logger.info(f'=======>{method_name}<======')
    print("--- Executing Scheme A (Micro Analysis: Perturbation Dispersion) ---")
    logger.info("--- Executing Scheme A (Micro Analysis: Perturbation Dispersion) ---")
    df_a = evaluate_scheme_a_micro(data_x, data_adv, label_y)
    
    print("--- Executing Scheme B (Macro Analysis: Statistical Consistency) ---")
    logger.info("--- Executing Scheme B (Macro Analysis: Statistical Consistency) ---")
    df_b = evaluate_scheme_b_macro(data_x, data_adv, label_y)
    
    # Merge results
    final_report = pd.merge(df_a, df_b, on='Label')
    
    print("\n" + "="*50)
    print(f"{method_name} Stealth Detailed Evaluation Report")
    print(final_report.to_string(index=False))
    print("="*50)
    
    
    # --- [Added] Calculate maximum CV and its corresponding label for easy problem localization ---
    max_cv_val = final_report['Avg_CV'].max()
    # Find the Label corresponding to the maximum CV, add iloc[0] to prevent errors when there are multiple maximum values
    max_cv_label = final_report.loc[final_report['Avg_CV'] == max_cv_val, 'Label'].values[0]
    
    # Output academic summary
    print(f"\n[Academic Conclusion Summary]")
    print(f"1. Average CV: {final_report['Avg_CV'].mean():.4f} (measures local clustering degree)")
    print(f"2. Maximum CV:  {max_cv_val:.4f} [Label: {max_cv_label}] (measures fluctuation in worst case)") # <--- Added this line
    print(f"3. Average EMD: {final_report['EMD_Score'].mean():.4f} (measures global distribution deviation)")
    
    logger.info("\n" + "="*50)
    logger.info(f"{method_name} Stealth Detailed Evaluation Report")
    logger.info(final_report.to_string(index=False))
    logger.info("="*50)
    
    # Output academic summary
    logger.info(f"\n[Academic Conclusion Summary]")
    logger.info(f"1. Average CV: {final_report['Avg_CV'].mean():.4f} (measures local clustering degree)")
    logger.info(f"2. Maximum CV:  {max_cv_val:.4f} [Label: {max_cv_label}] (measures fluctuation in worst case)") # <--- Added this line
    logger.info(f"3. Average EMD: {final_report['EMD_Score'].mean():.4f} (measures global distribution deviation)")
    return final_report

# final_df = run_full_evaluation(data_x, data_adv, label_y)