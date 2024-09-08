import numpy as np
import pandas as pd
import sympy as sp
import networkx as nx
import matplotlib.pyplot as plt

def generate_synthetic_data(seed = 42, n_samples = 150000):
    data = {
        'is_senior': np.random.binomial(1, 0.25, n_samples),
        'sale_calls': np.random.poisson(3, n_samples),
        'bugs_faced': np.random.poisson(2, n_samples),
        'monthly_usage': np.random.poisson(10, n_samples),
        # 'consumer_trust': np.round(np.random.triangular(0, 4, 5, n_samples)),
        # 'consumer_trust': np.round(skewnorm.rvs(4, size=n_samples, loc=4, scale=1)),
        'consumer_trust': np.round(1 + 4 * np.random.beta(3, 2, n_samples)).astype(int),
        }

    df = pd.DataFrame(data)

    df['interaction'] = np.where(
        df['is_senior'] == 0,
        np.exp(-0.5 * df['sale_calls']),
        1 - np.exp(-0.3 * df['sale_calls']))

    # discount dependant on bugs_faced and monthly_usage with some noise
    df['discount'] = (0.5 * df['bugs_faced'] + 0.3 * df['monthly_usage'] + np.random.normal(0, 1, n_samples)).astype(int)
    df['discount'] = df['discount'].clip(0, 4)  # discount is between 0 and 4

    # renewal dependant on interaction, bugs_faced, monthly_usage, consumer_trust discount
    logit_renewal = (
        df['interaction'] 
        - 0.5 * df['bugs_faced'] 
        + 0.3 * df['monthly_usage'] 
        + 0.2 * df['consumer_trust']
        - 0.4 * df['discount']
        + np.random.normal(0, 1, n_samples)
    )
    # df['renewal'] = (logit_renewal > 0).astype(int)
    # categorize renewel to true/false boolean type
    # df['renewal'] = df['renewal'].astype('bool')
    df['renewal'] = (logit_renewal > 0).astype(int)


    return df