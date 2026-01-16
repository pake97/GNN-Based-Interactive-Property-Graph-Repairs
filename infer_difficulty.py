import numpy as np
import pandas as pd
from scipy.optimize import minimize

# -------------------------------
# STEP 1: Define user abilities
# -------------------------------
user_abilities = {
    0: {'K': 0.9, 'S': 0.2},
    1: {'K': 0.5, 'S': 0.5},
    2: {'K': 0.1, 'S': 0.9},
    3: {'K': 0.7, 'S': 0.3},
    4: {'K': 0.3, 'S': 0.7}
}

# Combine K and S into a single ability metric for each user
user_total_ability = np.array([user_abilities[u]['K'] + user_abilities[u]['S'] for u in user_abilities])

# -------------------------------
# STEP 2: Define outcome matrix
# Rows = users, Columns = violations
# 1 = success, 0 = failure
# -------------------------------
outcome_matrix = np.array([
    [1, 1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 0],
    [1, 0, 0, 1, 0, 1],
    [0, 1, 1, 0, 1, 0]
])

num_users, num_items = outcome_matrix.shape

# -------------------------------
# STEP 3: Define negative log-likelihood function
# -------------------------------
def neg_log_likelihood(difficulties):
    # Compute probabilities using logistic model
    prob_matrix = 1 / (1 + np.exp(-(user_total_ability[:, None] - difficulties[None, :])))
    # Clip to avoid log(0)
    prob_matrix = np.clip(prob_matrix, 1e-5, 1 - 1e-5)
    # Compute log likelihood
    log_likelihood = outcome_matrix * np.log(prob_matrix) + (1 - outcome_matrix) * np.log(1 - prob_matrix)
    return -np.sum(log_likelihood)

# -------------------------------
# STEP 4: Optimize to find difficulties
# -------------------------------
initial_difficulties = np.zeros(num_items)
result = minimize(neg_log_likelihood, initial_difficulties, method='BFGS')
estimated_difficulties = result.x

# -------------------------------
# STEP 5: Output results
# -------------------------------
difficulty_df = pd.DataFrame({
    'Violation': [f'V{i+1}' for i in range(num_items)],
    'Estimated_Difficulty': estimated_difficulties
})

print(difficulty_df)




