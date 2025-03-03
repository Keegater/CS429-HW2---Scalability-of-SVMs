import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import make_classification as mc
from LinearSVC import LinearSVC


feature_scales = [10, 50, 100, 500, 1000]  # number of features (d)
sample_scales = [500, 1000, 5000, 10000]  # number of samples (n)

results = []


for d in feature_scales:
    for n in sample_scales:

        random_state = 42
        set_range = 100

        # Generate a linearly separable dataset
        X_train, X_test, y_train, y_test, a = mc.make_classification(d, n, set_range, random_state)



        clf = LinearSVC(learning_rate=0.001, epochs=100, reg_param=0.01, random_seed=random_state)


        start_time = time.time()
        clf.fit(X_train, y_train)
        end_time = time.time()

        runtime = end_time - start_time
        results.append({'n': n, 'd': d, 'runtime': runtime})
        print(f"Trained with n={n}, d={d} in {runtime:.4f} seconds.")

# Convert results into a DataFrame for analysis.
df_results = pd.DataFrame(results)
print("\nScalability Results:")
print(df_results)

# Plotting the results: training time vs. number of samples for different feature dimensions.
plt.figure(figsize=(10, 6))
for d in sorted(df_results['d'].unique()):
    df_d = df_results[df_results['d'] == d]
    plt.plot(df_d['n'], df_d['runtime'], marker='o', label=f'd = {d}')

plt.xlabel('Number of Samples (n)')
plt.ylabel('Training Time (seconds)')
plt.title('Scalability of LinearSVC: Training Time vs. Number of Samples')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.show()
