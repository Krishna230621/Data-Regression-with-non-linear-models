import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import io
import matplotlib.pyplot as plt
import scipy.stats as stats

# --- 1. Your Data ---
data_string = """
   1     2.65245295
   2     2.36739230
   3     1.84622991
   4     1.66815186
   5     1.08305144
   6     1.21724069
   7     0.79019010
   8     0.86625761
   9     0.88458228
  10     0.59665191
  11     0.76315606
  12     0.75276375
  13     0.61360216
  14     0.70784199
  15     0.69185406
  16     0.80202466
  17     0.80791056
  18     0.49587724
  19     0.52140725
  20     0.63804811
  21     0.55479097
  22     0.39703095
  23     0.43296981
  24     0.42494625
  25     0.87369204
  26     0.49509621
  27     0.56581318
  28     0.40268764
  29     0.52908576
  30     0.50621241
  31     0.71439928
  32     0.45285442
  33     0.72239810
  34     0.52219051
  35     0.74110603
  36     0.36722380
  37     0.55493581
  38     0.40376833
  39     0.38991120
  40     0.50303900
  41     0.73440540
  42     0.44264227
  43     0.37080035
  44     0.41760144
  45     0.41026187
  46     0.54957324
  47     0.45775944
  48     0.45890796
  49     0.55433273
  50     0.38786775
"""
df = pd.read_csv(io.StringIO(data_string.strip()), sep='\s+', header=None, names=['t', 'y'])
t = df['t'].values
y = df['y'].values
n = len(y) # Number of data points

# Create a finer range of t for plotting smooth curves
t_plot = np.linspace(t.min(), t.max(), 200)

# --- 2. Define Model Functions ---

# Model 1: y(t) = a0 + a1*exp(b1*t) + a2*exp(b2*t)
def model_1(t_val, a0, a1, b1, a2, b2):
    return a0 + a1 * np.exp(b1 * t_val) + a2 * np.exp(b2 * t_val)

# Model 2: y(t) = (a0 + a1*t) / (b0 + b1*t)
def model_2(t_val, a0, a1, b0, b1):
    denominator = b0 + b1 * t_val
    return (a0 + a1 * t_val) / (denominator + 1e-9)

# Model 3: y(t) = B0 + B1*t + B2*t^2 + B3*t^3 + B4*t^4
# This is handled by LinearRegression + PolynomialFeatures

# --- 3. Fit All Three Models ---

print("--- Fitting All Models ---")

# Fit Model 1 (Exponential)
p_model_1 = 5 # Number of parameters for Model 1
# Using your preferred initial guesses
#order of parameters: a0, a1, b1, a2, b2
p0_model_1 = [1, 0.4, -0.05, 0, 0] 
try:
    params_model_1, cov_model_1 = curve_fit(model_1, t, y, p0=p0_model_1, maxfev=5000)
    y_fit_model_1 = model_1(t_plot, *params_model_1)
    print("Model 1 fitted successfully.")
    #print all fitted parameters
    print("Fitted parameters for Model 1:")
    for i, param in enumerate(params_model_1):
        print(f"  Param {i}: {param}")
except RuntimeError as e:
    print(f"Model 1 fitting failed: {e}")
    y_fit_model_1 = np.full_like(t_plot, np.nan)
    # Add dummy values so code doesn't crash later
    params_model_1 = np.full(p_model_1, np.nan)
    cov_model_1 = np.full((p_model_1, p_model_1), np.nan)


# Fit Model 2 (Rational)
#order of
p0_model_2 = [3.0, 1.0, 1.0, 2.0] # Informed guesses
try:
    params_model_2, _ = curve_fit(model_2, t, y, p0=p0_model_2, maxfev=5000)
    y_fit_model_2 = model_2(t_plot, *params_model_2)
    print("Model 2 fitted successfully.")
    #print all fitted parameters
    print("Fitted parameters for Model 2:")
    for i, param in enumerate(params_model_2):
        print(f"  Param {i}: {param}")
except RuntimeError as e:
    print(f"Model 2 fitting failed: {e}")
    y_fit_model_2 = np.full_like(t_plot, np.nan)

# Fit Model 3 (Polynomial)
poly_features = PolynomialFeatures(degree=4, include_bias=True)
t_poly = poly_features.fit_transform(t.reshape(-1, 1))
model_3_lin_reg = LinearRegression(fit_intercept=False)
model_3_lin_reg.fit(t_poly, y)
t_plot_poly = poly_features.transform(t_plot.reshape(-1, 1))
y_fit_model_3 = model_3_lin_reg.predict(t_plot_poly)
print("Model 3 fitted successfully.")


# --- 4. Generate Plot 1: All Fitted Models ---

print("\nGenerating Plot 1: All Models Fit...")
plt.figure(figsize=(10, 6)) # <--- Creates the first figure
plt.scatter(t, y, label='Observed Data', color='blue', alpha=0.7)
plt.plot(t_plot, y_fit_model_1, label='Fitted Model 1 (Exponential)', color='green', linestyle='--')
plt.plot(t_plot, y_fit_model_2, label='Fitted Model 2 (Rational)', color='purple', linestyle=':')
plt.plot(t_plot, y_fit_model_3, label='Fitted Model 3 (Polynomial)', color='red', linestyle='-')

plt.title("Observed Data with All Three Fitted Models")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.grid(True)
plt.legend()
plt.ylim(ymin=0) 
plt.tight_layout()
# plt.show() can sometimes cause issues before savefig, so I've commented it.
# Save the first plot
#save to nice 
plt.savefig("all_models_plot.png")
print("Plot saved to 'all_models_plot.png'")


# --- 5. Generate Plot 2: Diagnostic Plots for Model 1 ---

print("\nGenerating Plot 2: Model 1 Diagnostics...")

# Calculate Residuals for Model 1
y_fitted_model_1 = model_1(t, *params_model_1)
residuals_model_1 = y - y_fitted_model_1

plt.figure(figsize=(10, 5)) # <--- Creates the second, separate figure

# Plot 1: Residuals vs. Fitted Values
plt.subplot(1, 2, 1)
plt.scatter(y_fitted_model_1, residuals_model_1, alpha=0.7)
plt.axhline(y=0, color='red', linestyle='--')
plt.title("Plot 1: Residuals vs. Fitted Values")
plt.xlabel("Fitted y-values (Model 1)")
plt.ylabel("Residuals (y - y_fitted)")
plt.grid(True)

# Plot 2: Q-Q Plot for Normality
plt.subplot(1, 2, 2)
stats.probplot(residuals_model_1, dist="norm", plot=plt)
plt.title("Plot 2: Q-Q Plot of Residuals")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")
plt.grid(True)

plt.tight_layout()
 # Same as before, commenting plt.show() to ensure savefig works

# *** ADDED THIS MISSING LINE ***
plt.savefig("diagnostic_plots.png")
print("Diagnostic plots saved to 'diagnostic_plots.png'")


# --- 6. Calculate Confidence Intervals (Question 5) ---


print("\n--- Question 5: 95% Confidence Intervals for Model 1 ---")

# Calculate Standard Errors
# variance = diagonal elements of the covariance matrix
try:
    variances = np.diag(cov_model_1)
    # std_error = sqrt(variance)
    std_errors = np.sqrt(variances)
    
    # Get the t-critical value
    dof = n - p_model_1  # degrees of freedom (50 - 5 = 45)
    alpha = 0.05
    t_crit = stats.t.ppf(1 - alpha/2, dof) # t-value for 95% CI
    print(f"Degrees of Freedom (n-p): {dof}")
    print(f"t-critical value (t_0.025, 45): {t_crit:.4f}")

    # Define parameter names for output
    param_names = ['a0', 'a1', 'b1', 'a2', 'b2']

    print("\nParameter Confidence Intervals:")
    print("----------------------------------------------------------")
    print(f"{'Parameter':<10} | {'Estimate':<10} | {'Std Error':<10} | {'95% CI Lower':<12} | {'95% CI Upper':<12}")
    print("----------------------------------------------------------")

    for i in range(p_model_1):
        param = params_model_1[i]
        se = std_errors[i]
        
        # Calculate confidence interval
        lower_bound = param - t_crit * se
        upper_bound = param + t_crit * se
        
        print(f"{param_names[i]:<10} | {param:<10.4f} | {se:<10.4f} | {lower_bound:<12.4f} | {upper_bound:<12.4f}")

    print("----------------------------------------------------------")

except Exception as e:
    print(f"\nCould not calculate confidence intervals. Error: {e}")
    print("This often happens if the model fit failed or was unstable.")