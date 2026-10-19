
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Generate reproducible data
rng = np.random.default_rng(1)
x = rng.uniform(0, 10, 60)
y = 2.5 * x + 4 + rng.normal(0, 2, 60)

# 2. Prepare the feature matrix
X = x.reshape(-1, 1)

# 3. Train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# 4. Get predictions for drawing the regression line
x_line = np.linspace(0, 10, 200)
X_line = x_line.reshape(-1, 1)
y_line = model.predict(X_line)

# 5. Calculate model performance
r2 = model.score(X, y)

print(f"Learned equation: y = {model.coef_[0]:.2f}x + "
      f"{model.intercept_:.2f}")
print(f"R-squared: {r2:.3f}")

# 6. Plot the observed data points
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color="steelblue", label="Observed data",
            alpha=0.8)

# 7. Plot the fitted regression line
plt.plot(x_line, y_line, color="darkorange", linewidth=2.5,
         label="Learned regression line")

# 8. Show residuals for a few observations
y_pred = model.predict(X)
for i in range(0, len(x), 5):
    plt.plot([x[i], x[i]], [y[i], y_pred[i]],
             color="gray", alpha=0.6, linestyle="--")

# 9. Add equation and R-squared to the graph
equation = (
    f"y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}\n"
    f"R-squared = {r2:.3f}"
)
plt.text(0.05, 0.95, equation, transform=plt.gca().transAxes,
         verticalalignment="top",
         bbox=dict(facecolor="white", alpha=0.85))

# 10. Format and display the plot
plt.title("Linear Regression: Fitting a Straight Line")
plt.xlabel("Input feature (x)")
plt.ylabel("Target value (y)")
plt.legend()
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.show()
