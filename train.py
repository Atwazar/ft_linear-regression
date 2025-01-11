import csv
import numpy as np
import matplotlib.pyplot as plt
import os

class MyLinearRegression():
	def __init__(self, thetas, alpha=0.1, max_iter=10000):
		self.alpha = alpha
		self.max_iter = max_iter
		self.thetas = thetas

	def gradient_(self, x, y):
		x = np.c_[np.ones(len(x)), x]
		gradient = (np.dot(x.T, (np.dot(x, self.thetas) - y))) / len(x)
		return gradient

	def fit_(self, x, y):
		self.thetas = thetas
		for i in range(self.max_iter):
			self.thetas = self.thetas - self.alpha * self.gradient_(x, y)
		return self.thetas

	def predict_(self, x, thetas):
		x_inter = np.c_[np.ones(len(x)), x]
		y_hat = np.dot(x_inter, thetas)
		return y_hat

	def score_(self, y_true, y_pred):
		ss_res = np.sum((y_true - y_pred) ** 2)
		ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
		r2 = 1 - (ss_res / ss_tot)

		mse = np.mean((y_true - y_pred) ** 2)

		rmse = np.sqrt(mse)

		mae = np.mean(np.abs(y_true - y_pred))

		return {
			'R2': r2,
			'MSE': mse,
			'RMSE': rmse,
			'MAE': mae
		}

def normalize_data(data):
	return (data - np.min(data)) / (np.max(data) - np.min(data))

def denormalize_data(data_normalized, original_data):
	return data_normalized * (np.max(original_data) - np.min(original_data)) + np.min(original_data)

def denormalize_thetas(thetas, x_original, y_original):
	x_min, x_max = np.min(x_original), np.max(x_original)
	y_min, y_max = np.min(y_original), np.max(y_original)

	theta1_normalized = thetas[1][0]
	theta0_normalized = thetas[0][0]

	theta1_denormalized = theta1_normalized * (y_max - y_min) / (x_max - x_min)
	theta0_denormalized = theta0_normalized * (y_max - y_min) + y_min - theta1_denormalized * x_min

	return np.array([[theta0_denormalized], [theta1_denormalized]])


def plot_graphs(thetas, MLR, x_original, y_original):
	figure, axis = plt.subplots(figsize=(10, 6))

	y_hat = MLR.predict_(x_original, thetas)

	metrics = MLR.score_(y_original, y_hat)

	axis.scatter(x_original, y_original, label='Actual Data', color='blue', alpha=0.5)
	axis.scatter(x_original, y_hat, label='Predicted Data', color='green', alpha=0.5)
	axis.plot(x_original, y_hat, ':', color='green', label='Regression Line')

	metrics_text = f'Metrics:\nR² = {metrics["R2"]:.4f}\nRMSE = {metrics["RMSE"]:.2f}\nMAE = {metrics["MAE"]:.2f}'

	box_props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
	axis.text(0.95, 0.95, metrics_text,
				transform=axis.transAxes,
				fontsize=10,
				verticalalignment='top',
				horizontalalignment='right',
				bbox=box_props)

	axis.set_xlabel('Mileage (km)')
	axis.set_ylabel('Price')
	axis.set_title('Linear Regression: Price vs Mileage')
	axis.grid(True, linestyle='--', alpha=0.7)
	axis.legend()

	plt.tight_layout()

	plt.show()

def update_thetas_file(thetas, file_path):
	with open(file_path, 'w', newline='') as f:
		writer = csv.writer(f)
		writer.writerow(['theta0', 'theta1'])
		writer.writerow([thetas[0][0], thetas[1][0]])
	print(f"Thetas updated in file: {file_path}")

if __name__ == '__main__':
	data_path = 'data.csv'
	thetas_path = 'thetas.csv'

	if not os.path.exists(data_path):
		print(f"Broken, broken, and broken again, '{data_path}' doesn't exist")
		exit(1)
	try:
		with open(data_path, 'r') as f:
			reader = csv.reader(f)
			data = list(reader)
		if len(data) < 2 or len(data[1]) < 2:
			raise ValueError(f"Datas is messed up in '{data_path}'")
		data_array = np.array(data[1:], dtype=float)
	except (OSError, ValueError, IndexError) as e:
		print(f"Please have mercy: {e}")
		exit(1)

	x = data_array[:, 0].reshape(-1, 1)
	y = data_array[:, 1].reshape(-1, 1)

	x_normalized = normalize_data(x)
	y_normalized = normalize_data(y)

	thetas = np.array([[0], [0]])

	MLR = MyLinearRegression(thetas)
	thetas = MLR.fit_(x_normalized, y_normalized)
	thetas_without_normalisation = MLR.fit_(x, y)
	thetas_denormalized = denormalize_thetas(thetas, x, y)
	update_thetas_file(thetas_denormalized, thetas_path)

	plot_graphs(thetas_denormalized, MLR, x, y)
