import csv
import numpy as np
import matplotlib.pyplot as plt

class MyLinearRegression():

	def __init__(self, thetas, alpha=0.1, max_iter=10000):
		self.alpha = alpha
		self.max_iter = max_iter
		self.thetas = thetas

	def gradient_(self, x, y):
		x = np.c_[np.ones(len(x)), x]
		gradient = (np.dot(x.T, (np.dot(x, self.thetas) - y))) / len(x)
		return gradient

	def fit_(self, x, y, thetas):
		for i in range(self.max_iter):
			thetas = thetas - self.alpha * self.gradient_(x, y)
		print(f"Thetas at the end are {thetas}")
		return thetas

	def predict_(self, x, thetas):
		x_inter = np.c_[np.ones(len(x)), x]
		y_hat = np.dot(x_inter, thetas)
		return y_hat

def normalize_data(data):
	return (data - np.min(data)) / (np.max(data) - np.min(data))

def denormalize_data(data_normalized, original_data):
	return data_normalized * (np.max(original_data) - np.min(original_data)) + np.min(original_data)

def plot_graphs(x, y, thetas, MLR, x_original, y_original):
	figure, axis = plt.subplots(1, 1)
	axis.scatter(x_original, y_original, label='Actual Data', color='b')

	# Predict with normalized data
	y_hat_normalized = MLR.predict_(x, thetas)

	# Dénormaliser les prédictions
	y_hat = denormalize_data(y_hat_normalized, y_original)

	axis.scatter(x_original, y_hat, label='Predicted Data', color='g')
	axis.plot(x_original, y_hat, ':', color='g')
	axis.set_xlabel('Mileage in km')
	axis.set_ylabel('Price in some currency')
	axis.legend()
	plt.show()

def update_thetas_file(thetas, file_path):
	with open(file_path, 'w', newline='') as f:
		writer = csv.writer(f)
		writer.writerow(['theta0', 'theta1'])
		writer.writerow([thetas[0][0], thetas[1][0]])
	print(f"Thetas updated in file: {file_path}")

if __name__ == '__main__':
	data_path = 'data.csv'
	with open(data_path, 'r') as f:
		reader = csv.reader(f)
		data = list(reader)
	data_array = np.array(data[1:], dtype=float)

	x = data_array[:, 0].reshape(-1, 1)
	y = data_array[:, 1].reshape(-1, 1)

	# Normalisation des données
	x_normalized = normalize_data(x)
	y_normalized = normalize_data(y)

	thetas_path = 'thetas.csv'
	with open(thetas_path, 'r') as f:
		reader = csv.reader(f)
		data = list(reader)
	data_array = np.array(data[1:], dtype=float)
	thetas = np.array([[0], [0]])
	print(f"Thetas at the beginning are {thetas}")

	MLR = MyLinearRegression(thetas)
	thetas = MLR.fit_(x_normalized, y_normalized, thetas)

	# Update thetas.csv file
	update_thetas_file(thetas, thetas_path)

	# Plot results
	plot_graphs(x_normalized, y_normalized, thetas, MLR, x, y)
