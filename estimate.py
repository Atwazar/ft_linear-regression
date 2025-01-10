import csv
import numpy as np


if __name__ == '__main__':
	path = 'thetas.csv'
	with open(path, 'r') as f:
		reader = csv.reader(f)
	data = list(reader)
	data_array = np.array(data[1:], dtype = float)
	theta1, theta2 = data_array[1][0], data_array[1][1]
	mileage = input("Enter a mileage: ")
	# if not isinstance(mileage, int):
	# 	print("Please enter a correct value, float only are accepted")
	# else:
	price = theta1 * mileage + theta2
	print(f"for a mileage of {mileage}km, the price estimated is {price}.")

##add some prints
