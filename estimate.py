import os
import csv
import numpy as np

if __name__ == '__main__':
	path = 'thetas.csv'

	if not os.path.exists(path):
		print(f"What have you done with the file '{path}'? it was right there !")
	else:
		try:
			with open(path, 'r') as f:
				reader = csv.reader(f)
				data = list(reader)

			if len(data) < 2 or len(data[1]) < 2:
				raise ValueError("Ok, well now the 'thetas.csv' file is broken, well done... Or maybe it's just the first time you launch the program, who knows... In that case, train the model first")

			data_array = np.array(data[1:], dtype=float)
			theta1, theta2 = data_array[0][1], data_array[0][0]

			while True:
				mileage = input("Enter a mileage: ")
				try:
					mileage = float(mileage)
					if (mileage < 0):
						print("I mean... you know this can't be right ? right ??")
					else:
						break
				except ValueError:
					print("Please enter a correct value, only numbers are accepted.")

			price = theta1 * mileage + theta2
			if (price < 0):
				print(f"Come on, you know this can't be right")
			else:
				print(f"For a mileage of {mileage} km, the estimated price is {price:.2f}.")
		except (OSError, ValueError, IndexError) as e:
			print(f"Ahem, we encountered some weird error parsing the file: {e}")


