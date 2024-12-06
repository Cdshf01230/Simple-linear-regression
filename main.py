import numpy as np
import pandas as pd
class Houseprice:
    def __init__(self, filepath):
        with open(filepath) as d:
            self.data = np.array(pd.read_csv(d))
    def House_price(self, k, n = 10):
        #xu ly du lieu:
        def data_processing(d):
            d = str(d)
            data_map = {'no': 0, 'yes' : 1, 'furnished' : 1, 'semi-furnished': 0.5, 'unfurnished': 0}
            if d in ['no', 'yes', 'furnished', 'semi-furnished', 'unfurnished']:
                return float(data_map.get(d))
            else:
                return float(d)
        processor = np.vectorize(data_processing)
        self.data = processor(self.data)
        #vong lap de tim ket qua toi uu
        result = list([])
        for i in range(n):
            #lay du lieu ngau nhien tu file
            np.random.shuffle(self.data)
            #tao ma tran X, Y, XT
            X = np.matrix(self.data[:29, 1:])
            X_transpose = np.transpose(X)
            Y = np.transpose(np.matrix(self.data[:29, 0]))
            #tinh ma tran trong so
            Z = np.linalg.pinv(np.dot(X_transpose, X))
            theta = np.dot(np.dot(Z, X_transpose), Y)
            #tinh trung binh sai so
            test_input = np.matrix(self.data[29: , 1:])
            result_input = np.matrix(self.data[29:, 0])
            result_output = np.dot(test_input, theta)
            error = np.subtract(result_output, result_input)
            mean_square_error = np.mean(np.square(error))
            result.append([theta, mean_square_error])
        final_theta = min(result, key = lambda x: x[1])
        k = np.array(k).reshape(1, 12)
        k = np.matrix(k)
        print(np.dot(k, final_theta[0]))
a = 'C:\\Users\Admin\Saved Games\VSC\Housing.csv'
b = Houseprice(a)
k = [7420 , 4 , 2 , 3 , 1 , 0 , 0 , 0 , 1 , 2 , 1 ,1]
b.House_price(k = k, n = 100)
