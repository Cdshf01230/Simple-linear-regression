{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "3050fe2b-48de-4ed7-8d43-958a147dc272",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import copy\n",
    "import math"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "e34de2e5-1848-499e-a90f-ae982c00cb2c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "        price  area  bedrooms  bathrooms  stories mainroad guestroom basement  \\\n",
      "0    13300000  7420         4          2        3      yes        no       no   \n",
      "1    12250000  8960         4          4        4      yes        no       no   \n",
      "2    12250000  9960         3          2        2      yes        no      yes   \n",
      "3    12215000  7500         4          2        2      yes        no      yes   \n",
      "4    11410000  7420         4          1        2      yes       yes      yes   \n",
      "..        ...   ...       ...        ...      ...      ...       ...      ...   \n",
      "540   1820000  3000         2          1        1      yes        no      yes   \n",
      "541   1767150  2400         3          1        1       no        no       no   \n",
      "542   1750000  3620         2          1        1      yes        no       no   \n",
      "543   1750000  2910         3          1        1       no        no       no   \n",
      "544   1750000  3850         3          1        2      yes        no       no   \n",
      "\n",
      "    hotwaterheating airconditioning  parking prefarea furnishingstatus  none  \n",
      "0                no             yes        2      yes        furnished     1  \n",
      "1                no             yes        3       no        furnished     1  \n",
      "2                no              no        2      yes   semi-furnished     1  \n",
      "3                no             yes        3      yes        furnished     1  \n",
      "4                no             yes        2       no        furnished     1  \n",
      "..              ...             ...      ...      ...              ...   ...  \n",
      "540              no              no        2       no      unfurnished     1  \n",
      "541              no              no        0       no   semi-furnished     1  \n",
      "542              no              no        0       no      unfurnished     1  \n",
      "543              no              no        0       no        furnished     1  \n",
      "544              no              no        0       no      unfurnished     1  \n",
      "\n",
      "[545 rows x 14 columns]\n",
      "[[13300000 7420 4 ... 'yes' 'furnished' 1]\n",
      " [12250000 8960 4 ... 'no' 'furnished' 1]\n",
      " [12250000 9960 3 ... 'yes' 'semi-furnished' 1]\n",
      " ...\n",
      " [1750000 3620 2 ... 'no' 'unfurnished' 1]\n",
      " [1750000 2910 3 ... 'no' 'furnished' 1]\n",
      " [1750000 3850 3 ... 'no' 'unfurnished' 1]]\n"
     ]
    }
   ],
   "source": [
    "#nhap du lieu va bo sung\n",
    "path = './Housing.csv'\n",
    "data = pd.read_csv(path)\n",
    "data = data.assign(none = [1 for x in range(545)])\n",
    "print(data)\n",
    "data = np.array(data)\n",
    "print(data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "3323d297-74d9-4d27-b778-e24d376ae826",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "class Linear_regression():\n",
    "    def __init__(self, data):\n",
    "        self.data = data\n",
    "    #Xu ly du lieu\n",
    "    def data_process(self):\n",
    "        def processor_1(r):\n",
    "            r = str(r)\n",
    "            data_dict = {'yes' : 1, 'no' : 0, 'furnished' : 1, 'unfurnished' : 0, 'semi-furnished': 0.5}\n",
    "            if r in ['yes', 'no', 'furnished', 'unfurnished', 'semi-furnished']:\n",
    "                return float(data_dict.get(r))\n",
    "            else:\n",
    "                return float(r)\n",
    "        processor1 = np.vectorize(processor_1)\n",
    "        self.data = processor1(self.data)\n",
    "    #Linear\n",
    "    def LN(self, c, k, n = 100):\n",
    "        c = float(c)\n",
    "        n = int(n)\n",
    "        #Tim toi uu\n",
    "        result = list([])\n",
    "        for i in range(n):\n",
    "            #lay du lieu ngau nhien tu file\n",
    "            np.random.shuffle(self.data)\n",
    "            #tao ma tran X, Y, XT\n",
    "            X = np.matrix(self.data[:29, 1:])\n",
    "            X_transpose = np.transpose(X)\n",
    "            Y = np.transpose(np.matrix(self.data[:29, 0]))\n",
    "            #tinh ma tran trong so\n",
    "            E = np.multiply(np.eye(13), c)\n",
    "            D = np.dot(X_transpose, X) + E\n",
    "            Z = np.linalg.pinv(D)\n",
    "            theta = np.dot(np.dot(Z, X_transpose), Y)\n",
    "            #tinh trung binh sai so\n",
    "            test_input = np.matrix(self.data[29: , 1:])\n",
    "            result_input = np.matrix(self.data[29:, 0])\n",
    "            result_output = np.dot(test_input, theta)\n",
    "            error = np.subtract(result_output, result_input)\n",
    "            mean_square_error = np.mean(np.square(error))\n",
    "            result.append([theta, mean_square_error])\n",
    "        final_theta = min(result, key = lambda x: x[1])\n",
    "        print(final_theta[0].astype(np.int32))\n",
    "        k = np.array(k).reshape(1, 13)\n",
    "        k = np.matrix(k)\n",
    "        print(np.dot(k, final_theta[0]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "bb322db5-d92c-4bc1-9732-448da640b8e8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 842]\n",
      " [6818]\n",
      " [3656]\n",
      " [4442]\n",
      " [1346]\n",
      " [ 162]\n",
      " [ 582]\n",
      " [-422]\n",
      " [2197]\n",
      " [ 581]\n",
      " [1028]\n",
      " [1443]\n",
      " [1623]]\n",
      "[[6310583.07904943]]\n"
     ]
    }
   ],
   "source": [
    "x = Linear_regression(data)\n",
    "k = [7420 , 4 , 2 , 3 , 1 , 0 , 0 , 0 , 1 , 2 , 1 , 1, 1]\n",
    "x.data_process()\n",
    "x.LN(10000, k)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2d254be9-9202-44d3-92ff-ce6a4ced080a",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0cd0b374-71cb-4ea0-b323-0324931d1e4a",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "33bbbaa1-2e11-479b-8f16-4c23e438c4fd",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
