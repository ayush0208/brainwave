#   importing the libraries

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.python.client import device_lib 
import keras
import time
from tensorflow.keras import layers
from tensorflow.python.client import device_lib 
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from numpy.random import seed
seed(42)  # keras seed fixing
import tensorflow as tf
tf.random.set_seed(42)  # tensorflow seed fixing
np.set_printoptions(threshold=np.inf)

# parsing the dataset

IN_Brain_location = {"AF3" : 0,
                  "AF4"  : 1,
                  "T7"  : 2,
                  "T8" : 3,
                  "PZ"  : 4}
MW_Brain_location = {"FP1" : 0}
N_Samples_device = {"MindWave" : 1024,
          "IN"     : 256 
          }
def Load_data(infile, device, data_size, event_start_point = 0, split_data = False):
    if device == "IN":
        Brain_location = IN_Brain_location
    elif device == "MindWave":
        Brain_location = MW_Brain_location
    N_locations = len(Brain_location)
    test_size = int(0.2 * data_size)
    train_size = data_size - test_size
    N_data = N_Samples_device[device]
    if N_locations > 1:
        arr = np.zeros([data_size, N_locations, N_data])
    else:
        arr = np.zeros([data_size, N_data], dtype = 'int32')
    label = np.zeros([data_size], dtype = 'int32')
    for i in range(N_locations*data_size):
        
        temp = infile.readline()
        if len(temp) < 10:
            break
        x = temp.split()
        header = x[0:6]
        event = int(header[1]) - event_start_point
        channel = Brain_location[header[3]]
        temp = x[6].split(',')
        while len(temp) < N_data:
            temp.append('0')
        n = int(header[4])
        if N_locations > 1:
            arr[i//N_locations][channel] = list(map(float,temp))[:N_data]
        else:
            arr[i//N_locations] = list(map(float,temp))[:N_data]
        label[i//N_locations] = n
    if split_data == True:
        return arr[0:train_size], label[0:train_size], arr[train_size:data_size], label[train_size:data_size]
    return arr, label    

infile = open("MW.txt")
data, label = Load_data(infile,"MindWave",data_size=10000)
print(data.shape)
print(label.shape)

#   one hot encoding of label vectors
def one_hot(y_):
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y_ = y_.reshape(len(y_))
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]

label=one_hot(label)
print(label.shape)

#   splitting the dataset into training and testing data in the ratio of 80:20
from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(data, label, test_size=0.20, shuffle= True, random_state = 31)
x_train=x_train.reshape(-1,32,32,1)
x_valid=x_valid.reshape(-1,32,32,1)
print(x_train.shape)
print(y_train.shape)

#loading the saved model
loaded_model = keras.models.load_model("model.h5")
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(x_valid,y_valid, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

#    41414092616802021212818706161631313083132  -->  i want a cup of tea        test1.txt
#    841431309313284143137171704141471717170616414142121231308616163291919  - the weather is nice today  test2.txt
#  4141405151541414515313086 1 6160751515291919 - i like to play      test3.txt
#   515156161681818313086161603261621212313 - love to dance       test4.txt
#   691919041461616212212919190414147171717086161607171731323021261616616165157171717 - my hobby is to read books test5.txt

testfile = open("test1.txt")
test_data, test_label = Load_data(testfile,"MindWave",data_size=41)
print(test_data.shape)
print(test_label.shape)
test_data=test_data.reshape(-1,32,32,1)

pred = loaded_model.predict(test_data)
pred_digits = np.argmax(pred , axis = 1)
pred_digits
s=""
for item in pred_digits.astype(str):
  s+=item

class Trie():

    def __init__(self):
        self._end = '*'
        self.trie = dict()

    def __repr__(self):
        return repr(self.trie)

    def make_trie(self, *words):
        trie = dict()
        for word in words:
            temp_dict = trie
            for letter in word:
                temp_dict = temp_dict.setdefault(letter, {})
            temp_dict[self._end] = self._end
        return trie

    def find_word(self, word):
        sub_trie = self.trie

        for letter in word:
            if letter in sub_trie:
                sub_trie = sub_trie[letter]
            else:
                return False
        else:
            if self._end in sub_trie:
                return True
            else:
                return False

    def add_word(self, word):
        if self.find_word(word):
            return self.trie

        temp_trie = self.trie
        for letter in word:
            if letter in temp_trie:
                temp_trie = temp_trie[letter]
            else:
                temp_trie = temp_trie.setdefault(letter, {})
        temp_trie[self._end] = self._end
        return temp_trie

my_trie = Trie()
file1 = open("dictionary.txt","r") 
lines = file1.readlines()
for line in lines:
  my_trie.add_word(line.strip())
my_trie.add_word("i")
print(my_trie.find_word("word"))

class Solution():
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        phone = {'2': ['a'],
                 '212':['b'],
                 '21212':['c'],
                 '3': ['d'],
                 '313':['e'],
                 '31313':['f'],
                 '4': ['g'],
                 '414':['h'],
                 '41414':['i'],
                 '5': ['j'],
                 '515':['k'],
                 '51515':['l'],
                 '6': ['m'],
                 '616':['n'],
                 '61616':['o'],
                 '7': ['p'],
                 '717':['q'],
                 '71717':['r'],
                 '7171717':['s'],
                 '8': ['t'],
                 '818':['u'],
                 '81818':['v'],
                 '9': ['w'],
                 '919':['x'],
                 '91919':['y'],
                 '9191919':['z'],
                 '0': [' ']}            
                
        ans = []       
        

        def backtrack(combination, next_digits):
            output=""
            while(len(next_digits)>0):
              if len(next_digits)==1 :
                for letter in phone[next_digits[0]]:
                  output+=letter
                break
              elif (next_digits[0]!='1' and next_digits[1]!='1'):
                for letter in phone[next_digits[0]]:
                  output+=letter
                next_digits=next_digits[1:]
              elif(len(next_digits)>6 and next_digits[5]=='1' and (next_digits[0]=='7' or next_digits[0]=='9') ):
                for letter in phone[next_digits[0]+next_digits[5]+next_digits[0]+next_digits[5]+next_digits[0]+next_digits[5]+next_digits[0]]:
                  output+=letter
                next_digits=next_digits[7:]
              elif(len(next_digits)>4 and next_digits[3]=='1'):
                for letter in phone[next_digits[0]+next_digits[3]+next_digits[0]+next_digits[3]+next_digits[0]]:
                 output+=letter
                next_digits=next_digits[5:]
              elif(len(next_digits)>2 and next_digits[1]=='1'):
                for letter in phone[next_digits[0]+next_digits[1]+next_digits[0]]:
                  output+=letter
                next_digits=next_digits[3:]
              elif(next_digits[0]=='1'):
                print("Incorrect input")
                break
              else:
                print("Incorrect input")
                break
              
            return output


        if digits:
           ans= backtrack("", digits)
        return ans

converter = Solution()
test = s
out = converter.letterCombinations(test)
print(out)
for word in out.split():
  if(my_trie.find_word(word)):
      print(word)

