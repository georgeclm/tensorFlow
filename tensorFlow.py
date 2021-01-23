import tensorflow as tf  # now import the tensorflow module
print(tf.version)  # make sure the version is 2.x
string = tf.Variable("this is a string", tf.string)
number = tf.Variable(324, tf.int16)
floating = tf.Variable(3.567, tf.float64)
print(string)
print(number)
print(floating)
rank1_tensor = tf.Variable(["Test"], tf.string)
# tensor have to make the array is rectangular shape only if not then error
rank2_tensor = tf.Variable(
    [["test", "ok", "another"], ["test", "yes", "no"]], tf.string)
# rank is like to coount 2d array or more
print(tf.rank(rank2_tensor))
print(rank2_tensor.shape)
# tf.ones mean to print number 1
# this mean we have 1 element inside 2 list and each of the list consist of 3 element
# first parameter mean how many list that want to create then second parameter is how many list inside 1 big list and the last parameter is how many element inside a list
# tf.ones() creates a shape [1,2,3] tensor full of ones
tensor1 = tf.ones([1, 2, 3])
# reshape existing data to shape [2,3,1]
tensor2 = tf.reshape(tensor1, [2, 3, 1])
# for tensor 3 -1 mean try to find first there is 6 element in the tensor1 then -1 mean how many multiplier to get 6 which is 3 *2
# -1 tells the tensor to calculate the size of the dimension in that place
tensor3 = tf.reshape(tensor2, [3, -1])
# this will reshape the tensor to [3,2]
# The numer of elements in the reshaped tensor MUST match the number in the original
print(tensor1)
print(tensor2)
print(tensor3)
# Notice the changes in shape
# Creating a 2D tensor
matrix = [[1, 2, 3, 4, 5],
          [6, 7, 8, 9, 10],
          [11, 12, 13, 14, 15],
          [16, 17, 18, 19, 20]]

tensor = tf.Variable(matrix, dtype=tf.int32)
print(tf.rank(tensor))
print(tensor.shape)
# Now lets select some different rows and columns from our tensor

three = tensor[0, 2]  # selects the 3rd element from the 1st row
print(three)  # -> 3

row1 = tensor[0]  # selects the first row
print(row1)

column1 = tensor[:, 0]  # selects the first column
print(column1)

row_2_and_4 = tensor[1::2]  # selects second and fourth row
print(row_2_and_4)

column_1_in_row_2_and_3 = tensor[1:3, 0]
print(column_1_in_row_2_and_3)
