rust port of python in http://neuralnetworksanddeeplearning.com/chap1.html

main goal: me understanding what's going on

done:
* implementation that 'works' without runtime errors

to do:
* verify correctness
* add unit tests
* train using actual training data
* evaluate with test/validation data
* make more efficient

training_data/test_data not included
format: json: 
[{"x":[float;784], "y": u32}]
=> x: 28x28 gray image as float
=> y: label 0.. 9

the data can be found here:
https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/data/mnist.pkl.gz
albeit in a different format