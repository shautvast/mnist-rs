rust port of python in http://neuralnetworksanddeeplearning.com/chap1.html

main goal: me understanding what's going on

results:  ~95% accuracy in 30 epochs, learning rate = 3.0, batchsize = 10

*training_data/test_data not included*<br/> too big for github

Format: json: <br/>
[{"x":[float;784], "y": u32}]<br/>
=> x: 28x28 gray image as float<br/>
=> y: label 0.. 9

the data can be found here:
https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/data/mnist.pkl.gz
albeit in a different format, convert it with `convert_pickle.py`