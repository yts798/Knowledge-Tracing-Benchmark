Currently, the model support dkt and 5 datasets. 
The test of DKVMN can be used as a tutorial about how to use the benchmark

It has 4 functions, which is 
"[1] The first is to test 1 model with 1 dataset",
"[2] The second is to test 1 model with all datasets",
"[3] The third is to test all models with 1 datasets",
"[4] The fourth is to test all models with all dataset",

Now strategy 1 and 2 is achieved
Strategy 3 and 4 can be achieved easily with strategy 1 and 2

If strategy 2 is selected, the dkt model will be tested on all datasets

To run the benchmark, use:
python main.py 
and follow the instructions

The package of tensorflow and mxnet is required

To test the DKT without error
The following should be runned
pip install numpy==1.18.5
pip install tensorflow==2.0.0
pip install tensorflow-gpu==2.0.0
