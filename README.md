# Distributed-SGD

This is a single python file realizing distribtued stochastic gradient descent on Spark to solve matrix factorization problem in recommendation system.
Use following command to run the dsgd_mf.py file.

#####`$spark-submit dsgd_mf.py <num_factors> <num_workers> <num_iterations> <beta_value> <lambda_value> <inputV_filepath> <outputW_filepath> <outputH_filepath>`

For example:

#####`spark-submit dsgd_mf.py 100 10 50 0.8 0.1 test.csv w.csv h.csv`

num_factors is number of factors, that is, how many factors you want the two decomposed matrix W and H have.

num_workers is how many workers you can provide to parallel the algorithm.

num_iterations is how many epoches you hope the SGD perform on the whole dataset.

beta_value is used to adjust the step size of SGD.

lambda_value is used to adjust the penalty of parameter size of the reconstructed matrixes.

Created by Shuguan Yang; bread5858 at gmail; Apr 16th 2015.

Update:V2 included the calculation of reconstruction error.
