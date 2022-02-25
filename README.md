Mondrian k-anonymization

In this repository, you can find two different scripts:

- The first is Mondrian_Multidimensional_K-Anonymization.ipynb, a Python notebook which includes all definitions of helper functions and the algorithm implementation, together with line-by-line comments about the whole procedure and a set of tests on both real, as requested by the assignment, and synthetic data, as found in the original paper, in order to compare our results with those obtained by the authors. Real data used are from the adult.all.txt file, an instance of the Adult Census Income dataset (http://archive.ics.uci.edu/ml/datasets/Adult). Synthetic data were generated following the paper configurations.
 
- The second is mondrian_k_anonymization.py, which is a script that can be run from terminal. It only includes the algorithm implementation and helper functions. It takes a non-anonymised dataset as input, desired k-level, partitioning mode (strict or relaxed) and aggregation statistics for numerical variables (range or mean) as parameters, and returns the anonymised dataset as a .csv file. No data pre-processing is done in this script. The input dataset should contain only quasi-identifier attributes and they will all be used to anonymise the items. IDs and SDs must be removed before running it to avoid having to manually specify which attributes are IDs, QIs or SD. The script will automatically decide whether attributes are numerical or categorical. In order to run it, open terminal and type:

python mondrian_k_anonymization.py inputFilename k r|s r|m, where:

  - inputFilename: name of the input dataset file,

  - k: desired level of k-anonymisation to achieve (positive integer),
  
  - r|s: relaxed or strict partitioning,
  
  - r|m: range or mean as aggregation statistics for numerical variables.

A sample of input (adult.txt) and output (anon_df.csv) datasets is included in this repository, obtained with k = 5, relaxed partitioning and mean as aggregation function.
