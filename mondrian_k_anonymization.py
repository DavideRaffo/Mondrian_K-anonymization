# import libraries
import numpy as np
import math
import pandas as pd
import time
import sys

# expect: filename.py | dataset | k_level | strict/relaxed_part | range/mean_agg
arguments = sys.argv
if len(arguments) != 5:
    print('Invalid number of arguments. Expected 5 arguments.')
    sys.exit(1)

print('Reading dataset...')
# load the dataset and apply column names
df = pd.read_csv('{}'.format(arguments[1]), sep=",", header=None, index_col=False, engine='python')

# remove NaNs
#df = df[df!='-1'] # -1 is nan
df.dropna(inplace=True)
df.reset_index(inplace=True)
df = df.iloc[:,1:]

# infer data types
types = list(df.dtypes)
cat_indices = [i for i in range(len(types)) if types[i]=="object"]

# convert df to numpy array
df = np.array(df)
#print(len(df))

# function to compute the span of a given column while restricted to a subset of rows (a data partition)
def colSpans(df, cat_indices, partition):
	spans = dict()
	for column in range(len(types)):
		#print('Current column:',column)
		dfp = df[partition,column] # restrict df to the current column
		#print(dfp)
		if column in cat_indices:
			span = len(np.unique(dfp)) # span of categorical variables is its number of unique classes
		else:
			span = np.max(dfp)-np.min(dfp) # span of numerical variables is its range
		spans[column] = span
	return spans

# function to split rows of a partition based on median value (categorical vs. numerical attributes)
def splitVal(df, dim, part, cat_indices, mode):
    dfp = df[part,dim] # restrict whole dataset to a single attribute and rows in this partition
    unique = list(np.unique(dfp))
    length = len(unique)
    if dim in cat_indices: # for categorical variables
        if mode=='strict': # i do not mind about |lhs| and |rhs| being equal
            lhv = unique[:length//2]
            rhv = unique[length//2:]
            lhs_v = list(list(np.where(np.isin(dfp,lhv)))[0]) # left partition
            rhs_v = list(list(np.where(np.isin(dfp,rhv)))[0]) # right partition
            lhs = [part[i] for i in lhs_v]
            rhs = [part[i] for i in rhs_v]
        elif mode=='relaxed': # i want |lhs| = |rhs| +-1
            lhv = unique[:length//2]
            rhv = unique[length//2:]
            lhs_v = list(list(np.where(np.isin(dfp,lhv)))[0]) # left partition
            rhs_v = list(list(np.where(np.isin(dfp,rhv)))[0]) # right partition
            lhs = [part[i] for i in lhs_v]
            rhs = [part[i] for i in rhs_v]
            diff = len(lhs)-len(rhs)
            if diff==0:
                pass
            elif diff<0:
                lhs1 = rhs[:(np.abs(diff)//2)] # move first |diff|/2 indices from rhs to lhs
                rhs = rhs[(np.abs(diff)//2):] 
                lhs = np.concatenate((lhs,lhs1))
            else:
                rhs1 = lhs[-(diff//2):]
                lhs = lhs[:-(diff//2)]
                rhs = np.concatenate((rhs,rhs1))
        else:
            lhs, rhs = splitVal(df, dim, part, cat_indices, 'relaxed')
    else: # for numerical variables, split based on median value (strict or relaxed)
        median = np.median(dfp)
        if mode=='strict': # strict partitioning (do not equally split indices of median values)
            lhs_v = list(list(np.where(dfp < median))[0])
            rhs_v = list(list(np.where(dfp >= median))[0])
            lhs = [part[i] for i in lhs_v]
            rhs = [part[i] for i in rhs_v]
        elif mode=='relaxed': # exact median values are equally split between the two halves
            lhs_v = list(list(np.where(dfp < median))[0])
            rhs_v = list(list(np.where(dfp > median))[0])
            median_v = list(list(np.where(dfp == median))[0])
            lhs_p = [part[i] for i in lhs_v]
            rhs_p = [part[i] for i in rhs_v]
            median_p = [part[i] for i in median_v]
            diff = len(lhs_p)-len(rhs_p) # i need to have |lhs| = |rhs| +- 1
            if diff<0:
                med_lhs = np.random.choice(median_p, size=np.abs(diff), replace=False) # first even up |lhs_p| and |rhs_p|
                med_to_split = [i for i in median_p if i not in med_lhs] # prepare remaining indices for equal split
                lhs_p = np.concatenate((lhs_p,med_lhs))
            else: # same but |rhs_p| needs to be levelled up to |lhs_p|
                med_rhs = np.random.choice(median_p, size=np.abs(diff), replace=False)
                med_to_split = [i for i in median_p if i not in med_rhs]
                rhs_p = np.concatenate((rhs_p,med_rhs))
            med_lhs_1 = np.random.choice(med_to_split, size=(len(med_to_split)//2), replace=False) # split remaining median indices equally between lhs and rhs
            med_rhs_1 = [i for i in med_to_split if i not in med_lhs_1]
            lhs = np.concatenate((lhs_p,med_lhs_1))
            rhs = np.concatenate((rhs_p,med_rhs_1))
        else:
            lhs, rhs = splitVal(df, dim, part, cat_indices, 'relaxed')
    return [int(x) for x in lhs], [int(x) for x in rhs]

# create k-anonymous equivalence classes
def partitioning(df, k, cat_indices, mode):

	final_partitions = []
	working_partitions = [[x for x in range(len(df))]] # start with full dataset
	#print('Working partitions:',working_partitions)
	
	while len(working_partitions) > 0: # while there is at least one working partition left
	
		partition = working_partitions[0] # take the first in the list
		working_partitions = working_partitions[1:] # remove it from list of working partitions
		#print('Current partition:',partition)
		
		if len(partition) <= 2*k: # if it is not at least 2k long, i.e. if i cannot get any new acceptable partition pair, at least k-long each
			#print('It is final')
			final_partitions.append(partition) # append it to final set of partitions
			# and skip to the next partition
		else:
			spans = colSpans(df, cat_indices, partition) # else, get spans of the feature columns restricted to this partition
			ordered_span_cols = sorted(spans.items(), key=lambda x:x[1], reverse=True) # sort col indices in descending order based on their span
			for dim, _ in ordered_span_cols: # select the largest first, then second largest, ...
				lhs, rhs = splitVal(df, dim, partition, cat_indices, mode) # try to split this partition
				if len(lhs) >= k and len(rhs) >= k: # if new partitions are not too small (<k items), this partitioning is okay
					working_partitions.append(lhs) 
					working_partitions.append(rhs) # re-append both new partitions to set of working partitions for further partitioning
					#print('New partitions:',lhs,rhs)
					break # break for loop and go to next partition, if available          
			else: # if no column could provide an allowable partitioning
				final_partitions.append(partition) # add the whole partition to the list of final partitions
		
	return final_partitions

# build k-anonymous equivalence classes
k = int(arguments[2])
modeArg = str(arguments[3])
mode = 'relaxed'
if modeArg == 's': mode = 'strict'
 

equivalence_classes = partitioning(df, k, cat_indices, mode)
print('Partitioning completed. {} partitions were created'.format(len(equivalence_classes)))

# generate the anonymised dataset
def anonymize_df(df, partitions, cat_indices, mode='range'):
  
    anon_df = []
    categorical = cat_indices

    for ip,p in enumerate(partitions):
        aggregate_values_for_partition = []
        partition = df[p]
        for column in range(len(types)):
            if column in categorical:
                values = list(np.unique(partition[:,column]))
                aggregate_values_for_partition.append(','.join(values))
            else:
                #print(column)
                if mode=='mean':
                    aggregate_values_for_partition.append(np.mean(partition[:,column]))
                else:
                    col_min = np.min(partition[:,column])
                    col_max = np.max(partition[:,column])
                    if col_min == col_max:
                        aggregate_values_for_partition.append(col_min)
                    else:
                        aggregate_values_for_partition.append('{}-{}'.format(col_min,col_max))
                    #print(col_min,col_max)
        for i in range(len(p)):
            anon_df.append([int(p[i])]+aggregate_values_for_partition)
  
    df_anon = pd.DataFrame(anon_df)
    df_anon = df_anon.infer_objects()
    dfn1 = df_anon.sort_values(df_anon.columns[0])
    dfn1 = dfn1.iloc[:,1:]
    return np.array(dfn1)

# anonymise dataset
aggregation = 'range'
aggregationArg = str(arguments[4])
if aggregationArg == 'm': aggregation = 'mean'
 
dfn = anonymize_df(df, equivalence_classes, cat_indices, aggregation)
#print(len(dfn))
np.savetxt('anon_df.csv', dfn, fmt='%s', delimiter=';')
print('Anonymization completed.')
sys.exit(1)