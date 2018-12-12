# 
# ArcMLP - Execute preprocessing compoment
#
# Perfom specified preprocessing methods and 
# create 4 csv files (x_train, x_test, y_train and y_test).
# 
# This component is still in progress and will be continue for
# future work
#

import arch_dp.data_preprocessing as dp
import sys
import csv

METHODS = ["read_file", "remove_variables", "filter_data", "add_features", \
           "split_features_labels", "one_hot_encoding", "impute_values", "split_train_test"]

def check_present(csv_file):
    methods_csv = []

    with open(csv_file,'r') as userFile:
        userFileReader = csv.reader(userFile)
        for row in userFileReader:
            methods_csv.append(row)

    # Removing name of columns
    methods_csv = methods_csv[1:]
    # Getting methods without arguments
    arguments = [item[1] for item in methods_csv]

    df = dp.read_file(arguments[0])
    
    if arguments[1] != None:
        df = dp.remove_variables(df, arguments[1])

    if arguments[2] != None:
        df = dp.filter_data(df, arguments[2])
    
    if arguments[3] != None:
        df = dp.add_features(df, arguments[3])
    
    X, Y = dp.split_features_labels(df, arguments[4])

    if arguments[5] != None:
        X = dp.one_hot_encoding(X)
    
    X_train, X_test, Y_train, Y_test = dp.split_train_test(X, Y, arguments[6])
    
    if arguments[5] != None:
        X_train, X_test = dp.impute_values(X_train, X_test, arguments[7][0], \
                                           arguments[7][1], arguments[7][2], arguments[7][3])

    X_train, X_test, Y_train, Y_test = dp.split_train_test(X, Y, arguments[8])

    X_train.to_csv('x_train.csv', sep='\t')
    X_test.to_csv('x_test.csv', sep='\t')
    Y_train.to_csv('y_train.csv', sep='\t')
    Y_test.to_csv('y_test.csv', sep='\t')

if __name__ == "__main__":
    """
    Run program from the command-line.
    """

    if len(sys.argv) == 2:
        check_present(sys.argv[1])
