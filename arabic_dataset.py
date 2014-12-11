## Don't FORGET TO CONSIDER LEARNING ONE NUMBER AT A TIME AGAINST ALL OTHERS THEN SLOWLY ADDING THE COMPLEXITY

# data structure: list of all (input, output) pairs. 
# There's one for training_data and testing_data.

# Encode outputs in binary.
output = []
output.append([0,0,0,0])
output.append([0,0,0,1])
output.append([0,0,1,0])
output.append([0,0,1,1])
output.append([0,1,0,0])
output.append([0,1,0,1])
output.append([0,1,1,0])
output.append([0,1,1,1])
output.append([1,0,0,0])
output.append([1,0,0,1])


# Read in the training data
i = 1
training_data = []
output_index = 0
with open('Train_Arabic_Digit.txt') as infile:
    next(infile)
    timeseries = []
    for line in infile:

        cells = line.split(' ')

        if len(cells[0]) > 0:
            timepoint = [float(x.strip()) for x in cells]
            timeseries.append(timepoint)

        else:
            training_data.append((timeseries,output[output_index]))
            timeseries = []
            if i % 660 == 0:
                output_index += 1
                print str(output_index)
            i += 1

    # add the last sample
    training_data.append((timeseries,output[output_index]))


# Read in the testing data
i = 1
testing_data = []
output_index = 0
with open('Test_Arabic_Digit.txt') as infile:
    next(infile)
    timeseries = []
    for line in infile:

        cells = line.split(' ')

        if len(cells[0]) > 0:
            timepoint = [float(x.strip()) for x in cells]
            timeseries.append(timepoint)

        else:
            testing_data.append((timeseries,output[output_index]))
            timeseries = []
            if i % 220 == 0:
                output_index += 1
                print str(output_index)
            i += 1

    # add the last sample
    testing_data.append((timeseries,output[output_index]))

