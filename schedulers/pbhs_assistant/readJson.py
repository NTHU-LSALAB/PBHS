import json
import copy

# def readConfig (numTrial, path):
#     path_modified = copy.deepcopy(path)
#     for i in range(numTrial):
#         path_modified[i] = path_modified[i][:-11]
#         path_modified[i] += "params.json"

#     config_return = []
#     for j in range (numTrial): 
#         config = []
#         with open(path_modified[j], 'r') as f:
#             lines = f.readlines()
#         merge = ""
#         for line in lines:
#             merge += line.strip()
#         parse = json.loads(merge)
#         for key in parse:
#             if (key != "args"):
#                 config.append( parse[key] )

#         config_return.append(config)
        
#     return config_return


# def readAcc (numTrial, numIteration, path):
#     acc_return = []
#     for j in range (numTrial): 
#         accuracy = []
#         with open(path[j], 'r') as f:
#             lines = f.readlines()
#             for i in range(numIteration):
#                parse = json.loads(lines[i])
#                accuracy.append( parse["mean_accuracy"] )
                
#             acc_return.append(accuracy)
    
#     return acc_return

def readSingleAcc (numIteration, path, metric_key):
    accuracy = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for i in range(numIteration):
            parse = json.loads(lines[i])
            accuracy.append( parse[metric_key] )
                
    return accuracy