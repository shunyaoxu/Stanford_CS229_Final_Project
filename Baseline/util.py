def label_generator(label):
    # Input: a list of integers
    output = label[0] * 6
    
    if label[1] > label[2] and label[1] > label[3]:
        output += 4
    elif label[1] > label[2] or label[1] > label[3]:
        output += 2
        
    if label[2] > label[3]:
        output += 1
    
    return output
