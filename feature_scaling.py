def featureScaling(arr):
    maxData = max(arr);
    minData = min(arr);
    if maxData == minData:
        return "No need to scale."
    else:
        return [(data - minData)*1.0 / (maxData - minData)*1.0 for data in arr]

data = [115, 140, 175]
print featureScaling(data)  
