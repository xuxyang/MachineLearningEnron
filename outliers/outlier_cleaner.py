#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    tuple_data = [(a[0], n[0], abs(p[0]-n[0])) for a, p, n in zip(ages, predictions, net_worths)]
    tuple_data.sort(key=lambda tup: tup[2])
    cleaned_data = tuple_data[0:81]

    
    return cleaned_data

