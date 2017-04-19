

def get_support_features(support_list, feature_list):
    if len(feature_list) < len(support_list) + 1:
        raise Exception('feature_list length and get_support list length do not match')

    support_feature_list = [feature_list[support_list[i]+1] for i in range(0,len(support_list))]
    support_feature_list.insert(0, 'poi')
    return support_feature_list

