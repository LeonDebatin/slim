from sklearn.model_selection import train_test_split



def basic_information(df):
    print('Total NA:',df.isna().sum().sum())
    print('Imbalance:',df['target'].value_counts())
    print('Dtyoes:',df.dtypes)
    print('duplicates', df.duplicated().sum())

def get_train_test_indices(df):
    train_indices = []
    test_indices = []
    for i in range(1,31):
        train, test = train_test_split(df.index, test_size=0.3, random_state=i, stratify=df['target'])
        train_indices.append(train.to_list())
        test_indices.append(test.to_list())
    
    return train_indices, test_indices


def info_dict(name, df, categoricals):
    info = {}
    info['name'] = name
    info['n_samples'] = df.shape[0]
    info['n_features'] = df.shape[1] - 1
    
    imbalance = df['target'].value_counts(ascending=True)[1] / df.shape[0]
    if imbalance > 0.5:
        imbalance = 1 - imbalance
        
    info['imbalance'] = imbalance
    info['categoricals'] = categoricals
    info['n_categoricals'] = len(categoricals)
    info['train_indices'], info['test_indices'] = get_train_test_indices(df)
    
    return info
    