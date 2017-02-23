import pandas as pd
import re
import numpy as np

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    #return string.strip().lower()
    return string.strip()


def load_data_and_labels(data_file, delimiter):
    data = pd.read_csv(data_file, sep=delimiter)
    y, entity1_list, entity2_list, x_text, vocab = [], [], [], [], [] 
    for index, row in data.iterrows():
        label = row['GAD_ASSOC']
        entity1 = clean_str(row['NER_GENE_ENTITY'])
        entity2 = clean_str(row['NER_DISEASE_ENTITY'])
        sentence = clean_str(row['GAD_CONCLUSION'])
        # join phrase which has multiple words, e.g., liver cancer ---> liver-cancer
        joined_entity1 = '-'.join(entity1.split())
        joined_entity2 = '-'.join(entity2.split())
        sentence = sentence.replace(entity1, joined_entity1)
        sentence = sentence.replace(entity2, joined_entity2)
        entity1 = joined_entity1 
        entity2 = joined_entity2
        if label == 'Y':
            label = [0, 1]
        else:
            label = [1, 0]
        y.append(label)
        entity1_list.append(entity1) 
        entity2_list.append(entity2)
        x_text.append(sentence)
        #vocab += sentence.split() 
    # Get the vocabulary 
    #vocab = set(vocab)
    y = np.array(y) 
    return (x_text, entity1_list, entity2_list, y)


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        print epoch
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
