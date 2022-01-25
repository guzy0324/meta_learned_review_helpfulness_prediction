from re import sub

# https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    string = sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = sub(r"\'s", " \'s", string)
    string = sub(r"\'ve", " \'ve", string)
    string = sub(r"n\'t", " n\'t", string)
    string = sub(r"\'re", " \'re", string)
    string = sub(r"\'d", " \'d", string)
    string = sub(r"\'ll", " \'ll", string)
    string = sub(r",", " , ", string)
    string = sub(r"!", " ! ", string)
    string = sub(r"\(", " \( ", string)
    string = sub(r"\)", " \) ", string)
    string = sub(r"\?", " \? ", string)
    string = sub(r"\s{2,}", " ", string)
    return string.strip().lower()
