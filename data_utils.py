%%writefile data_utils.py
import pandas as pd

def load_datasets(config):
    train_df = pd.read_csv(config.TRAIN_PATH)
    val_df = pd.read_csv(config.VAL_PATH)
    test_df = pd.read_csv(config.TEST_PATH)
    
    laws_df = pd.read_csv(config.LAWS_PATH)
    laws_df['text'] = laws_df['text'].fillna('')
    laws_df['title'] = laws_df['title'].fillna('') if 'title' in laws_df.columns else ''
    
    court_df = pd.read_csv(config.COURT_PATH, usecols=['citation', 'text'])
    court_df['text'] = court_df['text'].fillna('')
    
    return train_df, val_df, test_df, laws_df, court_df

def prepare_texts(laws_df, court_df):
    laws_texts = (laws_df['citation'] + ' ' + laws_df['title'] + ' ' + laws_df['text']).tolist()
    court_texts = (court_df['citation'] + ' ' + court_df['text']).tolist()
    
    law_map = dict(zip(laws_df['citation'], laws_texts))
    court_map = dict(zip(court_df['citation'], court_texts))
    
    return laws_texts, court_texts, law_map, court_map
