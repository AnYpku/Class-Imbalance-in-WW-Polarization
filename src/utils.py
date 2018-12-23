import pickle

def pkl_save_obj(obj, name):
    with open('results/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def pkl_load_obj(name):
    with open('results/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
