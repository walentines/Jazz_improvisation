from music_utils import * 
from preprocess import *
from keras.utils import to_categorical 
import rnn_functions as lstm

chords, abstract_grammars = get_musical_data('data/original_metheny.mid')
corpus, tones, tones_indices, indices_tones = get_corpus_data(abstract_grammars)
N_tones = len(set(corpus))
n_a = 64
x_initializer = np.zeros((1, 1, 78))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))

def load_music_utils():
    chords, abstract_grammars = get_musical_data('data/original_metheny.mid')
    corpus, tones, tones_indices, indices_tones = get_corpus_data(abstract_grammars)
    N_tones = len(set(corpus))
    X, Y, N_tones = data_processing(corpus, tones_indices, 60, 30)   
    return (X, Y, N_tones, indices_tones)

def generate_music(parameters, corpus = corpus, abstract_grammars = abstract_grammars, tones = tones, tones_indices = tones_indices, indices_tones = indices_tones, T_y = 10, max_tries = 1000, diversity = 0.5):
    
    out_stream = stream.Stream()
    
    curr_offset = 0.0                                     
    num_chords = int(len(chords) / 3)                    
    
    print("Predicting new values for different set of chords.")
    for i in range(1, num_chords):
        
        curr_chords = stream.Voice()
        
        for j in chords[i]:
            curr_chords.insert((j.offset % 4), j)
        
        _, indices = predict_and_sample(n_values = 78, n_a = n_a, parameters = parameters)
        indices = list(indices.squeeze())
        pred = [indices_tones[p] for p in indices]
        
        predicted_tones = 'C,0.25 '
        for k in range(len(pred) - 1):
            predicted_tones += pred[k] + ' ' 
        
        predicted_tones +=  pred[-1]
        predicted_tones = predicted_tones.replace(' A',' C').replace(' X',' C')

        predicted_tones = prune_grammar(predicted_tones)
        
        sounds = unparse_grammar(predicted_tones, curr_chords)

        sounds = prune_notes(sounds)

        sounds = clean_up_notes(sounds)

        print('Generated %s sounds using the predicted values for the set of chords ("%s") and after pruning' % (len([k for k in sounds if isinstance(k, note.Note)]), i))
        
        for m in sounds:
            out_stream.insert(curr_offset + m.offset, m)
        for mc in curr_chords:
            out_stream.insert(curr_offset + mc.offset, mc)

        curr_offset += 4.0
        
    out_stream.insert(0.0, tempo.MetronomeMark(number=130))

    mf = midi.translate.streamToMidiFile(out_stream)
    mf.open("output/my_music.midi", 'wb')
    mf.write()
    print("Your generated music is saved in output/my_music.midi")
    mf.close()

    return out_stream

#def predict_and_sample(inference_model, x_initializer = x_initializer, a_initializer = a_initializer, c_initializer = c_initializer):
    #pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
    #indices = np.argmax(pred, 2)
    #results = to_categorical(indices, num_classes = x_initializer.shape[2])
    
    #return results, indices

def predict_and_sample(n_values, n_a, parameters, T_y = 100, temperature = 1.0):    
    x = np.zeros((n_values, 1))
    a0 = np.zeros((n_a, 1))
    c0 = np.zeros((n_a, 1))
    a = a0
    c = c0
    indices = []
    X = []
    for t in range(T_y):
        a, c, y_pred, _ = lstm.lstm_cell_forward(x, a, c, parameters)
        y_pred = np.squeeze(y_pred)
        y_pred = np.log(y_pred) / temperature
        exp_y = np.exp(y_pred)
        y_pred = exp_y / np.sum(exp_y)
        probas = np.random.multinomial(1, y_pred, 1)
        idx = np.random.choice(list(range(n_values)), p = probas.ravel())
        indices.append(idx)
        
        x = np.zeros((n_values, 1))
        x[idx, 0] = 1
        X.append(x)
   
    X = np.array(X)
    X = X.squeeze()
    indices = np.array(indices)
    indices = np.expand_dims(indices, axis = 1)  
    return X, indices
