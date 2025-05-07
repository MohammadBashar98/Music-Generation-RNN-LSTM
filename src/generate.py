import numpy as np
from music21 import instrument, note, stream, chord
from tensorflow.keras.models import load_model

def generate_music(model, network_input, pitchnames, n_vocab):
    prediction_output = []
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    
    pattern = network_input[0]
    for note_index in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)
        pattern = np.append(pattern[1:], [[index / float(n_vocab)]], axis=0)
    
    offset = 0
    output_notes = []
    for pattern in prediction_output:
        if '.' in pattern or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = [note.Note(int(n)) for n in notes_in_chord]
            for n in notes: n.storedInstrument = instrument.Piano()
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        offset += 0.5
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='output/generated.mid')
