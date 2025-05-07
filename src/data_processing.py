import numpy as np
from music21 import converter, instrument, note, chord

def parse_midi(file_path):
    notes = []
    midi = converter.parse(file_path)
    parts = instrument.partitionByInstrument(midi)
    elements = parts.parts[0].recurse() if parts else midi.flat.notes

    for element in elements:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes
