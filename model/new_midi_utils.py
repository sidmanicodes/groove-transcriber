from tqdm import tqdm
import pretty_midi
import collections
import pandas as pd
import torch

DRUM_NOTES = {
    0: [35, 36],           # Kick
    1: [38, 40],           # Snare head / rim
    2: [37],               # Cross-stick
    3: [48, 50],           # Rack toms
    4: [45, 47, 43],       # Floor toms
    5: [46],               # Open hi-hats
    6: [42],               # Closed hi-hats
    7: [44],               # Hi-hat pedal
    8: [49, 55, 57, 52],   # Crash cymbals
    9: [51, 59],           # Ride cymbal
    10: [53]               # Ride bell
}

PITCH_TO_INDEX = {
    36: 0,              # Kick
    38: 1,              # Snare head / rim
    37: 2,              # Cross-stick
    48: 3,              # Rack toms
    45: 4,              # Floor toms
    46: 5,              # Open hi-hats
    42: 6,              # Closed hi-hats
    44: 7,              # Hi-hat pedal
    49: 8,              # Crash cymbals
    51: 9,              # Ride cymbal
    53: 10              # Ride bell
}

INDEX_TO_PITCH = {
    0: 36,              # Kick
    1: 38,              # Snare head / rim
    2: 37,              # Cross-stick
    3: 48,              # Rack toms
    4: 45,              # Floor toms
    5: 46,              # Open hi-hats
    6: 42,              # Closed hi-hats
    7: 44,              # Hi-hat pedal
    8: 49,              # Crash cymbals
    9: 51,              # Ride cymbal
    10: 53              # Ride bell
}

def convert_bpm_to_microseconds(tempo: int) -> int:
    if tempo != 0:
        return int(60_000_000 / tempo)
    return 0

def midi_to_tensor(midi_path: str, max_samples: int, sr: int) -> torch.Tensor:
    """
    Convert a MIDI file to a tensor

    Arguments
    ---------
    - midi_path | str:
        - Path to MIDI file
    - max_samples | int:
        - Max number of samples to extract
    - sr | int:
        - Sample size
    
    Returns
    -------
    - midi_tensor | torch.Tensor [shape=(10, # of ticks)]
        - A tensor containing velocities for 10 different drums. 
          Each row corresponds to a drum and each column corresponds
          to a tick, which is how time is measured in MIDI format
    """
    # Load in the MIDI file
    pm = pretty_midi.PrettyMIDI(midi_path)
    instrument = pm.instruments[0]
    notes = collections.defaultdict(list)

    # Sort the notes by start time
    sorted_notes: list[pretty_midi.Note] = sorted(instrument.notes, key=lambda note: note.start)

    # Since we pad or truncate the actual audio samples in accordance to
    # a max_samples parameter, we need to ensure that the corresponding
    # MIDI tensor is padded / truncated in the same manner
    max_tick = pm.time_to_tick(max_samples / sr)

    for note in sorted_notes:
        start_tick = pm.time_to_tick(note.start)
        end_tick = min(pm.time_to_tick(note.end), max_tick) # Prevent the duration from exceeding max_tick length

        # Only parse MIDI information while
        # the tick is less than the max tick
        if start_tick <= max_tick:
            notes['drum'].append(note.pitch)
            notes['start_tick'].append(start_tick)
            notes['end_tick'].append(end_tick)
            notes['velocity'].append(note.velocity)
        else:
            break
    
    # Create the MIDI tensor
    midi_tensor = torch.zeros((len(DRUM_NOTES), max_tick + 1, 2))
    for drum_index in DRUM_NOTES:
        for i in range(len(notes['drum'])):
            extracted_drum = notes['drum'][i]
            start_tick = notes['start_tick'][i]
            end_tick = notes['end_tick'][i]
            if extracted_drum in DRUM_NOTES[drum_index] and start_tick <= max_tick:
                velocity = notes['velocity'][i]
                midi_tensor[drum_index][start_tick][0] = velocity
                midi_tensor[drum_index][start_tick][1] = end_tick

    return midi_tensor

def tensor_to_midi(
    midi_tensor: torch.Tensor,
    tempo: int,
    out_file: str = "output.mid",
) -> pretty_midi.PrettyMIDI:
    # 
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo, resolution=480)
    instrument = pretty_midi.Instrument(
        program=0, is_drum=True
    )

    for drum_index in range(midi_tensor.shape[0]):
        for tick in tqdm(range(midi_tensor.shape[1]), desc="Ticks..."):
            velocity = int(midi_tensor[drum_index][tick][0])
            if velocity >= 0:
                start_time = pm.tick_to_time(tick)
                end_time = pm.tick_to_time(int(midi_tensor[drum_index][tick][1]))

                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=DRUM_NOTES[drum_index][0],  # Assuming first pitch in DRUM_NOTES
                    start=start_time,
                    end=end_time
                )
                instrument.notes.append(note)

    pm.instruments.append(instrument)
    pm.write(out_file)
    print("Successfully saved output as a midi file!")
    return pm

# midi_path = "../data/drummer1/session1/5_jazz-funk_116_beat_4-4.mid"

# tempo = convert_bpm_to_microseconds(tempo=116)
# all_velocities = midi_to_tensor(midi_path=midi_path, max_samples=1_000_000, sr=44_100)
# tensor_to_midi(all_velocities, tempo=116)