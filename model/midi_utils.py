"""
Utility file for MIDI operations to be made during the pre-processing / training / post-processing
of the JojoNet model
"""

from typing import Union
import mido.messages
import mido.midifiles
from tqdm import tqdm
import mido
import numpy as np
import torch

DRUM_NOTES = [
    [35, 36],           # Kick
    [38, 40],           # Snare head / rim
    [37],               # Cross-stick
    [48, 50],           # Rack toms
    [45, 47, 43],       # Floor toms
    [46],               # Open hi-hats
    [42],               # Closed hi-hats
    [44],               # Hi-hat pedal
    [49, 55, 57, 52],   # Crash cymbals
    [51, 59],           # Ride cymbal
    [53]                # Ride bell
]

DRUM_MAPPINGS = {
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

def extract_velocities_from_midi(midi_path: str) -> Union[torch.Tensor, int]:
    """
    Extracts the velocities of a given drum note from a given MIDI file

    Arguments
    ---------
    - midi_path | str:
        - Path to MIDI file

    Returns
    -------
    - velocities | torch.Tensor
        - Array of velocities of the given drum note
    """

    # Load the MIDI file
    midi = mido.MidiFile(midi_path)

    # Initialize max tick
    max_tick = 0

    tpb = midi.ticks_per_beat

    # Create velocities and onset dicts
    velocities_dict = {i:{} for i in range(len(DRUM_NOTES))}

    for i, notes in enumerate(DRUM_NOTES):
        # Initialize the time accumulator
        accumulator = 0
        for track in midi.tracks:
            for msg in track:
                if not msg.is_meta:
                    # Accumulate the time for the current track
                    accumulator += msg.time

                    # Get the current global time by considering all tracks
                    current_tick = accumulator

                    # Update the max tick
                    max_tick = max(max_tick, current_tick)

                    # Update the velocity vector with the new velocity
                    if msg.type == 'note_on' and msg.note in notes:
                        velocities_dict[i][current_tick] = msg.velocity

                # Change tempo if a tempo change is detected
                elif msg.type == 'set_tempo':
                    tempo = msg.tempo
    
    # Convert the velocity dictionary into a tensor corresponding 
    # to all the ticks across every drum note
    velocities = torch.zeros((len(DRUM_NOTES), max_tick + 1))
    for i in range(len(velocities_dict)):
        for tick in velocities_dict[i]:
            velocities[i][tick] = velocities_dict[i][tick]

    return velocities, midi.ticks_per_beat

def write_to_midi(velocities: torch.Tensor, tempo: int, tpb) -> None:
    midi = mido.MidiFile(ticks_per_beat=tpb)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    channel = 9

    track.append(mido.MetaMessage('set_tempo', tempo=tempo))
    track.append(mido.Message('program_change', program=0, channel=channel))

    # Create a list to keep track of the last tick for each instrument
    last_tick = [0] * velocities.shape[0]

    for tick in range(velocities.shape[1]):
        for instr in range(velocities.shape[0]):
            if velocities[instr][tick] > 0:
                note_status = 'note_on'
            else:
                note_status = 'note_off'

            # Calculate the delta time for this instrument
            delta = tick - last_tick[instr]
            last_tick[instr] = tick

            # Only create a message if there's an event to record (note_on or note_off)
            if delta > 0 or note_status == 'note_on':
                msg = mido.Message(
                    note_status, 
                    note=DRUM_MAPPINGS[instr], 
                    velocity=int(velocities[instr][tick]),
                    time=delta,
                    channel=channel
                )
                track.append(msg)

    midi.save("output.mid")
    print("Successfully saved output as a midi file!")

midi_path = "../data/drummer1/session1/32_latin-samba_116_fill_4-4.mid"

tempo = convert_bpm_to_microseconds(tempo=116)
all_velocities, tpb = extract_velocities_from_midi(midi_path=midi_path)
write_to_midi(all_velocities, tempo=tempo, tpb=tpb)