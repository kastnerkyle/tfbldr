# music21 is an optional dep
try:
    from .music import pitch_and_duration_to_piano_roll
    from .music import pitches_and_durations_to_pretty_midi
    from .music import quantized_to_pretty_midi
    from .music import quantized_to_pitch_duration
    from .music import plot_pitches_and_durations
    from .music import music21_to_pitch_duration
    from .music import music21_to_piano_roll
    from .music import plot_piano_roll
    from .music import piano_roll_imlike_to_image_array
    from .analysis import midi_to_notes
    from .analysis import notes_to_midi
except ImportError:
    pass
