import mne
import parameters
from pathlib import Path

def clean_epochs(epochs):
    """
    Remove any channels listed in parameters.DROPPED_CHANNELS that exist in the epochs.
    """
    chans_in_data = epochs.ch_names
    chans_to_drop = [ch for ch in parameters.DROPPED_CHANNELS if ch in chans_in_data]
    if chans_to_drop:
        epochs.drop_channels(chans_to_drop)
    return epochs

def get_valid_subjects():
    """
    Parse parameters.HEARTBEAT and parameters.SILENCE to pair up the subjects
    that have both conditions. Returns a dictionary mapping kid_id -> info.
    """
    subjects = {}
    
    # Process heartbeat files
    for f in parameters.HEARTBEAT:
        stem = Path(f).stem
        parts = stem.split('_')
        if len(parts) >= 5:
            group, cond, kid_id = parts[1], parts[2], parts[4]
            if kid_id not in subjects:
                subjects[kid_id] = {'group': group, 'hb': None, 'si': None}
            if cond == 'hb': 
                subjects[kid_id]['hb'] = f
                
    # Process silence files
    for f in parameters.SILENCE:
        stem = Path(f).stem
        parts = stem.split('_')
        if len(parts) >= 5:
            group, cond, kid_id = parts[1], parts[2], parts[4]
            if kid_id not in subjects:
                subjects[kid_id] = {'group': group, 'hb': None, 'si': None}
            if cond == 'si': 
                subjects[kid_id]['si'] = f

    # Keep only subjects that have both conditions
    valid_subjects = {k: v for k, v in subjects.items() if v['hb'] and v['si']}
    return valid_subjects
