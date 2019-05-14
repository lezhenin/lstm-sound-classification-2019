import os
import csv
import gc
import pickle
import soundfile as sf
import librosa
import numpy as np

def load_metadata(filename, base_path='./UrbanSound8K'):

    metadata = dict({
        'wav_file': [],
        'duration': [],
        'class_id': [],
        'salience': [],
        'fold': []
    })

    class_dict = dict()

    file = open(base_path + '/' + filename, 'r')
    reader = csv.reader(file, delimiter=',')
    header = next(reader)

    for row in reader:
        slice_file_name, fs_id, start, end, salience, fold, class_id, class_name = row
        metadata['wav_file'].append('audio/fold%s/%s' % (fold, slice_file_name))
        metadata['duration'].append(float(end) - float(start))
        metadata['salience'].append(int(salience))
        metadata['class_id'].append(int(class_id))
        metadata['fold'].append(int(fold))
        class_dict[int(class_id)] = class_name

    metadata['class_dict'] = [class_dict[i] for i in range(len(class_dict))]

    return metadata


def split_metadata(metadata, start, end):
    metadata_part = dict()
    metadata_part['wav_file'] = metadata['wav_file'][start:end]
    metadata_part['duration'] = metadata['duration'][start:end]
    metadata_part['salience'] = metadata['salience'][start:end]
    metadata_part['class_id'] = metadata['class_id'][start:end]
    metadata_part['class_dict'] = metadata['class_dict']
    return metadata_part


def prepare_resampled_audio(metadata, new_sample_rate, mono=True, base_path='./UrbanSound8K', verbose=False):

    if verbose:
        print('Prepare %d files. Resample to %d Hz.' % (len(metadata['wav_file']), new_sample_rate))

    for i in range(len(metadata['wav_file'])):

        wav_file = base_path + '/' + metadata['wav_file'][i]
        resampled_wav_file = wav_file[:-4] + '_%d' % new_sample_rate + '.wav'

        if verbose:
            print('%d/%d ' % (i, len(metadata['wav_file'])), end='')

        if not os.path.isfile(resampled_wav_file):
            data, sample_rate = sf.read(wav_file)
            if mono:
                data = librosa.core.to_mono(data.transpose())
            data = librosa.core.resample(data, sample_rate, new_sample_rate)
            sf.write(resampled_wav_file, data, new_sample_rate)

            if verbose:
                print('File %s created.' % resampled_wav_file)
        else:

            if verbose:
                print('File %s exists. Skip.' % resampled_wav_file)

    print('Done.')


def load_wav(wav_file, sample_rate=None):

    if sample_rate is not None:
        prepared_wav_file = wav_file[:-4] + '_%d' % sample_rate + '.wav'
        if os.path.exists(prepared_wav_file):
            data, loaded_sample_rate = sf.read(prepared_wav_file)
            assert loaded_sample_rate == sample_rate
            return data, loaded_sample_rate

    data, loaded_sample_rate = sf.read(wav_file)
    data = librosa.core.to_mono(data.transpose())

    if sample_rate is not None:
        data = librosa.core.resample(data, loaded_sample_rate, sample_rate)
        loaded_sample_rate = sample_rate

    return data, loaded_sample_rate


def load_cache_dump(cache_tag, base_path='./UrbanSound8K'):
    
    dump_filename = base_path + '/cache/' + cache_tag + '.pck'
    if not os.path.exists(dump_filename):
        return [], False
    with open(dump_filename, 'rb') as dump_file:
        return pickle.load(dump_file), True
    
    
def save_cache_dump(data, cache_tag, base_path='./UrbanSound8K'):
    
    dump_filename = base_path + '/cache/' + cache_tag + '.pck'
    with open(dump_filename, 'wb') as dump_file:
        pickle.dump(data, dump_file)
            
            
def load_data(
    metadata,
    callback=None,
    args=None,
    sample_rate=None,
    cache_tag=None,
    base_path='./UrbanSound8K',
    verbose=False
):
    
    data_len = len(metadata['wav_file'])
    
    if cache_tag is not None:
        all_data, exists = load_cache_dump(cache_tag, base_path)
        if exists:
            assert len(all_data) == data_len       
            return all_data

    all_data = []

    for i in range(len(metadata['wav_file'])):
        wav_file = base_path + '/' + metadata['wav_file'][i]
        data, _ = load_wav(wav_file, sample_rate)
        if callback is not None:
            data = callback(data, *args)
        all_data.append(data)

        if verbose:
            print('%d/%d File %s processed' % (i, len(metadata['wav_file']), wav_file))

    if cache_tag is not None:
        save_cache_dump(all_data, cache_tag, base_path)

    if verbose:
        print('Done.')

    return all_data


def load_large_data(
    metadata,
    n_parts=8,
    callback=None,
    args=None,
    sample_rate=None,
    cache_tag=None,
    base_path='./UrbanSound8K',
    verbose=False
):
    
    data_len = len(metadata['wav_file'])
    
    if cache_tag is not None:
        all_data, exists = load_cache_dump(cache_tag, base_path)
        if exists:
            assert len(all_data) == data_len       
            return all_data
        
    checkpoints = [(data_len // n_parts) * i for i in range(n_parts + 1)]
    checkpoints[-1] = data_len
    
    all_data = []
    
    for i in range(len(checkpoints) - 1):
    
        label = '_p%d' % (i + 1)
        start, end = checkpoints[i], checkpoints[i + 1]
    
        if verbose:
            print('Process %d part from %d to %d.' % (i, start, end))
      
        metadata_part = split_metadata(metadata, start, end)
 
        data = load_data(metadata_part, callback, args, sample_rate, cache_tag + label, base_path, verbose)
        all_data += data
        
        gc.collect()
      
    assert(len(all_data) == data_len)
    
    if cache_tag is not None:
        save_cache_dump(all_data, cache_tag, base_path)

    if verbose:
        print('Done.')
        
    return all_data
    
    
    
    
    
    
    
    
    
    
    
    
    
    
