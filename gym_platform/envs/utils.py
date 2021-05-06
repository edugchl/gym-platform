import collections


def time_elapsed(t1, t2, unit='hours', digit=2):
    elapsed = t2 - t1
    
    if unit == 'days':
        return round(elapsed.total_seconds()/86400, digit)
    elif unit == 'hours':
        return round(elapsed.total_seconds()/3600, digit)
    elif unit == 'minutes':
        return round(elapsed.total_seconds()/60, digit)
    elif unit == 'seconds':
        return round(elapsed.total_seconds(), digit)
    else:
        raise NotImplementedError('Required unit is not implemented.')

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
