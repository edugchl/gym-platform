

def time_elapsed(t1, t2, unit='hours', digit=2):
    elapsed = t2 - t1
    if unit == 'hours':
        return round(elapsed.total_seconds()/3600, digit)
    elif unit == 'minutes':
        return round(elapsed.total_seconds()/60, digit)
    elif unit == 'seconds':
        return round(elapsed.total_seconds(), digit)
    else:
        raise NotImplementedError('Required unit is not implemented.')
