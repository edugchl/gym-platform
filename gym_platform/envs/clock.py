from datetime import datetime, timedelta

import numpy as np


class Clock():

    CYCLE_MAP = {
        'month': 12,
        'day': 31,
        'hour': 23,
        'minute': 59,
        'weekday': 6,
    }

    def __init__(self, ratio, artifical_start=None, real_start=None):
        """
        Args:
            - ratio: number of seconds in aritifical world per 1s of real world
        """
        self.ratio = ratio
        self.artifical_start = self.initialize(artifical_start)
        self.real_start = self.initialize(real_start)
        # for observation space in gym env
        self.low = np.ones(len(self.CYCLE_MAP))*-1
        self.high = np.ones(len(self.CYCLE_MAP))
        #TODO: current_time, getter, setter

    def time_features(self, dt: datetime, output: str = 'numpy') -> np.array:
        features = {}

        for timing, cycle in self.CYCLE_MAP.items():
            x = getattr(dt, timing)
            if callable(x):  # for dt.weekday()
                x = x() 
            feature = self.feature_vector(x, cycle)
            features[timing] = feature
        
        if output == 'numpy':
            return np.array(list(features.values())).flatten()
        elif output == 'dict':
            return features
        else:
            raise NotImplementedError('Unrecognized output option.')

    def feature_vector(self, x, cycle):
        """Encode the datetime and capture the cyclical pattern.
        https://www.kaggle.com/avanwyk/encoding-cyclical-features-for-deep-learning
        """
        sin_feature = np.sin(2 * np.pi * x/cycle)
        cos_feature = np.cos(2 * np.pi * x/cycle)
        return np.array([sin_feature, cos_feature])

    def new_aritifical_time(self, delta_real):
        elapsed = delta_real * self.ratio
        new = self.artifical_start + timedelta(seconds=elapsed) 
        return new

    def new_real_time(self, delta_aritifical):
        elapsed = delta_aritifical / self.ratio
        new = self.real_start + timedelta(seconds=elapsed)
        return new

    def initialize(self, start):
        now = datetime.now()
        if start == None:
            return now
        else:
            return self.convert(start)
        
    def convert(self, t):
        """str, int, datetime"""
        if isinstance(t, datetime):
            return t
        elif isinstance(t, float):
            return datetime.fromtimestamp(t)
        elif isinstance(t, str):
            from dateutil import parser
            return parser.parse(t)
        else:
            raise NotImplementedError('Unrecognized data type.')