import unittest
from datetime import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


from gym_platform import User
from utils.visualize import plot_heatmap

class TestUser(unittest.TestCase):
    now = datetime.now()
    user = User('WHITE COLLAR', 10, now)

    def test_job_freeness(self):
        data, label_x = [], []

        for day in range(1, 8):
            hour_data = []

            for hour in range(0, 24): 
                dt = datetime(self.now.year, self.now.month, day, hour)
                score = self.user.job_freeness(job=self.user.job_type, dt=dt)
                self.assertTrue(score >= 0 and score <= 1)
                hour_data.append(score)
            
            data.append(hour_data)
            label_x.append(dt.weekday())

        label_y = list(range(0, 24))
        data = np.array(data).T
        # plot
        fig = plot_heatmap(data, label_x, label_y, 'Weekday', 'Hour', 'Job Freeness')
        plt.show()

    def test_dependent_freeness(self):
        data, label_x = [], []
        num_range = list(range(0, 10))

        for day in range(1):
            num_data = []
            for no in num_range: 
                score = self.user.dependent_freeness(no)
                self.assertTrue(score >= 0 and score <= 1)
                num_data.append(score)
            
            data.append(num_data)
            label_x.append('NA')

        label_y = num_range
        data = np.array(data).T
        # plot
        fig = plot_heatmap(data, label_x, label_y, 'NA', 'Num Dependents', 'Dependent Freeness')
        plt.show()

    def test_freeness(self):
        data, label_x = [], []

        for day in range(1, 8):
            hour_data = []

            for hour in range(0, 24): 
                dt = datetime(self.now.year, self.now.month, day, hour)
                score = self.user.freeness(dt=dt)
                self.assertTrue(score >= 0 and score <= 1)
                hour_data.append(score)
            
            data.append(hour_data)
            label_x.append(dt.weekday())

        label_y = list(range(0, 24))
        data = np.array(data).T
        # plot
        fig = plot_heatmap(data, label_x, label_y, 'Weekday', 'Hour', 'Freeness')
        plt.show()

if __name__ == '__main__':
    unittest.main()