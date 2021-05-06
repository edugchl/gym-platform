import unittest
from datetime import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


from gym_platform import User
from utils.visualize import plot_heatmap

class TestUser(unittest.TestCase):
    now = datetime.now()
    user = User('WHITE COLLAR', 0, now)

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


if __name__ == '__main__':
    unittest.main()