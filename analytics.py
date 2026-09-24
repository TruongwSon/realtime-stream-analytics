import numpy as np


class KalmanFilter:

    def __init__(
        self,
        q=0.01,
        r=1
    ):

        # Process Noise
        self.Q = q

        # Measurement Noise
        self.R = r

        # Estimated state
        self.x = None

        # Error covariance
        self.P = 1



    def update(self, measurement):

        if self.x is None:
            self.x = measurement


        # Prediction step
        x_predict = self.x

        P_predict = self.P + self.Q



        # Kalman Gain
        K = P_predict / (
            P_predict + self.R
        )


        # Innovation residual
        error = measurement - x_predict



        # Update state estimation
        self.x = (
            x_predict +
            K * error
        )


        # Update uncertainty
        self.P = (
            1 - K
        ) * P_predict



        return self.x, error