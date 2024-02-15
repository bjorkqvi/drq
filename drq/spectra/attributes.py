import numpy as np


class SpecAttributes:
    def m0(self, method: str = "integrate"):
        return self.m(moment=0, method=method)

    def hs(self, method: str = "integrate"):
        return 4 * np.sqrt(self.m0(method=method))

    def dtheta(self) -> float:
        try:
            return np.median(np.diff(self.theta()))
        except AttributeError:
            return None

    def dk(self) -> float:
        try:
            return np.median(np.diff(self.k()))
        except AttributeError:
            return None

    def df(self) -> float:
        try:
            return np.median(np.diff(self.freq()))
        except AttributeError:
            return None

    def dkx(self) -> float:
        try:
            return np.median(np.diff(self.kx()))
        except AttributeError:
            return None

    def dky(self) -> float:
        try:
            return np.median(np.diff(self.ky()))
        except AttributeError:
            return None
