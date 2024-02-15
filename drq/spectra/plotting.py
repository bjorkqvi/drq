import matplotlib.pyplot as plt


def get_fig_ax(ax, return_handle):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
        return_handle = True

    return fig, ax, return_handle


def return_fig_ax(fig, ax, return_handle):
    if not return_handle:
        fig.show()
        return None
    elif fig is None:
        return ax
    else:
        return fig, ax


class Plotting1D:
    def plot(self, ax=None, power: float = 0.0, return_handle: bool = False):
        fig, ax, return_handle = get_fig_ax(ax, return_handle)
        x = self.get(self.x_label)
        y = self.spec(squeeze=True) * x**power
        ax.plot(x, y)
        ax = self._add_text(ax)
        return return_fig_ax(fig, ax, return_handle)

    def loglog(self, ax=None, power: float = 0.0, return_handle: bool = False):
        fig, ax, return_handle = get_fig_ax(ax, return_handle)
        x = self.get(self.x_label)
        y = self.spec(squeeze=True) * x**power
        ax.loglog(x, y)
        ax = self._add_text(ax)
        return return_fig_ax(fig, ax, return_handle)

    def _add_text(self, ax):
        ax.set_xlabel(f"{self.x_label} ({self.x_unit})")
        ax.set_ylabel(f"{self.y_label} ({self.y_unit})")
        return ax
