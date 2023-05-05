import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class Window:
    """
    Window to draw a gridworld instance using Matplotlib
    """

    def __init__(self, title):
        self.fig = None

        self.imshow_obj1 = None

        # Create the figure and axes
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, width_ratios=[2, 1])

        # Show the env name in the window title
        # self.fig.canvas.setWindowTitle(title)

        # Turn off x/y axis numbering/ticks
        self.ax1.xaxis.set_ticks_position("none")
        self.ax1.yaxis.set_ticks_position("none")
        _ = self.ax1.set_xticklabels([])
        _ = self.ax1.set_yticklabels([])

        # Flag indicating the window was closed
        # self.closed = False

        # def close_handler(evt):
        #     self.closed = True

        # self.fig.canvas.mpl_connect("close_event", close_handler)

    def get_fig(self):
        return self.fig

    def _build_nested_pie(self, local_map, n_arm_extensions=2, n_angles=8):
        size = 0.3
        v1 = np.ones((n_angles,))
        vmax = 5
        vmin = -5

        local_map = (local_map.T).astype(np.float32)

        def array_to_colors(a):
            vvmax = max(vmax, a.max())
            vvmin = min(vmin, a.min())
            return ((a - vvmin) / (vvmax - vvmin))[:, None].repeat(3, axis=-1)

        for i in range(n_arm_extensions):
            labels = [str(x) for x in local_map[i].astype(np.int16)]
            radius = (i + 1) / n_arm_extensions
            colors = array_to_colors(local_map[i])
            white_width = size * radius

            self.ax2.pie(
                v1,
                radius=radius,
                colors=colors.tolist(),
                labeldistance=1.1 * radius,
                labels=labels,
                wedgeprops=dict(width=white_width, edgecolor="w"),
                startangle=90 - (180 / n_angles),
            )

        self.ax2.set(aspect="equal")

    def show_img(self, img_global, img_local, mode):
        """
        Show an image or update the image being shown
        """
        if mode == "gif":
            matplotlib.use("Agg")

        # Show the first image of the environment
        self.ax1.clear()
        self.ax2.clear()
        self.ax1.imshow(img_global, interpolation="bilinear")
        self._build_nested_pie(img_local)
        if not mode == "gif":
            self.fig.canvas.draw()

        # Let matplotlib process UI events
        # This is needed for interactive mode to work properly
        plt.pause(0.001)

    def set_title(self, title):
        self.ax1.set_title(title)

    def set_caption(self, text):
        """
        Set/update the caption text below the image
        """

        plt.xlabel(text)

    def reg_key_handler(self, key_handler):
        """
        Register a keyboard event handler
        """

        # Keyboard handler
        self.fig.canvas.mpl_connect("key_press_event", key_handler)

    def show(self, block=True):
        """
        Show the window, and start an event loop
        """

        # If not blocking, trigger interactive mode
        if not block:
            plt.ion()

        # Show the plot
        # In non-interative mode, this enters the matplotlib event loop
        # In interactive mode, this call does not block
        plt.show()

    def close(self):
        """
        Close the window
        """

        plt.close()
        self.closed = True
