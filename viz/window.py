import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class Window:
    """
    Window to draw a gridworld instance using Matplotlib
    """

    def __init__(self, title, n_imgs_row, show_local=False):
        self.fig = None

        self.imshow_obj1 = None
        self.show_local = show_local

        # Create the figure and axes
        if show_local:
            self.fig, self.axs = plt.subplots(
                n_imgs_row, 2 * n_imgs_row, width_ratios=[2, 1] * n_imgs_row
            )
        else:
            self.fig, self.axs = plt.subplots(n_imgs_row, n_imgs_row)
        self.axs = self.axs.reshape(-1)
        # Show the env name in the window title
        # self.fig.canvas.setWindowTitle(title)

        # Turn off x/y axis numbering/ticks
        for i in range(len(self.axs)):
            self.axs[i].xaxis.set_ticks_position("none")
            self.axs[i].yaxis.set_ticks_position("none")
            _ = self.axs[i].set_xticklabels([])
            _ = self.axs[i].set_yticklabels([])

        # Flag indicating the window was closed
        # self.closed = False

        # def close_handler(evt):
        #     self.closed = True

        # self.fig.canvas.mpl_connect("close_event", close_handler)

    def get_fig(self):
        return self.fig

    def _build_nested_pie(self, ax, local_map, n_arm_extensions=2, n_angles=8):
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

            ax.pie(
                v1,
                radius=radius,
                colors=colors.tolist(),
                labeldistance=1.1 * radius,
                labels=labels,
                wedgeprops=dict(width=white_width, edgecolor="w"),
                startangle=90 - (180 / n_angles),
            )

        ax.set(aspect="equal")

    def show_img(self, imgs_global, imgs_local=None, mode="human"):
        """
        Show an image or update the image being shown
        """
        if mode == "gif":
            matplotlib.use("Agg")

        if imgs_local is not None and self.show_local:
            for i, (img_global, img_local) in enumerate(zip(imgs_global, imgs_local)):
                # Show the first image of the environment
                j = i * 2
                self.axs[j].clear()
                self.axs[j + 1].clear()
                self.axs[j].imshow(img_global, interpolation="bilinear")
                self._build_nested_pie(self.axs[j + 1], img_local)
                self.axs[j].set_yticklabels([])
                self.axs[j].set_xticklabels([])
                if not mode == "gif":
                    self.fig.canvas.draw()
        else:
            for i, img_global in enumerate(imgs_global):
                # Show the first image of the environment
                self.axs[i].clear()
                self.axs[i].imshow(img_global, interpolation="bilinear")
                self.axs[i].set_yticklabels([])
                self.axs[i].set_xticklabels([])
                if not mode == "gif":
                    self.fig.canvas.draw()

        # Let matplotlib process UI events
        # This is needed for interactive mode to work properly
        plt.pause(0.001)

    def set_title(self, title, idx):
        self.axs[idx].set_title(title)

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
