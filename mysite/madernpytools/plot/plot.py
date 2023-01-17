import numpy as np
import scipy as sp
import scipy.linalg

import matplotlib.lines
import matplotlib.colorbar
import matplotlib.patches as patches
import matplotlib.cm as cm

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.text import Annotation
from matplotlib.animation import FuncAnimation

from madernpytools.backbone import EventSubscriber
import madernpytools.models.toolset_model as mts

cols = cm.Set1(np.linspace(0, 1, 10))[:, :3]

def periodic_clip(val, n_min, n_max):
    ''' keeps val within the range [n_min, n_max) by assuming that val is a periodic value'''
    if val < n_max and val >= n_min:
        val = val
    elif val >= n_max:
        val = val - (n_max - n_min)
    elif val < n_max:
        val = val + (n_max - n_min)

    return val


def get_points(mu, sigma, n_rings, n_points, n_std=1):
    # Compute eigen components:
    (D0, V0) = np.linalg.eig(sigma)
    U0 = np.real(V0.dot(np.diag(D0) ** 0.5) * n_std)

    # Compute first rotational path
    psi = np.linspace(0, np.pi * 2, n_rings, endpoint=True)
    ringpts = np.vstack((np.zeros((1, len(psi))), np.cos(psi), np.sin(psi)))

    U = np.zeros((3, 3))
    U[:, 1:3] = U0[:, 1:3]
    ringtmp = U.dot(ringpts)

    # Compute touching circular paths
    phi = np.linspace(0, np.pi, n_points)
    pts = np.vstack((np.cos(phi), np.sin(phi), np.zeros((1, len(phi)))))

    xring = np.zeros((n_rings, n_points, 3))
    for j in range(n_rings):
        U = np.zeros((3, 3))
        U[:, 0] = U0[:, 0]
        U[:, 1] = ringtmp[:, j]
        xring[j, :] = (U.dot(pts).T + mu)

    # Reshape points in 2 dimensional array:
    return xring.reshape((n_rings * n_points, 3))


def tri_ellipsoid(n_rings, n_points):
    ''' Compute the set of triangles that covers a full ellipsoid of n_rings with n_points per ring'''
    tri = []
    for n in range(n_points - 1):
        # Triange down
        #       *    ring i+1
        #     / |
        #    *--*    ring i
        tri_up = np.array([n, periodic_clip(n + 1, 0, n_points),
                           periodic_clip(n + n_points + 1, 0, 2 * n_points)])
        # Triangle up
        #    *--*      ring i+1
        #    | /
        #    *    ring i

        tri_down = np.array([n, periodic_clip(n + n_points + 1, 0, 2 * n_points),
                             periodic_clip(n + n_points, 0, 2 * n_points)])

        tri.append(tri_up)
        tri.append(tri_down)

    tri = np.array(tri)
    trigrid = tri
    for i in range(1, n_rings - 1):
        trigrid = np.vstack((trigrid, tri + n_points * i))

    return np.array(trigrid)


class SquarePatch(object):

    def __init__(self, ax: Axes, p1=None, p2=None, artist='patches', **properties):
        """
        @param ax: Axis to which patch should be placed
        @param p1: p1 left upper corner
        @param p2: p2 right lower corner
        @param artist: which artist to use to place the patch ('patches/artists')
        @param properties: additional patch properties
        """

        self._ax = ax
        self._codes = [
            Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.CLOSEPOLY,
        ]

        self._patch = None
        self._properties = properties
        self._artist = artist

        if not (p1 is None or p2 is None):
            self.update(p1, p2)

    def update(self, p1, p2, update_canvas=False):
        # Define square:
        verts = [
            (p1[0], p1[1]),  # left, bottom
            (p1[0], p2[1]),  # left, bottom
            (p2[0], p2[1]),  # left, bottom
            (p2[0], p1[1]),  # left, bottom
            (p2[0], p1[1]),  # left, bottom
        ]
        path = Path(verts, self._codes)

        # Remove existing patches
        if self._patch is not None:
            self._patch.remove()

        # Add new patches
        self._patch = patches.PathPatch(path, **self._properties)
        if self._artist == 'artists':
            self._ax.add_artist(self._patch)
        elif self._artist == 'patches':
            self._ax.add_patch(self._patch)
        else:
            raise RuntimeError('Cannot add patch to {0}. Use either \'artists\' or \'patches\'.'.format(self._artist))

        # Update canvas
        if update_canvas:
            self._ax.figure.canvs.draw_idle()


class Line2D(EventSubscriber):

    def __init__(self, ax, **lineprops):
        self._line, = ax.plot([], [], **lineprops)

    def update(self, publisher):
        self.set_data(publisher.get_data())

    def set_data(self, data):
        """Sets new data

        :param data: N x 2 ndarray of data samples. First column refers to x, second column to y
        """
        self._line.set_data(*data.T)

    def set_color(self, color):
        self._line.set_color(color)


class PSDLines(object):

    def __init__(self, ax, spectral_data=None, plot_keys=None, **kwargs):
        if plot_keys is None:
            plot_keys = ['Sxx', 'Sxf', 'Sff', 'H']

        labels = kwargs.pop('labels', True)
        self._lines = {}
        for i, key in enumerate(plot_keys):
            label = key if labels else ''
            self._lines[key] = Line2D(ax, color=cols[i, :], label=label, **kwargs)

        if spectral_data is not None:
            self.set_data(spectral_data)

    def set_data(self, spectral_data):
        """Set spectral data"""
        for key in self._lines.keys():
            if key in spectral_data.keys():
                self._lines[key].set_data(np.vstack([spectral_data.get('freq'),
                                                     self._to_db(spectral_data.get(key))]
                                                    ).T
                                          )

    def _to_db(self, value):
        return 20 * np.log10(abs(value))

    def set_db_conversion(self, f_handle):
        """Sets the conversion from Spectral data into magnitude"""
        self._to_db = f_handle

    @property
    def colordict(self):
        colordict = {}
        for key, line in self._lines.items():
            colordict[key] = line.get_color()

    @colordict.setter
    def colordict(self, colordict):
        for key, line in self._lines.items():
            line.set_color(colordict[key])

    @property
    def linewidth(self):
        key = self._lines.keys()[0]
        return self._lines[key].get_linewidth()

    @linewidth.setter
    def linewidth(self, value):
        for key, line in self._lines.items():
            line.set_linewidth(value)

    @property
    def linestyle(self):
        key = self._lines.keys()[0]
        return self._lines[key].get_linestyle()

    @linestyle.setter
    def linestyle(self, value):
        for key, line in self._lines.items():
            line.set_linestyle(value)

    def set_color(self, color, key=None):
        if key is not None:
            self._lines[key].set_color(color)


class SpectralPlot(object):

    def __init__(self, spectral_data=None, axs=None, **kwargs):
        self._lines = {}

        # Pop settings for this class:
        plot_coherence = kwargs.pop('plot_coherence', True)
        plot_samples = kwargs.pop('plot_samples', False)
        xlim = kwargs.pop('xlim', [spectral_data.frequencies.min(), spectral_data.frequencies.max()] if
                          spectral_data is not None else [0, 1500])

        # Determine to make fig and or axis:
        if axs is not None:
            self._axs = axs
            self._fig = self._axs['magnitude'].figure
        else:
            if 'figure' in kwargs.keys():
                self._fig = kwargs.pop('figure')
            else:
                self._fig = Figure(figsize=(10, 5))

            self._axs = {}
            self._axs['magnitude'] = self._fig.add_subplot(111)
            self._axs['coherence'] = self._axs['magnitude'].twinx()

        self._axs['magnitude'].set_xlim(xlim)
        self._axs['magnitude'].set_ylabel('Magnitude [dB]')
        self._axs['magnitude'].set_xlabel('Freq [Hz]')
        self._axs['coherence'].set_ylabel('Coherence [-]', color='gray')

        # Initialize lines:
        print('Initializing lines...')
        self._PSD_mean = PSDLines(self._axs['magnitude'], linewidth=2, **kwargs)
        self._PSD_samples = []

        self._Cxf_mean = self._axs['coherence'].plot([], [], color='gray', linewidth=2, alpha=0.5)
        self._Cxf_samples = []
        self._eigenfreqs = []

        self._spectral_data = None

        if spectral_data is not None:
            self.set_data(spectral_data, plot_coherence=plot_coherence,
                          plot_samples=plot_samples)

    def grid(self, value=True):
        """Set grid value"""
        self._axs['magnitude'].grid(value)

    @property
    def fig(self):
        return self._fig

    @property
    def xlim(self):
        return self._axs['magnitude'].get_xlim()

    @xlim.setter
    def xlim(self, xlim):
        self._axs['magnitude'].set_xlim(xlim)

    def set_colordict(self, color_dict):
        """ Set color dictionary

        :param color_dict:
        :return:
        """
        self._PSD_mean.colordict = color_dict

        for item in self._PSD_samples:
            item.colordict = color_dict

    def show_eigen_frequencies(self, freqs, **kwargs):
        """Plot eigen frequencies"""

        lw = kwargs.pop('linewidth', 1)
        col = kwargs.pop('color', 'red')
        linestyle = kwargs.pop('linestyle', '--')

        ylim = self._axs['magnitude'].get_ylim()
        xlim = self._axs['magnitude'].get_xlim()

        for w in [f for f in freqs if (xlim[0] < f) and (f < xlim[1]) ]:
            l, = self._axs['magnitude'].plot([w, w], ylim, linewidth=lw, color=col,
                                             linestyle=linestyle)
            t = self._axs['magnitude'].text(w, ylim[0], r'${0:.0f}$ [Hz] '.format(w),
                                            horizontalalignment='left',
                                            verticalalignment='bottom',
                                            rotation=-90)
            self._eigenfreqs.append((l, t))

    def legend(self, *args, **kwargs):
        self._axs['magnitude'].legend(*args, **kwargs)

    def set_data(self, spectral_data, plot_coherence=True, plot_samples=False):
        '''Plot frequency response
        '''

        self._spectral_data = spectral_data

        # Plot average results:
        self._PSD_mean.set_data(spectral_data)

        # Plot samples:
        if plot_samples:
            for tmp in spectral_data:
                self._PSD_samples.append(PSDLines(self._axs['magnitude'], spectral_data=tmp,
                                                  linewidth=0.5, alpha=0.5,
                                                  labels=False)
                                         )

        # Update xlim:
        xlim = self._axs['magnitude'].get_xlim()
        self._axs['magnitude'].relim()
        self._axs['magnitude'].autoscale_view()
        self._axs['magnitude'].set_xlim(xlim)

        if 'Cxf' in spectral_data.keys() and plot_coherence:
            self._axs['coherence'].plot(spectral_data.frequencies,
                                        spectral_data.Cxf,
                                        color='gray', linewidth=2, alpha=0.5)
            if plot_samples:
                for tmp in spectral_data:
                    l = self._axs['coherence'].plot(tmp['freq'], tmp['Cxf'],
                                          color='grey', linewidth=0.5, alpha=0.5)
                    self._Cxf_samples.append(l)

            self._axs['coherence'].set_ylim([0, 1.05])


class ToolsetSkeleton(object):

    def __init__(self, ax, q, l, alpha=1.0, x_shift=0):

        self._ax = ax
        self._l = l
        self._q = q
        self._x_shift = x_shift

        # Plot objects:
        self._lines = {}
        self._lines['uc'], = ax.plot([],[], '.-', alpha=alpha, color='orange', lw=1)
        self._lines['lc'], = ax.plot([],[], '.-', alpha=alpha, color='purple', lw=1)
        self._lines['bbuOS'], = ax.plot([],[], alpha=alpha, color='red', lw=2)
        self._lines['bblOS'], = ax.plot([],[], alpha=alpha, color='blue', lw=2)
        self._lines['bbuDS'], = ax.plot([],[], alpha=alpha, color='red', lw=2)
        self._lines['bblDS'], = ax.plot([],[], alpha=alpha, color='blue', lw=2)
        self._lines['gu'], = ax.plot([],[], alpha=alpha, color='gray', lw=2)
        self._lines['gl'], = ax.plot([],[], alpha=alpha, color='gray', lw=2)

        self._points = {}
        self._points['bbuOS'], = ax.plot([],[], '.', alpha=alpha, color='red')
        self._points['bblOS'], = ax.plot([],[], '.', alpha=alpha, color='blue')
        self._points['bbuDS'], = ax.plot([],[], '.', alpha=alpha, color='red')
        self._points['bblDS'], = ax.plot([],[], '.', alpha=alpha, color='blue')

        self.update(self._q)

    def update(self, q, rescale=True):
        self._q = q
        self._draw_upper_cylinder()
        self._draw_lower_cylinder()

        # Bearing blocks:
        self._draw_bearing_block(self._lines['bbuOS'], self._points['bbuOS'],
                                 x=self._q['x_{0}'.format('bb_uOS')]-0.2,
                                 y=-self._l+self._x_shift)
        self._draw_bearing_block(self._lines['bblOS'], self._points['bblOS'],
                                 x=self._q['x_{0}'.format('bb_uOS')]+0.2,
                                 y=-self._l+self._x_shift)
        self._draw_bearing_block(self._lines['bbuDS'], self._points['bbuDS'],
                                 x=self._q['x_{0}'.format('bb_uDS')]-0.2,
                                 y=self._l+self._x_shift)
        self._draw_bearing_block(self._lines['bblDS'], self._points['bblDS'],
                                 x=self._q['x_{0}'.format('bb_uDS')]+0.2,
                                 y=self._l+self._x_shift)

        # Gears
        self._draw_gear(self._lines['gl'],
                        x=self._q['x_{0}'.format('gear_lower')]-0.2,
                        y=self._l+0.1+self._x_shift)
        self._draw_gear(self._lines['gu'],
                        x=self._q['x_{0}'.format('gear_up')]+0.2,
                        y=self._l+0.1+self._x_shift)

        if rescale:
            self._ax.relim()
            self._ax.autoscale_view()

    def _draw_upper_cylinder(self):
        self._draw_cylinder(self._lines['uc'],
                            x=self._q['x_cyl_up'] + 0.2,
                            theta=self._q['th_cyl_up'])

    def _draw_lower_cylinder(self):
        self._draw_cylinder(self._lines['lc'],
                            x=self._q['x_cyl_low'] - 0.2,
                            theta=self._q['th_cyl_low'])

    def _draw_cylinder(self, line: matplotlib.lines.Line2D, x, theta):
        R = self._R(theta)
        v = np.array([[-self._l, 0, self._l],
                      [0               , 0, 0               ]
                      ]
                    )
        v_rot = R.dot(v)
        line.set_data(v_rot[0,:]+self._x_shift, v_rot[1,:]+x)

    def _draw_bearing_block(self, line: matplotlib.lines.Line2D, point: matplotlib.lines.Line2D, x, y):

        line.set_data([y, y,  y],
                 [x-0.2, x, x+0.2])
        point.set_data([y], [x])

    def _draw_gear(self, line, x, y):
        line.set_data([y, y,  y], [x-0.15, x, x+0.15])

    def _R(self, theta):
        return np.array([[ np.cos(theta), np.sin(theta)],
                         [-np.sin(theta), np.cos(theta)]]
                       )


class DeflectionPlot(object):

    def __init__(self, ax):
        self._upcyl = None
        self._lowcyl = None
        self._dimension_model = None
        self._lines = {
            'lower cylinder': matplotlib.lines.Line2D([], [], label='Lower cylinder', color=cols[0,]),
            'upper cylinder': matplotlib.lines.Line2D([], [], label='Upper cylinder', color=cols[1,]),
            'deflection': matplotlib.lines.Line2D([], [], label='Total deflection', color=cols[2,])
        }

        for _, l in self._lines.items():
            ax.add_line(l)

    @property
    def lines(self):
        return self._lines

    def update(self, ts):
        self._update_plot(ts)

    def _update_plot(self, ts):
        a = ts.layout.max_axial_line / 2
        b = ts.upper_cylinder.br_location
        c = ts.upper_cylinder.body.length / 2
        d = c + ts.upper_bearing_block.s

        x_val = np.linspace(0, d, 100)
        up_defl = -ts.deflections.upper_cylinder.deflection(x_val) * 1e3
        up_defl += ts.deflections.upper_cylinder.deflection(b) * 1e3

        low_defl = -ts.deflections.lower_cylinder.deflection(x_val) * 1e3
        low_defl += ts.deflections.lower_cylinder.deflection(b) * 1e3

        x_valb = np.linspace(0, b, 100)
        dg_b = -ts.deflections.gap_deflection(x_valb) * 1e3
        dg_b += ts.deflections.gap_deflection(b) * 1e3

        # Plot results:
        self._lines['lower cylinder'].set_data(*DeflectionPlot.mirror_points(x_val, low_defl))
        self._lines['upper cylinder'].set_data(*DeflectionPlot.mirror_points(x_val, up_defl))
        self._lines['deflection'].set_data(*DeflectionPlot.mirror_points(x_valb, dg_b))

    @staticmethod
    def mirror_points(x,y):
        return np.hstack([-x[::-1], x]), np.hstack([y[::-1], y])




class ModeMotionPlot(object):

    def __init__(self, ax, v, l=0.7, N=40, alpha=0.1):

        self._N = 40
        self._dx = 0.1 / self._N
        self._phases = np.sin(np.linspace(0, np.pi * 2, self._N, endpoint=True))

        # State Var:
        self._skeletons = []
        for i in range(self._N):
            q = dict(zip(mts.DynamicsAnalyzer.state_var, v * np.sin(self._phases[i])))
            self._skeletons.append(ToolsetSkeleton(ax, q=q, l=l, x_shift=self._dx * i, alpha=alpha))

    def update(self, v):
        for i, item in enumerate(self._skeletons):
            q = dict(zip(mts.DynamicsAnalyzer.state_var, v * np.sin(self._phases[i])))
            item.update(q)


class ToolsetSkeletonPlot(object):

    def __init__(self, ax, q, pars, alpha=1.0, x_shift=0):
        """ Creates a tool skeleton plot.

        :param ax: Axis to plot to
        :param q: state-variable dictionary containing keys: x_uc x_lc theta_uc theta_lc x_bblOS x_bbuOS x_bblDS
        x_bbuDS x_suOS x_slOS x_suDS x_slDS x_gu x_gl
        :param pars: toolset parameters (currently dictionary {l: <value>}
        :param alpha: transparency of plot
        :param x_shift:
        """

        raise RuntimeWarning("ToolSkeletonPlot is depriated, please use ToolSkeleton instead")
        self._object = ToolsetSkeleton(ax, q, l=pars['l'], alpha=alpha, x_shift=x_shift)

    def update(self, q):
        self._object.update(q)


def plot_mode(ax, mode, **kwargs):
    """Plot mode on the provided ax"""
    dis = mode.shape.shape/np.abs(mode.shape.shape).max()
    label = kwargs.pop('label', '{0:.2f}[Hz]'.format(mode.natural_frequency))
    ax.plot(mode.locations, dis, label=label, **kwargs)

class AbstractGaussianGraphic(object):
    def __init__(self, ax, mu, sigma):
        mes = """The init class is not implemented. It should add the objects required 
               for GaussianGraphic instance to the axis. The references to 
               these objects should be stored internally such that they
               can be changed by the set_[property]() methods."""
        raise NotImplementedError(mes)

    # Shared properties:
    def set_color(self, color):
        raise NotImplementedError()

    def get_color(self):
        raise NotImplementedError()

    def get_label(self):
        raise NotImplementedError()

    def set_data(self, mu, sigma):
        raise NotImplementedError()

    def set_alpha(self, alpha):
        raise NotImplementedError()

    # Contour properties
    def set_contouralpha(self, alpha):
        raise NotImplementedError()

    def set_contourwidth(self, width):
        raise NotImplementedError()

    # Center properties:
    def set_centersize(self, centersize):
        raise NotImplementedError()


class GaussianPatch3d(AbstractGaussianGraphic):
    def __init__(self, ax, mu, sigma, n_rings=20, n_points=30,
                 color='red', alpha=0.5, contourwidth=2,
                 n_std=1, label=''):
        # Save properties:
        self._ax = ax
        self._n_points = n_points
        self._n_rings = n_rings
        self._alpha = alpha
        self._linewidth = contourwidth
        self._linecolor = color
        self._patchcolor = color
        self._sigma = sigma
        self._mu = mu
        self._surf = None
        self._label = label
        self._n_std = n_std
        self._update()

    def _update(self):
        points = get_points(self._mu, self._sigma, self._n_rings, self._n_points, self._n_std)
        triangles = tri_ellipsoid(n_rings=self._n_rings, n_points=self._n_points)

        # Plot surface:
        if self._surf is not None:
            self._surf.remove()

        # Create new:
        self._surf = self._ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=triangles,
                                           linewidth=self._linewidth,
                                           alpha=self._alpha,
                                           color=self._patchcolor,
                                           edgecolor=self._linecolor)

    # Shared properties:
    def set_color(self, color):
        self._linecolor = color
        self._patchcolor = color
        self._surf.set(edgecolor=color, facecolor=color)

    def get_color(self):
        return self._surf.get_color()

    def get_label(self):
        return self._label

    def set_data(self, mu, sigma):
        self._mu = mu
        self._sigma = sigma
        self._update()

    def set_alpha(self, alpha):
        self._surf.set(alpha=alpha)

    # Contour properties
    def set_contouralpha(self, alpha):
        raise NotImplementedError()

    def set_contourwidth(self, width):
        self.set(linewidth=width)

    # Center properties:
    def set_centersize(self, centersize):
        raise NotImplementedError()

    def set(self, **kwargs):
        self._surf.set(**kwargs)


class GaussianPatch2d(AbstractGaussianGraphic):
    def __init__(self, ax, mu, sigma, n_segments=35,
                 color='red', facealpha=0.5, centersize=8,
                 linewidth=1, contouralpha=1, contourwidth=2,
                 n_std=1, label=''):
        # Create line object:
        self._n_segments = n_segments
        self._label = label
        self._n_std = n_std

        # Generate points
        points = GaussianPatch2d.get_contourpoints(mu, sigma, n_segments, self._n_std)

        # polygon
        self._center, = ax.plot(mu[0], mu[1], '.', markersize=centersize,
                                color=color, alpha=contouralpha)
        self._contour, = ax.plot(points[:, 0], points[:, 1],
                                 color=color, alpha=contouralpha,
                                 linewidth=contourwidth)
        self._polygon = patches.Polygon(points, color=color, alpha=facealpha)
        ax.add_patch(self._polygon)

    @staticmethod
    def get_contourpoints(mu, sigma, n_segments, n_std):
        """Get contour points for a 2d Gaussian distribution"""
        # Compute
        t = np.linspace(-np.pi, np.pi, n_segments);
        R = np.real(sp.linalg.sqrtm(n_std * sigma))

        return (R.dot(np.array([[np.cos(t)], [np.sin(t)]]).reshape([2, len(t)])).T + mu)

    def get_label(self):
        return self._label

    # Shared properties:
    def set_color(self, color):
        """Set color of Gaussian contour, center and transparant patches."""
        self._center.set_color(color)
        self._contour.set_color(color)
        self._polygon.set_color(color)

    def get_color(self):
        return self._center.get_color()

    def set_data(self, mu, sigma):
        """Update data of Gaussian."""
        # Update point
        self._center.set_data(mu)

        # Update contour:
        points = GaussianPatch2d.get_contourpoints(mu, sigma, self._n_segments, self._n_std)
        self._contour.set_data(points[:, 0], points[:, 1])
        self._polygon.set_xy(points)

    def set_alpha(self, alpha):
        self._polygon.set_alpha(alpha)

    # Contour properties
    def set_contouralpha(self, alpha):
        """Set transparency of Gaussian"""
        self._contour.set_alpha(alpha)
        self._center.set_alpha(alpha)

    def set_contourwidth(self, width):
        """Set width of the contour"""
        self._contour.set_linewidth(width)

    # Center properties:
    def set_centersize(self, centersize):
        """Set the size of the center"""
        self._center.set_linewidth(centersize)


class GaussianPatch1d(AbstractGaussianGraphic):
    def __init__(self, ax, mu, sigma, position=0.0,
                 direction='hor',
                 color='red',
                 mirrorpatch=False,
                 scale=1,
                 n_std=5,
                 contourwidth=2,
                 contouralpha=1,
                 facealpha=0.5,
                 label=''):

        self._ax = ax  # axis to plot to
        self._mu = mu  # mean (scalar)
        self._sigma = sigma  # variance (scalar)
        self._position = position  # Position of the 'baseline'
        self._color = color  # Color
        self._contourwidth = contourwidth
        self._contouralpha = contouralpha
        self._facealpha = facealpha
        self._nstd = n_std  # Number of standard deviations to plot
        self._scale = scale  # Scale of the likelihood values
        self._npoints = int(10 * n_std)  # Number of points to plot
        self._direction = direction  # Direction: 'vert' or 'hor'
        self._mirrorpatch = mirrorpatch
        self._label = ''

        self._patch = None
        self._contour, = self._ax.plot([], [], linewidth=self._contourwidth,
                                       alpha=self._contouralpha, color=self._color, label=label)

        self.update()

    @staticmethod
    def get_contourpoints(mu, sigma, n_std, n_points):
        ext = np.sqrt(sigma) * n_std
        y = np.linspace(-ext, ext, n_points) + mu
        lik = np.exp(-0.5 * (y - mu) ** 2 / sigma)
        lik /= lik.max()

        return (y, lik)

    def update(self):

        # Get points:
        (vals, lik) = GaussianPatch1d.get_contourpoints(
            self._mu, self._sigma,
            self._nstd, self._npoints)
        # Create mesh:
        vals = np.append(vals, vals[-1])

        if self._mirrorpatch:
            lik *= -1
        lik = np.append(lik, lik[-1]) * self._scale + self._position

        msh = np.vstack([vals, lik]).T
        if self._direction == 'vert':
            msh = np.roll(msh, 1, axis=1)

        # Create codes for path:
        codes = [Path.MOVETO]
        codes.extend([Path.LINETO] * (msh.shape[0] - 2))
        codes.extend([Path.CLOSEPOLY])

        path = Path(msh, codes)
        # Remove old patches:
        if self._patch is not None:
            self._patch.remove()
        self._patch = patches.PathPatch(path, alpha=self._facealpha,
                                        color=self._color)
        self._ax.add_patch(self._patch)
        # Update contour:
        if self._direction == 'vert':
            self._contour.set_data(lik[:-1], vals[:-1])
        else:
            self._contour.set_data(vals[:-1], lik[:-1])
        # self._ax.plot(vals, lik)

    def set_color(self, color):
        self._color = color
        self._contour.set_color(color)
        self._patch.set_color(color)

    def get_color(self):
        return self._color

    def get_label(self):
        return self._label

    def set_data(self, mu, sigma):
        self._mu = mu
        self._sigma = sigma
        self.update()

    def set_alpha(self, alpha):
        self._facealpha = alpha
        self._patch.set_alpha(alpha)

    # Contour properties
    def set_contouralpha(self, alpha):
        self._contouralpha = alpha
        self._contour.set_alpha(alpha)

    def set_contourwidth(self, width):
        self._contour.set_linewidth(width)

    # Center properties:
    def set_centersize(self, centersize):
        raise NotImplementedError()


class GaussianGraphicList(AbstractGaussianGraphic, list):
    def __init__(self, *args):
        """Initalize list of Gaussian Graphics"""
        list.__init__(self, *args)

    def __getitem__(self, sl):
        return GaussianGraphicList(list.__getitem__(self, sl))

    def append(self, item):
        if issubclass(type(item), AbstractGaussianGraphic):
            list.append(self, item)
        else:
            raise TypeError("Item is not of type {0}".format(AbstractGaussianGraphic))

    # Shared properties:
    def set_color(self, color):
        for g in self:
            g.set_color(color)
        self._color = color

    def get_color(self):
        return self[0].get_color()

    def get_label(self):
        return self[0].get_label()

    def set_data(self, mu, sigma):
        for g in self:
            g.set_data(mu, sigma)

    def set_alpha(self, alpha):
        for g in self:
            g.set_alpha(alpha)

    # Contour properties
    def set_contouralpha(self, alpha):
        for g in self:
            g.set_contouralpha(alpha)

    def set_contourwidth(self, width):
        for g in self:
            g.set_contourwidth(width)

    # Center properties:
    def set_centersize(self, centersize):
        for g in self:
            g.set_centersize(centersize)

class IBlitManager(object):

    def add_artist(self, art):
        raise NotImplementedError()


class BlitManager(IBlitManager):

    def __init__(self, canvas, animated_artists=()):
        """
        Got this from
        https://matplotlib.org/stable/tutorials/advanced/blitting.html


        Parameters
        ----------
        canvas : FigureCanvasAgg
            The canvas to work with, this only works for sub-classes of the Agg
            canvas which have the `~FigureCanvasAgg.copy_from_bbox` and
            `~FigureCanvasAgg.restore_region` methods.

        animated_artists : Iterable[Artist]
            List of the artists to manage
        """
        self.canvas = canvas
        self._bg = None
        self._artists = []

        for a in animated_artists:
            self.add_artist(a)
        # grab the background on every draw
        self.cid = canvas.mpl_connect("draw_event", self.on_draw)

    def on_draw(self, event):
        """Callback to register with 'draw_event'."""
        cv = self.canvas
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        self._bg = cv.copy_from_bbox(cv.figure.bbox)
        self._draw_animated()

    def add_artist(self, art):
        """
        Add an artist to be managed.

        Parameters
        ----------
        art : Artist

            The artist to be added.  Will be set to 'animated' (just
            to be safe).  *art* must be in the figure associated with
            the canvas this class is managing.

        """
        if art.figure != self.canvas.figure:
            raise RuntimeError
        art.set_animated(True)
        self._artists.append(art)

    def _draw_animated(self):
        """Draw all of the animated artists."""
        fig = self.canvas.figure
        for a in self._artists:
            fig.draw_artist(a)

    def update(self):
        """Update the screen with animated artists."""
        cv = self.canvas
        fig = cv.figure
        # paranoia in case we missed the draw event,
        if self._bg is None:
            self.on_draw(None)
        else:
            # restore the background
            cv.restore_region(self._bg)
            # draw all of the animated artists
            self._draw_animated()
            # update the GUI state
            cv.blit(fig.bbox)
        # let the GUI event loop process anything it has to do
        cv.flush_events()


class ResetableFuncAnimation(FuncAnimation):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self):
        self.event_source.stop()
        self._blit_cache.clear()
        self._init_draw()
        self._resize_id = self._fig.canvas.mpl_connect('draw_event', self._end_redraw)
        self._fig.canvas.draw_idle()


class InteractiveAnnotationData(object):

    def __init__(self, x, y, text):
        """ Data structure for Interactive Annotation

        :param x: X-Coordinate of annotation data
        :param y: Y-Coordinate of annotation data
        :param text: Text of annotation
        """
        self._x = x
        self._y = y
        self._text = text

    @property
    def x(self):
        """ X-coordinate of annotation

        :return:
        """
        return self._x

    @property
    def y(self):
        """ Y-coordinate of annotation

        :return:
        """
        return self._y

    @property
    def text(self):
        """ Annotation text

        :return:
        """
        return self._text


class InteractiveAnnotation(object):

    def __init__(self, ax, content_callback, event_name="button_press_event", **kwargs):
        """ Provides an interactive annotation for user interaction.

        When the selected event becomes active, the content_callback function is called. If the callback function
        returns a InteractiveAnnotationData object, an annotation is drawn onto the canvas. Otherwise, the annotation
        is removed from Canvas.

        This class relies on Blitting, providing a fast drawing

        :param ax: Axis on which the annotion should appear
        :param content_callback: Callback to call when event is triggered func(event) -> InteractiveAnnotationData
        :param event_name: Name of an event to respond to, defaults to 'button_press_event'
        :param kwargs: Any optional arguments are passed to Annotation object
        """

        # Store relevant figure variables:
        self._ax = ax
        self._canvas = ax.figure.canvas

        # Create annotation on given axis
        self._annotation = self._get_annotation(self._ax, **kwargs)

        # Create interactive component:
        self._blit_manager = BlitManager(self._canvas, animated_artists=(self._annotation,))
        self._content_callback = content_callback
        self._canvas.mpl_connect(event_name, self._event_callback)

    def _get_annotation(self, ax, **kwargs):
        """ Create Annotation

        :param ax:
        :param kwargs:
        :return:
        """
        annotation = ax.annotate(text='', xy=(0, 0),
                                 xytext=(20, 20),
                                 xycoords=ax.transData,
                                 bbox=dict(boxstyle="round", fc="w"),
                                 textcoords="offset points",
                                 arrowprops=dict(arrowstyle="->"),
                                 **kwargs
                                 )
        annotation.set_animated(True)
        annotation.set_visible(False)
        return annotation

    @property
    def annotation(self) -> Annotation:
        """ Annotation used by Interactive annotation

        :return:
        """
        return self._annotation

    def _event_callback(self, event):
        """ Handle event callback

        :param event:
        :return:
        """
        if event.inaxes == self._ax:
            resp = self._content_callback(event)
            self.set_annotation(resp)

    def set_annotation(self, data: InteractiveAnnotationData):
        """ Set annotation according to annotation data. If data is not of supplied type, annotation is removed from Canvas

        @param data:
        @return:
        """
        if isinstance(data, InteractiveAnnotationData):
            self._annotation.xy = (data.x, data.y)
            self._annotation.set_text(data.text)
            self._annotation.set_visible(True)
        else:
            self._annotation.set_visible(False)
        self._blit_manager.update()


def distribution_patch(ax, x, mu, var, color=[1, 0, 0], num_std=2, alpha=0.5, linewidth=1, label=''):
    '''
    Function plots the mean and corresponding variance onto the specified axis

    ax : axis object where the distribution patches should be plotted
    X  : nbpoints array of x-axis values
    Mu : nbpoints array of mean values corresponding to the x-axis values
    Var: nbpoints array of variance values corresponding to the x-axis values

    Author: Martijn Zeestrate, 2015
    '''

    # get number of points:
    npoints = len(x)

    # Create vertices:
    xmsh = np.append(x, x[::-1])
    vTmp = np.sqrt(var) * num_std
    ymsh = np.append(vTmp + mu, mu[::-1] - vTmp[::-1])
    msh = np.concatenate((xmsh.reshape((2 * npoints, 1)), ymsh.reshape((2 * npoints, 1))), axis=1)
    msh = np.concatenate((msh, msh[-1, :].reshape((1, 2))), axis=0)

    # Create codes
    codes = [Path.MOVETO]
    codes.extend([Path.LINETO] * (2 * npoints - 1))
    codes.extend([Path.CLOSEPOLY])

    # Create Path
    path = Path(msh, codes)
    patch = patches.PathPatch(path, facecolor=color, lw=0, edgecolor=color, alpha=alpha)

    # Add to axis:
    ax.add_patch(patch)  # Patch
    ax.plot(x, mu, linewidth=linewidth, color=color, label=label)  # Mean


def computeCorrelationMatrix(sigma):
    var = np.sqrt(np.diag(sigma))
    return sigma/var[None, :].T.dot(var[None,:])


def plotCorrelationMatrix(sigma, labels=None, ax=None, labelsize=20):
    cormatrix = computeCorrelationMatrix(sigma)
    n_var = sigma.shape[0]

    if ax == None:
        fig = Figure(figsize=(4, 3))
        ax = fig.gca()

    if labels is None:
        labels = range(1, n_var+1)
    h = ax.pcolor(cormatrix, cmap='RdBu', vmax=1, vmin=-1)
    ax.invert_yaxis()
    ax.xaxis.set_label_position('top')
    ax.set_xticks(np.arange(0, n_var) + 0.5)
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(0, n_var)+0.5)
    ax.set_yticklabels(labels)
    ax.tick_params(labelsize=labelsize)
    l = ax.figure.Colorbar(ax, h, ticks=[-1, 0, 1])
    #l = c.colorbar(h, ticks=[-1, 0, 1])
    l.ax.set_yticklabels([r'$-1$', r'$0$', r'$1$'])
    l.ax.tick_params(labelsize=labelsize)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(4, 3))
    ax = fig.gca()

    mu = np.ones(2)
    sigma = np.eye(2)
    g = GaussianPatch2d(ax=ax, mu=mu, sigma=sigma)
    ax.set_ylim([-3, 3])
    ax.set_xlim([-3, 3])

    fig.canvas.draw()
    plt.show()

    input('Press any [Enter] to quit.')
