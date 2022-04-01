"""
Plot EIS or DRT data for academic articles
Methods:
    plot_eis(freq, z_real, z_imag, title='', legends=(), format_ids=(()), hdl=None):
    returns hdl (handle)
        freq, z_real and z_imag can be one or more EIS data sets, stored in 1d or 2d arrays
        title: optional, the title of the plot
        legends: optional, labels of each data set
        format_id is optional. For one single data set, please use double brackets: ((style_id, color_id, shape_id), )
        format_id = ((style_id, color_id, shape_id), (style_id, color_id, shape_id), ...)
            style_id = (0: solid markers 1: empty markers 2: line)
            color_id see color map on top of this file
            shape_id see marker map and line map on top of this file
        highlight: if False, all data points are plotted in the same way
            if True, data points at decade frequencies (i.e. 0.1 Hz, 1 Hz, 10 Hz, 100 kHz etc.) are highlighted
            if an array of indices, the specified data points are highlighted
        hdl: handles, for iterative plotting. hdl contains hdl.fig, hdl.axs and other fields (optional)

    plot_drt(gamma, tau, L=None, R_inf_ref=None, show_LR=False, show_tau=True, title='', labels=(), format_ids=(()),
             LR_pos_x=0.02, LR_pos_y=0.75, hdl=None):
    returns hdl (handle)
        gamma and tau can be one or more DRT data sets, stored in 1d or 2d arrays
        L and R_inf_ref are inductance(s) and series resistance(s) given by the DRT fit(s), can be floats or arrays
            note that L may contain self induction (affects z_imag) and mutual/wire induction (affects z_real)
                or self induction only
        show_LR: boolean, whether or not to show L and R_inf_ref on the plot
        show_tau: if True, plot gamma to tau; if False, plot gamma to frequency
        LR_pos_x and LR_pos_y: position of the frame showing L and R_inf_ref
        The usage of title, labels and format_ids is the same as in plot_eis

    plot_eisdrt(eis_freq, eis_z_real, eis_z_imag, drt_gamma, drt_tau, drt_L=None, drt_R_inf=None,
                eis_title='', eis_labels=(), eis_format_ids=(()), eis_highlight=False,
                drt_show_LR=False, drt_show_tau=True, drt_title='',
                drt_labels=(), drt_format_ids=(()), drt_LR_pos_x=0.02, drt_LR_pos_y=0.75, hdl=None):
    returns hdl (handle)
        Plot EIS and DRT in one single figure/window
        Input parameters are explained in plot_eis and plot_drt

    show_plot(hdl):
        show the plot if modified outside util_plot.

    Note exception handling or messaging is not complete
    Misuse might end up in confusing exceptions from other modules
Coded by Hangyue Li, Tsinghua University
"""

message_popup_switch = False


from matplotlib import use
use('TkAgg')
from matplotlib import pyplot as plt
plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
from matplotlib import rcParams, rcParamsDefault, axes, colors
import numpy as np
# import keyboard


def _cmap(index):  # color map
    colors = ('#000000', '#d00000', '#00d000', '#0000d0', '#00a0c0', '#a000c0', '#c0a000',  # black, RGB, CMY
              '#707070', '#d07070', '#70d070', '#7070d0', '#70d0e0', '#d070e0', '#e0d070',  # brighter version of above
              '#c0c0c0')  # light gray
    return colors[index % len(colors)]
# cmap = plt.get_cmap("tab10")  # default colors


def _mmap(index):  # marker map
    markers = ('o', 'v', '^', 's', '*', 'P', 'D', 'x')
    # circles, downward triangles, upward triangles, squares, stars, plus signs, diamonds
    # https://matplotlib.org/3.2.0/api/markers_api.html
    return markers[index % len(markers)]


def _lmap(index):  # line map
    lines = ('-', '--', '-.', ':')
    # solid, dashed, dash-dotted, dotted
    return lines[index % len(lines)]


def _format_map(format_id):
    # returns 'marker', 'color', 'markerfacecolor', 'linestyle'
    style_id, color_id, shape_id = format_id
    # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html
    if style_id == 0:  # solid markers
        return _mmap(shape_id), _cmap(color_id), _cmap(color_id), ''
    elif style_id == 1:  # empty markers
        return _mmap(shape_id), _cmap(color_id), 'None', ''
    elif style_id == 2:  # line
        return '', _cmap(color_id), _cmap(color_id), _lmap(shape_id)
    else:
        raise ValueError('Unsupported style ID!')


def _subplot(ax, x=(0, 1), y=(0, 1), xlabel='', ylabel='', title='', label='', highlight=None,
             marker='.', color='black', markerfacecolor='black', linestyle='solid',
             linewidth=2, markersize=4, highlighted_markersize=8):
    if highlight is not None:
        # https://matplotlib.org/3.2.0/gallery/lines_bars_and_markers/markevery_demo.html
        ax.plot(x, y, marker=marker, color=color, linestyle=linestyle, linewidth=linewidth, markevery=highlight,
                markerfacecolor=markerfacecolor, markersize=highlighted_markersize)
    if contains_Chinese(label) or contains_Japanese(label):
        rcParams['font.family'] = "STXiHei"
    ax.plot(x, y, marker=marker, color=color, linestyle=linestyle, linewidth=linewidth,
            markerfacecolor=markerfacecolor, markersize=markersize, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        if contains_Chinese(title) or contains_Japanese(title):
            ax.set_title(title, fontname="STXiHei")
        else:
            ax.set_title(title)
    ax.grid(True)


def _subplot_with_weights(ax, x=(0, 1), y=(0, 1), xlabel='', ylabel='', title='', label='', highlight=None,
                          marker='.', color='black', markerfacecolor='black', linestyle='solid',
                          linewidth=2, markersize=4, highlighted_markersize=8, weights=None):
    if weights is None:
        weights = np.ones(len(x))
    else:
        weights = np.array(weights)
        weights = weights / np.max(weights)
        weights = 0.2 + 0.8 * weights
    colorlist = [colors.to_rgba(color, weight) for weight in weights]
    if highlight is not None:
        # https://matplotlib.org/3.2.0/gallery/lines_bars_and_markers/markevery_demo.html
        ax.scatter(x[highlight], y[highlight], marker=marker, c=markerfacecolor,
                   edgecolors=colorlist, s=highlighted_markersize ** 2)
    if contains_Chinese(label) or contains_Japanese(label):
        rcParams['font.family'] = "STXiHei"
    ax.scatter(x, y, marker=marker, c=markerfacecolor,
               edgecolors=colorlist, s=markersize ** 2, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if contains_Chinese(title) or contains_Japanese(title):
        ax.set_title(title, fontname="STXiHei")
    else:
        ax.set_title(title)
    ax.grid(True)


def _mark_decade_points(freq, tolerance=0.01):
    lgfreq = np.log10(freq)
    decade_point_indices = []
    for i in range(len(freq)):
        if np.abs(lgfreq[i] - np.floor(lgfreq[i] + 0.5)) < tolerance:
            decade_point_indices.append(i)
    return decade_point_indices


def _plot_eis(axs, freq, z_real, z_imag, title='', labels=None, format_ids=None, highlight=False, weights=None,
              thin_line=False):
    # \/ is the whitespace
    # freq = np.array(freq)  # fails if freq contains arrays of different lengths
    # z_real = np.array(z_real)
    # z_imag = np.array(z_imag)
    # if not len(freq.shape) == len(z_real.shape) == len(z_imag.shape):
    #     raise TypeError('Input data shape mismatch!')

    if thin_line:
        linewidth = 1
    else:
        linewidth = 2

    def _plot_eis_line(f, z_r, z_i, label='', format_id=(), index=0):
        hl_ind = None
        if hasattr(highlight, '__len__'):  # if decade_highlight has len, then it is expected to be an array
            hl_ind = highlight
        elif highlight:
            hl_ind = _mark_decade_points(f)  # decade point indices
        if len(format_id) == 3:
            marker, color, markerfacecolor, linestyle = _format_map(format_id)
        else:
            marker, color, markerfacecolor, linestyle = _format_map((2, index, 0))
        if len(format_id) == 0:
            format_id = (2, 0, 0)
        if format_id[0] >= 2:  # is line, not points
            _subplot(axs[0], z_r, -z_i, title=title, color=color, marker=marker,
                     markerfacecolor=markerfacecolor, linewidth=linewidth,
                     linestyle=linestyle, label=label, highlight=hl_ind,
                     xlabel=r'$Z_{\mathrm{real}} \/ / \/ \mathrm{\Omega \bullet cm^2}$',
                     ylabel=r'$-Z_{\mathrm{imag}} \/ / \/ \mathrm{\Omega \bullet cm^2}$')
        else:
            _subplot_with_weights(axs[0], z_r, -z_i, title=title, color=color, marker=marker,
                                  markerfacecolor=markerfacecolor, weights=weights, linewidth=linewidth,
                                  linestyle=linestyle, label=label, highlight=hl_ind,
                                  xlabel=r'$Z_{\mathrm{real}} \/ / \/ \mathrm{\Omega \bullet cm^2}$',
                                  ylabel=r'$-Z_{\mathrm{imag}} \/ / \/ \mathrm{\Omega \bullet cm^2}$')
        axs[0].axis('equal')
        # axs[0].set(xlim=(-0.2, 2.6), ylim=(-0.2, 0.8))

        if len(format_id) == 0:
            format_id = (2, 0, 0)
        if format_id[0] >= 2:  # is line, not points
            _subplot(axs[1], f, -z_i, title='', color=color, marker=marker,
                     markerfacecolor=markerfacecolor,
                     linestyle=linestyle, label=label, linewidth=linewidth,
                     xlabel='Frequency / Hz',
                     ylabel=r'$-Z_{\mathrm{imag}} \/ / \/ \mathrm{\Omega \bullet cm^2}$')
        else:
            _subplot_with_weights(axs[1], f, -z_i, title='', color=color, marker=marker,
                                  markerfacecolor=markerfacecolor, linewidth=linewidth,
                                  linestyle=linestyle, label=label, weights=weights,
                                  xlabel='Frequency / Hz',
                                  ylabel=r'$-Z_{\mathrm{imag}} \/ / \/ \mathrm{\Omega \bullet cm^2}$')
        axs[1].set_xscale('log')
        if len(label) > 0:
            # https://stackoverflow.com/questions/19125722/adding-a-legend-to-pyplot-in-matplotlib-in-the-simplest-manner-possible
            # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html
            axs[0].legend()
            axs[1].legend()

            # make labels opaque
            # https://stackoverflow.com/questions/12848808/set-legend-symbol-opacity-with-matplotlib/12850923
            leg = axs[0].legend()
            for lh in leg.legendHandles:
                lh.set_alpha(1)
            leg = axs[1].legend()
            for lh in leg.legendHandles:
                lh.set_alpha(1)

            # axs[1].legend(loc='upper left')
        # axs[1].set(ylim=(-0.1, 0.6))

    if (not hasattr(freq[0], '__len__')) or len(freq) == 1:  # if freq only contains 1 data set
        if format_ids is None or len(format_ids) == 0:
            _plot_eis_line(freq, z_real, z_imag)
        else:
            _plot_eis_line(freq, z_real, z_imag, format_id=format_ids[0])
    elif len(freq) > 1:
        n = len(freq)  # number of data sets
        if format_ids is None or len(format_ids) == 0:
            format_ids = tuple([[1, 0, 0]] * n)
        if labels is None:
            labels = tuple([''] * n)
        if len(labels) == n:
            for i in range(len(freq)):
                _plot_eis_line(freq[i], z_real[i], z_imag[i], index=i, label=labels[i], format_id=format_ids[i])
        else:
            for i in range(len(freq)):
                _plot_eis_line(freq[i], z_real[i], z_imag[i], index=i, format_id=format_ids[i])
    # plt.show()


def _plot_drt(ax, gamma, tau, L=None, R_inf=None, R_0=None, show_LR=False, show_tau=True, title='', labels=None,
              drt_highlights=None, format_ids=None, thin_line=False):
    # gamma = np.array(gamma)  # fails if gamma contains arrays of different lengths
    # tau = np.array(tau)

    if thin_line:
        linewidth = 1
    else:
        linewidth = 2

    if R_inf is not None:
        R_inf = np.array(R_inf)
    if L is None or R_inf is None:
        if show_LR:
            if message_popup_switch:
                from tkinter import messagebox
                messagebox.showwarning(title=None, message=None)
            else:
                print('Cannot show L and/or R_inf_ref because R_inf_ref must be supplied if show_LR.')
        show_LR = False
    if L is None:
        L = [None] * len(gamma)  # make L subscriptable
    elif L[0] is not None:
        L = np.array(L)
    # if not len(gamma.shape) == len(tau.shape):
    #     raise TypeError('Input data shape mismatch!')

    # \/ is the whitespace

    def _plot_drt_line(g, t, L=None, R_inf=None, R_0=None, label='', format_id=(), index=0):
        if len(format_id) == 3:
            marker, color, markerfacecolor, linestyle = _format_map(format_id)
        else:
            marker, color, markerfacecolor, linestyle = _format_map((2, index, 0))
        g = np.array(g)
        t = np.array(t)
        if show_tau:
            _subplot(ax, t, g, title=title, marker=marker, color=color, label=label,
                     markerfacecolor=markerfacecolor, linestyle=linestyle, linewidth=linewidth,
                     xlabel=r'$\tau$' + ' / s',
                     ylabel=r'$\gamma$')
            if drt_highlights is not None:
                ax.scatter(t[drt_highlights], g[drt_highlights], marker='+', c=markerfacecolor)
        else:
            _subplot(ax, 1 / (2 * np.pi * t), g, title=title, marker=marker, color=color, label=label,
                     markerfacecolor=markerfacecolor, linestyle=linestyle, linewidth=linewidth,
                     xlabel=r'Frequency / Hz',
                     ylabel=r'$\gamma$')
            if drt_highlights is not None:
                ax.scatter(1 / (2 * np.pi * t[drt_highlights]), g[drt_highlights], marker='+', c=markerfacecolor)
        ax.set_xscale('log')
        if len(label) > 0:
            # https://stackoverflow.com/questions/19125722/adding-a-legend-to-pyplot-in-matplotlib-in-the-simplest-manner-possible
            # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html
            ax.legend()
            # ax.legend(loc='upper left')
        if show_LR:
            textstr = ''
            if L is not None and hasattr(L, '__len__') and L[0] is not None:
                L = L * 1e6
                if hasattr(L, '__len__'):
                    # L contains L_self_induction and L_mutual_induction (wire induction, affects z_real)
                    L_str = '[{:s}]'.format(', '.join(['{:.2e}'.format(x) for x in L]))[1:-1]
                else:  # L itself is L_self_induction
                    L_str = '{:.2e}'.format(L)
                textstr += '\n' + r'$L$' + ' = ' + L_str + r'$\/\mathrm{\mu H}$'
            textstr += '\n' + r'$R_{\mathrm{inf}}$' + ' = ' + '{:.2e}'.format(R_inf) + \
                       r'$\/\mathrm{\Omega \bullet cm^2}$'
            if R_0 is not None:
                textstr += ', ' + r'$R_{\mathrm{0}}$' + ' = ' + '{:.2e}'.format(R_0) + \
                           r'$\/\mathrm{\Omega \bullet cm^2}$'
            return textstr
        else:
            return ''
        # ax.set(ylim=(-0.05, 1.05))

    textstr = ''
    if (not hasattr(gamma[0], '__len__')) or len(gamma) == 1:  # if gamma only contains 1 data set
        if format_ids is None or len(format_ids) == 0:
            textstr = _plot_drt_line(gamma, tau, L, R_inf, R_0)
        else:
            textstr = _plot_drt_line(gamma, tau, L, R_inf, R_0, format_id=format_ids[0])
    elif len(gamma) > 1:
        n = len(gamma)  # number of data sets
        if L is None:
            L = [None] * n
        if R_inf is None:
            R_inf = [None] * n
        if R_0 is None:
            R_0 = [None] * n
        if format_ids is None or len(format_ids) == 0:
            format_ids = tuple([[1, 0, 0]] * n)
        if labels is None:
            labels = tuple([''] * n)
        if len(labels) == n:
            for i in range(len(gamma)):
                textstr += _plot_drt_line(gamma[i], tau[i], L[i], R_inf[i], R_0[i],
                                          label=labels[i], format_id=format_ids[i], index=i)
        else:
            for i in range(len(gamma)):
                textstr += _plot_drt_line(gamma[i], tau[i], L[i], R_inf[i], R_0[i],
                                          format_id=format_ids[i], index=i)

    if show_LR:
        if show_tau:
            LR_pos_x = 0.03
            LR_pos_y = 0.9
        else:  # show frequency
            LR_pos_x = 0.5
            LR_pos_y = 0.9
        # https://matplotlib.org/3.2.0/gallery/recipes/placing_text_boxes.html
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax.text(LR_pos_x, LR_pos_y, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=props)
    # plt.show()


def contains_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False


def contains_Japanese(word):
    for ch in word:
        if '\u3040' <= ch <= '\u30ff' or '\uff66' <= ch <= '\ff9f':
            return True
    return False


def plot_eis(freq, z_real, z_imag, title='', labels=None, format_ids=None, highlight=False, hdl=None,
             weights=None, clear_old=True, thin_line=False):
    if hdl is None:
        hdl = Handle()
    if not isinstance(hdl.fig, plt.Figure):
        fig, axs = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    elif hdl.axs is None or not isinstance(hdl.axs[0], axes.Axes):
        fig = hdl.fig
        fig._constrained = True
        gs = fig.add_gridspec(2, 1)
        axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]
    else:
        fig, axs = hdl.fig, hdl.axs
        if clear_old:
            axs[0].clear()
            axs[1].clear()
    _plot_eis(axs, freq, z_real, z_imag, title=title, labels=labels, format_ids=format_ids, highlight=highlight,
              weights=weights, thin_line=thin_line)
    hdl.fig = fig
    hdl.axs = axs
    # show_plot(hdl)
    return hdl


def plot_drt(gamma, tau, L=None, R_inf=None, show_LR=False, show_tau=True, title='', labels=None, format_ids=None,
             hdl=None, R_0=None, clear_old=True, thin_line=False):
    if hdl is None:
        hdl = Handle()
    if not isinstance(hdl.fig, plt.Figure):
        fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    elif hdl.axs is None or not isinstance(hdl.axs, axes.Axes):
        fig = hdl.fig
        fig._constrained = True
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0, 0])
    else:
        fig, ax = hdl.fig, hdl.axs
        if clear_old:
            ax.clear()
    _plot_drt(ax, gamma, tau, L=L, R_inf=R_inf, show_LR=show_LR, show_tau=show_tau, title=title, labels=labels,
              format_ids=format_ids, R_0=R_0, thin_line=thin_line)
    hdl.fig = fig
    hdl.axs = ax
    # show_plot(hdl)
    return hdl


def plot_eisdrt(eis_freq, eis_z_real, eis_z_imag, drt_gamma, drt_tau,
                eis_z_indiv=None, drt_gamma_indiv=None,
                drt_L=None, drt_R_inf=None, drt_R_0=None,
                eis_title='', eis_labels=None, eis_format_ids=None, eis_highlight=False, eis_weights=None,
                drt_show_LR=False, drt_show_tau=True, drt_title='',
                drt_labels=None, drt_format_ids=None,
                hdl=None, clear_old=True):
    if hdl is None:
        hdl = Handle()
    if not isinstance(hdl.fig, plt.Figure):
        # https://matplotlib.org/3.1.1/users/dflt_style_changes.html#figure-size-font-size-and-screen-dpi
        rcParams['figure.figsize'] = [10.0, 4.0]  # set temporary default figure size
        fig = plt.figure(constrained_layout=True)
        rcParams['figure.figsize'] = rcParamsDefault['figure.figsize']  # restore to default
        gs = fig.add_gridspec(2, 2)
        axs_eis = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]  # top-left, bottom-left
        ax_drt = fig.add_subplot(gs[:, 1])  # top-through-bottom-right
    elif hdl.axs is None or not isinstance(hdl.axs[0][0], axes.Axes):
        fig = hdl.fig
        fig._constrained = True
        gs = fig.add_gridspec(2, 2)
        axs_eis = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]  # top-left, bottom-left
        ax_drt = fig.add_subplot(gs[:, 1])  # top-through-bottom-right
    else:
        fig, axs = hdl.fig, hdl.axs
        axs_eis, ax_drt = axs
        if clear_old:
            axs_eis[0].clear()
            axs_eis[1].clear()
            ax_drt.clear()
    if eis_z_indiv is not None:
        n = len(eis_z_indiv)
        if hasattr(eis_freq[0], '__len__'):
            freq_list = [eis_freq[0]] * n
        else:
            freq_list = [eis_freq] * n
        eis_indiv_format_ids = tuple([(eis_format_ids[-1][0], 14, 6), ] * n)
        _plot_eis(axs_eis, freq_list, eis_z_indiv.real, eis_z_indiv.imag, format_ids=eis_indiv_format_ids,
                  thin_line=True)
    _plot_eis(axs_eis, eis_freq, eis_z_real, eis_z_imag, title=eis_title, labels=eis_labels,
              format_ids=eis_format_ids, highlight=eis_highlight, weights=eis_weights)
    if drt_gamma_indiv is not None:
        n = len(eis_z_indiv)
        if hasattr(drt_tau[0], '__len__'):
            tau_list = [drt_tau[0]] * n
        else:
            tau_list = [drt_tau] * n
        drt_indiv_format_ids = tuple([(drt_format_ids[-1][0], 14, 2), ] * n)
        _plot_drt(ax_drt, drt_gamma_indiv, tau_list,
                  show_LR=False, show_tau=drt_show_tau, format_ids=drt_indiv_format_ids, thin_line=True)
    _plot_drt(ax_drt, drt_gamma, drt_tau, L=drt_L, R_inf=drt_R_inf, R_0=drt_R_0,
              show_LR=drt_show_LR, show_tau=drt_show_tau, title=drt_title, labels=drt_labels,
              format_ids=drt_format_ids)
    axs = [axs_eis, ax_drt]
    hdl.fig = fig
    hdl.axs = axs
    # show_plot(hdl)
    return hdl


def _apply_lim(ax, lim, x_or_y='x'):
    if np.nan in lim:
        return
    else:
        if 'x' == x_or_y.lower():
            ax.set_xlim(lim)
        elif 'y' == x_or_y.lower():
            ax.set_ylim(lim)
        else:
            raise ValueError("Input argument \'x_or_y\' must be either \'x\' or \'y\'.")


def plot_eisdrtres(eis_freq, eis_z_real, eis_z_imag, drt_gamma, drt_tau, eis_res_real, eis_res_imag,
                   eis_z_indiv=None, drt_gamma_indiv=None,
                   drt_L=None, drt_R_inf=None, drt_R_0=None,
                   eis_title='', eis_labels=None, eis_format_ids=None, eis_highlight=False, eis_weights=None,
                   drt_show_LR=False, drt_show_tau=True, drt_title='', drt_highlights=None,
                   drt_labels=None, drt_format_ids=None, res_format_ids=None,
                   hdl=None, clear_old=True):
    # This is the method to plot in EISART
    if hdl is None:
        hdl = Handle()
    if not isinstance(hdl.fig, plt.Figure):
        # https://matplotlib.org/3.1.1/users/dflt_style_changes.html#figure-size-font-size-and-screen-dpi
        rcParams['figure.figsize'] = [10.0, 6.0]  # set temporary default figure size
        fig = plt.figure(constrained_layout=True)
        rcParams['figure.figsize'] = rcParamsDefault['figure.figsize']  # restore to default
        gs = fig.add_gridspec(2, 2)
        axs_eis = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]  # top-left, bottom-left
        ax_drt = fig.add_subplot(gs[0, 1])  # top-right
        ax_res = fig.add_subplot(gs[1, 1])  # bottom-right
    elif hdl.axs is None or (not isinstance(hdl.axs[0][0], axes.Axes)):
        fig = hdl.fig
        fig._constrained = True
        gs = fig.add_gridspec(2, 2)
        axs_eis = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]  # top-left, bottom-left
        ax_drt = fig.add_subplot(gs[0, 1])  # top-right
        ax_res = fig.add_subplot(gs[1, 1])  # bottom-right
    else:
        fig, axs = hdl.fig, hdl.axs
        axs_eis, ax_2 = axs
        ax_drt, ax_res = ax_2
        if clear_old:
            axs_eis[0].clear()
            axs_eis[1].clear()
            ax_drt.clear()
            ax_res.clear()
    if eis_z_indiv is not None:
        n = len(eis_z_indiv)
        if hasattr(eis_freq[0], '__len__'):
            freq_list = [eis_freq[0]] * n
        else:
            freq_list = [eis_freq] * n
        eis_indiv_format_ids = tuple([(2, 14, 6), ] * n)
        _plot_eis(axs_eis, freq_list, eis_z_indiv.real, eis_z_indiv.imag, format_ids=eis_indiv_format_ids,
                  thin_line=True)
    _plot_eis(axs_eis, eis_freq, eis_z_real, eis_z_imag, title=eis_title, labels=eis_labels,
              format_ids=eis_format_ids, highlight=eis_highlight, weights=eis_weights, thin_line=False)
    if drt_gamma_indiv is not None:
        n = len(eis_z_indiv)
        if hasattr(drt_tau[0], '__len__'):
            tau_list = [drt_tau[0]] * n
        else:
            tau_list = [drt_tau] * n
        drt_indiv_format_ids = tuple([(drt_format_ids[-1][0], 14, 2), ] * n)
        _plot_drt(ax_drt, drt_gamma_indiv, tau_list,
                  show_LR=False, show_tau=drt_show_tau, format_ids=drt_indiv_format_ids, thin_line=True)
    _plot_drt(ax_drt, drt_gamma, drt_tau, L=drt_L, R_inf=drt_R_inf, R_0=drt_R_0,
              show_LR=drt_show_LR, show_tau=drt_show_tau, title=drt_title, labels=drt_labels,
              drt_highlights=drt_highlights, format_ids=drt_format_ids, thin_line=False)
    # assume that the first 2 eis data sets are measured data and fitted data
    # and their eis_freq are identical
    ax_res.set_xlim(ax_drt.get_xlim())
    res_x = np.array(eis_freq[:2])
    res_xlabel = ax_drt.get_xlabel()
    if drt_show_tau:
        res_x = 1 / (2 * np.pi * res_x)
    if res_format_ids is None:
        res_format_ids = ((1, 4, 1), (1, 10, 3))
    hdl_res = Handle()
    hdl_res.fig = fig
    hdl_res.axs = ax_res
    plot_xy(res_x, [100 * eis_res_real, 100 * eis_res_imag], xlog=True, xlabel=res_xlabel,
            ylabel='%',
            # ylabel=r'$\/\mathrm{\Omega \bullet cm^2}$',
            title='Fitting Residual from Input', labels=['Real', 'Imag'],
            format_ids=res_format_ids,
            hdl=hdl_res)
    ax_res = hdl_res.axs

    # apply plot limits
    if hdl.plot_ranges.enable_ranges:
        aspect_ratio = get_axes_ratio(axs_eis[0]) * 0.75  # 4.8 / 6.4
        yrange = (hdl.plot_ranges.z_re_max - hdl.plot_ranges.z_re_min) * aspect_ratio
        hdl.plot_ranges.z_im_min = hdl.plot_ranges.z_im_max - yrange
        axs_eis[0].axis('square')
        _apply_lim(axs_eis[0], (hdl.plot_ranges.z_re_min, hdl.plot_ranges.z_re_max), 'x')
        _apply_lim(axs_eis[0], (-hdl.plot_ranges.z_im_max, -hdl.plot_ranges.z_im_min), 'y')
        _apply_lim(axs_eis[1], (hdl.plot_ranges.eis_f_min, hdl.plot_ranges.eis_f_max), 'x')
        _apply_lim(axs_eis[1], (-hdl.plot_ranges.z_im_max, -hdl.plot_ranges.z_im_min), 'y')
        _apply_lim(ax_drt, (hdl.plot_ranges.drt_g_min, hdl.plot_ranges.drt_g_max), 'y')
        _apply_lim(ax_res, (hdl.plot_ranges.res_min, hdl.plot_ranges.res_max), 'y')
        if drt_show_tau:
            _apply_lim(ax_drt, (hdl.plot_ranges.drt_t_min, hdl.plot_ranges.drt_t_max), 'x')
            _apply_lim(ax_res, (hdl.plot_ranges.drt_t_min, hdl.plot_ranges.drt_t_max), 'x')
        else:
            drt_f_min = 1 / (2 * np.pi * hdl.plot_ranges.drt_t_max)
            drt_f_max = 1 / (2 * np.pi * hdl.plot_ranges.drt_t_min)
            _apply_lim(ax_drt, (drt_f_min, drt_f_max), 'x')
            _apply_lim(ax_res, (drt_f_min, drt_f_max), 'x')

    axs = [axs_eis, [ax_drt, ax_res]]
    hdl.fig = fig
    hdl.axs = axs
    # show_plot(hdl)
    return hdl


def save_plot(hdl, fullfilename, img_fmt=None):
    fig = hdl.fig
    if img_fmt is None:
        fig.savefig(fullfilename)
    else:
        fig.savefig(fullfilename, format=img_fmt)


def show_plot(hdl):
    key_pressed = ['', ]

    def get_key(event):
        key_pressed[0] = event.key

    fig = hdl.fig
    if plt.fignum_exists(fig.number):
        fig.canvas.draw()
        fig.canvas.flush_events()
        fig.canvas.mpl_connect('key_press_event', get_key)
    else:
        # Plot not shown
        hdl.hold = False
        pass
    keypress = False

    def waitkey():
        if plt.fignum_exists(fig.number):
            try:
                plt.waitforbuttonpress()
            except Exception:  # expected
                # I would assume that there's an exception because the figure is closed
                pass
        else:  # if there is no figure, do not wait at all for button press
            hdl.hold = False
            return True
        # if keyboard.is_pressed('F5'):
        #     hdl.hold = True
        #     return True
        # if keyboard.is_pressed('enter'):
        #     hdl.hold = False
        #     return True
        # if keyboard.is_pressed('escape'):
        #     hdl.hold = False
        #     plt.close()
        #     return True
        if key_pressed[0] == 'f5':
            hdl.hold = True
            return True
        if key_pressed[0] == 'enter':
            hdl.hold = False
            return True
        if key_pressed[0] == 'escape':
            hdl.hold = False
            plt.close()
            return True

    while not keypress:
        keypress = waitkey()

    return hdl


def close_plot():
    plt.close()


class PlotRanges:
    def __init__(self, filename=None):
        # limits are applied in (min, max) pairs.
        # If np.nan is in a pair, this pair of limits will be ignored and automatically set
        self.z_re_min = 0.0  # Ohm * cm^2
        self.z_re_max = 3.0
        self.z_im_min = -1.0
        self.z_im_max = 0.2
        self.eis_f_min = 1e-2  # Hz
        self.eis_f_max = 1e+6
        self.drt_g_min = 0.0  # Ohm * cm^2 / log(s)
        self.drt_g_max = 1.0
        self.drt_t_min = 1e-8  # s
        self.drt_t_max = 1e+2
        self.res_min = -5.0  # percentage, +5%
        self.res_max = 5.0  # percentage, -5%
        self.enable_ranges = False

        if filename is not None:
            self.load(filename)

    def load(self, filename):
        from os import path
        if not path.exists(filename):
            # raise ValueError("Plot settings file \'" + filename + '\' does not exist.')
            self.save(filename)
        else:
            with open(filename, 'r') as f:
                s = f.read()
                s = s.replace(',\n\'', ', \'')
                self.from_str(s)

    def save(self, filename):
        with open(filename, 'w+') as f:
            s = str(self)
            s = s.replace(', \'', ',\n\'')
            f.write(s)  # TODO may overwrite without warning

    def from_str(self, s):
        dict_ = eval(s)  # evaluate the string. The result is expected to be a dictionary
        for key, val in dict_.items():
            setattr(self, key, val)
        return self

    def __str__(self):
        return str(self.__dict__)  # convert this instance to a dictionary, then to a string


class Handle:
    def __init__(self):
        self.fig = None
        self.axs = None
        self.hold = False
        self.data = None
        self.plot_ranges = PlotRanges()


# https://github.com/matplotlib/matplotlib/issues/8013
def get_axes_ratio(ax=None):
    """Calculate the aspect ratio of an axes boundary/frame"""
    if ax is None:
        ax = plt.gca()
    fig = ax.figure
    ll, ur = ax.get_position() * fig.get_size_inches()
    width, height = ur - ll
    return height / width

def _plot_xy(ax, x, y, xlog=False, ylog=False, xlabel='', ylabel='', title='', labels=None, format_ids=None):
    x = np.array(x)
    y = np.array(y)
    if len(x.shape) == 1:
        if format_ids is None:
            format_ids = ((1, 0, 0),)
        if labels is None:
            labels = ('', )
        if len(format_ids[0]) == 3:
            marker, color, markerfacecolor, linestyle = _format_map(format_ids[0])
        else:
            marker, color, markerfacecolor, linestyle = _format_map((2, 0, 0))
        _subplot(ax, x, y, xlabel=xlabel, ylabel=ylabel, title=title, label=labels[0],
                 marker=marker, color=color, markerfacecolor=markerfacecolor, linestyle=linestyle)
        if len(labels[0]) > 0:
            # https://stackoverflow.com/questions/19125722/adding-a-legend-to-pyplot-in-matplotlib-in-the-simplest-manner-possible
            # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html
            ax.legend()
            # ax.legend(loc='lower right')
    elif len(x.shape) == 2:
        n = x.shape[0]
        if format_ids is None:
            format_ids = tuple([[1, 0, 0]] * n)
        if labels is None:
            labels = tuple([''] * n)
        for i in range(x.shape[0]):
            if len(format_ids[i]) == 3:
                marker, color, markerfacecolor, linestyle = _format_map(format_ids[i])
            else:
                marker, color, markerfacecolor, linestyle = _format_map((2, i, 0))
            _subplot(ax, x[i], y[i], xlabel=xlabel, ylabel=ylabel, title=title, label=labels[i],
                     marker=marker, color=color, markerfacecolor=markerfacecolor, linestyle=linestyle)
            if len(labels[i]) > 0:
                # https://stackoverflow.com/questions/19125722/adding-a-legend-to-pyplot-in-matplotlib-in-the-simplest-manner-possible
                # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html
                ax.legend()
                # ax.legend(loc='lower right')
    else:
        raise ValueError("x must be a 1D or 2D array, while x now has ", len(x.shape), " dimensions.")
    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')

def plot_xy(x, y, xlog=False, ylog=False, xlabel='', ylabel='', title='', labels=None, format_ids=None, hdl=None):
    if hdl is None:
        hdl = Handle()
    if not isinstance(hdl.fig, plt.Figure):
        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0, 0])
    elif hdl.axs is None or (not isinstance(hdl.axs, axes.Axes)):
        fig = hdl.fig
        fig._constrained = True
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0, 0])
    else:
        fig, ax = hdl.fig, hdl.axs
    _plot_xy(ax, x, y, xlog, ylog, xlabel, ylabel, title, labels, format_ids)
    hdl.fig, hdl.axs = fig, ax
    return hdl


def plot_manual_peaks(tau=None, gamma=None, m_ecm_pars=None, show_tau=True, format_ids=None, hdl=None):
    clear_plot = False
    if tau is None or gamma is None:
        tau = (1e-6, 1e2)
        gamma = (0.0, 1.0)
        clear_plot = True  # This is only working on initialization
    tau = np.array(tau)
    gamma = np.array(gamma)
    rel_gamma = np.maximum(gamma, 0.0) / np.max(gamma)
    if show_tau:
        xlabel = r'$\tau$' + ' / s'
        x = tau
    else:
        xlabel = r'Frequency / Hz'
        x = 1 / (2 * np.pi * tau)
    ylabel = 'relative ' + r'$\gamma$'
    y2label = 'alpha'
    if hdl is None:
        hdl = Handle()
    fig = hdl.fig
    if not isinstance(fig, plt.Figure):
        fig = plt.figure(constrained_layout=False)

    try:  # if ax1 exists and was plotted with a different show_tau state, redraw
        ax1 = hdl.axs[0]
        xlabel_prev = ax1.get_xlabel()
        line = ax1.get_lines()[0]
        if xlabel_prev != xlabel or len(line.get_xdata()) < 3:
            hdl.axs = None
    except Exception:
        pass

    def val_to_rpos(t, a):
        if show_tau:
            pos_x = t
        else:  # show frequency
            pos_x = 1.0 / (2.0 * np.pi * t)
        pos_y = a ** 2
        return pos_x, pos_y

    def draw_m_ecm_bounds(ax, m_ecm_pars):
        [p.remove() for p in reversed(ax.patches)]
        if m_ecm_pars is None:
            return
        from matplotlib.patches import Rectangle
        opacity = '80'
        for i in range(m_ecm_pars.n):
            elem = m_ecm_pars.get(i + 1)
            rec_x0, rec_y0 = val_to_rpos(elem.taumin, elem.alphamin)
            rec_x1, rec_y1 = val_to_rpos(elem.taumax, elem.alphamax)
            rec_x = min(rec_x0, rec_x1)
            rec_y = min(rec_y0, rec_y1)
            rec_w = abs(rec_x0 - rec_x1)
            rec_h = abs(rec_y0 - rec_y1)
            if elem.fixed:
                rec_color = '#d95050' + opacity
            else:
                if elem.isGerischer:
                    rec_color = '#50b930' + opacity
                else:
                    rec_color = '#5070d9' + opacity
            rect = Rectangle((rec_x, rec_y), rec_w, rec_h, linewidth=2, edgecolor=rec_color, facecolor=rec_color)
            ax.add_patch(rect)
        return ax

    if hdl.axs is None or (not hasattr(hdl.axs, '__len__')) or (not isinstance(hdl.axs[0], axes.Axes)):
        fig.clear()
        fig._constrained = False
        gs = fig.add_gridspec(1, 1)
        ax1 = fig.add_subplot(gs[0, 0])
        if len(tau.shape) != 1:
            raise ValueError("tau must be a 1D array/vector, while tau now has ", len(tau.shape), " dimensions.")
        if format_ids is None:
            format_ids = ((2, 0, 0),)
        if len(format_ids[0]) == 3:
            marker, color, markerfacecolor, linestyle = _format_map(format_ids[0])
        else:
            marker, color, markerfacecolor, linestyle = _format_map((2, 0, 0))

        axes_x0 = 0.10
        axes_x1 = 0.90
        axes_y0 = 0.26
        axes_y1 = 0.95
        fig.subplots_adjust(bottom=axes_y0, top=axes_y1, left=axes_x0, right=axes_x1)
        ax1.tick_params(axis='x', labelsize=8, pad=1)
        ax1.tick_params(axis='y', labelsize=8, pad=1)
        # ax.spines['left'].set_position(('axes', 0.0))
        ax1.plot(x, rel_gamma, marker=marker, color=color, linestyle=linestyle, markerfacecolor=markerfacecolor)
        if clear_plot:
            line = ax1.get_lines()[0]
            line.set_xdata([])
            line.set_ydata([])
        ax1.set_xlabel(xlabel, fontsize=8, labelpad=-1)
        ax1.set_ylabel(ylabel, fontsize=8, labelpad=2)
        ax1.set_xscale('log')
        ax1.grid(axis='x')
        ax1.set_ylim(bottom=0.0, top=1.0)
        rel_gamma_ticks = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax1.yaxis.set_ticks(rel_gamma_ticks)

        right_side_color = '#104070'
        ax2 = ax1.twinx()
        ax2.set_ylim(bottom=0.0, top=1.0)
        ax2.tick_params(axis='y', labelsize=8, pad=1, colors=right_side_color)
        ax2.set_ylabel(y2label, fontsize=8, labelpad=2)
        alpha_ticks = np.array([0.0, 0.4, 0.6, 0.8, 0.9, 1.0])
        ax2.yaxis.set_ticks(alpha_ticks ** 2)
        ax2.yaxis.set_ticklabels(alpha_ticks)
        ax2.spines['right'].set_color(right_side_color)
        ax2.yaxis.label.set_color(right_side_color)
        ax2.grid(axis='y')
        ax2 = draw_m_ecm_bounds(ax2, m_ecm_pars)
        axs = (ax1, ax2)
    else:  # hdl.fig is Figure and hdl.axs are two Axes
        fig, axs = hdl.fig, hdl.axs
        ax1, ax2 = axs
        ax1.set_xlabel(xlabel, fontsize=8, labelpad=-1)
        line = ax1.get_lines()[0]
        line.set_xdata(x)
        line.set_ydata(rel_gamma)
        ax2 = draw_m_ecm_bounds(ax2, m_ecm_pars)
        axs = (ax1, ax2)
    hdl.fig, hdl.axs = fig, axs
    return hdl


default_q_units = {'T': 'K',
                   'y': '%',
                   'x': '%',
                   'j': 'A * m^-2',
                   'u': 'cm * s^-1',
                   'v': 'cm * s^-1',
                   'p': 'kPa',
                   'k': 'W * m^-1 * K^-1',
                   'rho': 'kg * m^-3',
                   'cp': 'J * kg^-1 * K^-1',
                   'rhocp': 'J * m^-3 * K^-1',
                   'R': 'mOhm * m'}

default_q_full_name = {'T': 'Temperature',
                       'y': 'Mass fraction',
                       'x': 'Molar fraction',
                       'j': 'Current density',
                       'u': 'X velocity',
                       'v': 'Y velocity',
                       'p': 'Gas pressure',
                       'k': 'Heat conductivity',
                       'rho': 'Density',
                       'cp': 'Heat capacity',
                       'rhocp': 'Volume-specific heat capacity',
                       'R': 'Resistivity'}


def plot_mesh_field(sim, quantity, unit, layer, species='None',
                    max_num_lvl=21, min_num_lvl=11, max_lvl=None, min_lvl=None):
    # https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/contourf_demo.html
    mesh, bc = sim.solver.mesh_new, sim.solver.bc

    x = np.linspace(0, mesh.lX * 100, mesh.nX)  # cm
    y = np.linspace(0, mesh.lY * 100, mesh.nY)  # cm
    X, Y = np.meshgrid(x, y)
    if quantity == 'rhocp':
        Z = mesh.get(unit, layer, 'rho') * mesh.get(unit, layer, 'cp')
    else:
        Z = mesh.get(unit, layer, quantity)

    quantity_title = default_q_full_name[quantity]
    if quantity == 'y' or quantity == 'x':
        if mesh.fluid_layers[layer] == 1:  # fuel layer
            species_id = bc.f_species.index(species)
        elif mesh.fluid_layers[layer] == 2:  # oxidant layer
            species_id = bc.o_species.index(species)
        else:
            species_id = 0
        Z = Z[:, :, species_id] * 100  # percentage
        quantity_title += ' of ' + species
    elif quantity == 'p':
        Z /= 1e3  # kPa
    elif quantity == 'u' or quantity == 'v':
        Z *= 100  # cm / s
    elif quantity == 'R':
        Z *= 1e3  # mOhm * m
    minlvl = np.min(Z)
    maxlvl = np.max(Z)
    if quantity == 'T':
        maxlvl = max(0, maxlvl)
    else:
        minlvl = min(0, minlvl)
        maxlvl = max(0, maxlvl)
    nlvl = int(min(21, maxlvl - minlvl))
    if nlvl < min_num_lvl:
        lvlspan = maxlvl - minlvl
        if lvlspan >= 1:
            format = '%.1f'
            nlvl = max_num_lvl
        elif lvlspan >= 0.1:
            format = '%.2f'
            nlvl = max_num_lvl
        else:
            format = '%.3e'
            nlvl = max_num_lvl
    else:
        format = '%.0f'
        round_up = 10 ** max(0, int(np.log10(max(abs(minlvl), abs(maxlvl))) - 2))
        if maxlvl > 0:
            maxlvl = round_up * int(1 + maxlvl / round_up) * (1 + 1e-10)
        if minlvl < 0:
            minlvl = -round_up * int(1 - minlvl / round_up) * (1 + 1e-10)
    if max_lvl is not None:
        maxlvl = max_lvl
    if min_lvl is not None:
        minlvl = min_lvl
    levels = np.linspace(minlvl, maxlvl, nlvl)
    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_title(quantity_title)
    ax.set_xlabel('x / cm')
    ax.set_ylabel('y / cm')

    # https://problemsolvingwithpython.com/06-Plotting-with-Matplotlib/06.14-Contour-Plots/
    # https://www.python-course.eu/matplotlib_contour_plot.php
    # https://matplotlib.org/3.2.1/tutorials/colors/colormaps.html
    line_color = 'k'
    fill_color = 'jet'
    ax.set_aspect('equal')
    contour = ax.contour(X, Y, Z, levels, colors=line_color)
    ax.clabel(contour, colors=line_color, fmt=format, fontsize=8)
    contour_filled = plt.contourf(X, Y, Z, levels, cmap=fill_color)
    plt.colorbar(contour_filled, format=format, label=default_q_units[quantity])


def rc_unit_step():
    R = 2.0
    tau = 1.0
    xlabel = r'$t$'
    ylabel = r'$V\/\/\mathrm{or}\/\/I$'
    title = ''
    labels = (r'$V_{\mathrm{RC}}$', r'$I$')
    format_ids = ((2, 0, 0), (2, 8, 1))
    x = np.linspace(-1, 5, 5000)
    y2 = (x >= 0) * 1.0
    y1 = y2 * R * (1 - np.exp(-x / tau))
    hdl = plot_xy((x, x), (y1, y2), xlabel=xlabel, ylabel=ylabel, title=title, labels=labels, format_ids=format_ids)
    show_plot(hdl)


if __name__ == '__main__':
    rc_unit_step()
