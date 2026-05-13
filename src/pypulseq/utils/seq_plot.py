from __future__ import annotations

import contextlib
import itertools
import math
import typing

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from pypulseq.calc_rf_center import calc_rf_center
from pypulseq.Sequence import parula
from pypulseq.supported_labels_rf_use import get_supported_labels
from pypulseq.utils.cumsum import cumsum

try:
    import mplcursors

    _MPLCURSORS_AVAILABLE = True
except ImportError:
    _MPLCURSORS_AVAILABLE = False

if typing.TYPE_CHECKING:
    from pypulseq.Sequence.sequence import Sequence


class SeqPlot:
    """
    Interactive plotter for a Pulseq `Sequence` object.

    Parameters
    ----------
    seq : Sequence
        The Pulseq sequence object to plot.
    label : str, default=str()
        Plot label values for ADC events. Valid labels are accepted as a comma-separated list.
    save : bool, default=False
        Boolean flag indicating if plots should be saved. The two figures will be saved as JPG with numerical
        suffixes to the filename 'seq_plot'.
    show_blocks : bool, default=False
        Boolean flag to indicate if grid and tick labels at the block boundaries are to be plotted.
    time_range : iterable, default=(0, np.inf)
        Time range (x-axis limits) for plotting the sequence. Default is 0 to infinity (entire sequence).
    time_disp : str, default='s'
        Time display type, must be one of `s`, `ms` or `us`.
    grad_disp : str, default='kHz/m'
        Gradient display unit, must be one of `kHz/m` or `mT/m`.
    plot_now : bool, default=True
        If true, function immediately shows the plots, blocking the rest of the code until plots are exited.
        If false, plots are shown when plt.show() is called. Useful if plots are to be modified.
    clear : bool, default=True
        If True, clear existing figures before plotting (default behavior).
        If False, overlay on existing figures for sequence comparison.
    overlay : SeqPlot or None, default=None
        If provided, overlay this plot on the figures from the given SeqPlot object. Overrides fig1, fig2, and sets clear=False.
    stacked : bool, default=False
        If True, plot all channels (ADC, RF mag, RF phase, Gx, Gy, Gz) in a single stacked figure (MATLAB Pulseq style).
        If False, use separate figures for RF/ADC and gradients.
    show_guides : bool, default=False
        If True, enable dynamic vertical hairline guides that follow the cursor. Requires `mplcursors`.
    show_rf_shim : bool, default=False
        If True, plot per-channel RF magnitude (Fig 3) and RF phase (Fig 4) for pTx sequences.
        Shim weights are read from ``rf.shim`` (shape ``(num_tx_ch, 2)``, columns: magnitude, phase)
        when present. Without shim data the raw signal magnitude and phase are shown uniformly on
        all channels.

    Attributes
    ----------
    fig1 : matplotlib.figure.Figure
        Figure containing RF/ADC channels (or all channels if stacked).
    fig2 : matplotlib.figure.Figure
        Figure containing Gradient channels (same as fig1 if stacked).
    ax1 : tuple of matplotlib.axes.Axes
        Tuple of axes for fig1: (sp11, sp12, sp13) if not stacked, or (sp11, sp12, sp13, sp21, sp22, sp23) if stacked.
    ax2 : tuple of matplotlib.axes.Axes
        Tuple of axes for fig2: (sp21, sp22, sp23) if not stacked, or same as ax1 if stacked.
    vlines : dict (axis -> Line2D) if show_guides enabled
    """

    def __init__(
        self,
        seq: Sequence,
        label: str = str(),
        show_blocks: bool = False,
        save: bool = False,
        time_range=(0, np.inf),
        time_disp: str = 's',
        grad_disp: str = 'kHz/m',
        plot_now: bool = True,
        clear: bool = True,
        overlay: 'SeqPlot' = None,
        stacked: bool = False,
        show_guides: bool = False,
        show_rf_shim: bool = False,
    ):
        # Handle optional dependencies
        if _MPLCURSORS_AVAILABLE is False:
            show_guides = False

        # Prepare fig1/fig2 from overlay if provided
        if overlay is not None:
            if overlay.__class__.__name__ != 'SeqPlot':  # not sure why isinstance() does not work here
                raise ValueError('overlay must be an instance of SeqPlot or None')

            # If the overlay's figure objects have been closed, create new figures instead.
            fig1 = (
                overlay.fig1
                if getattr(overlay, 'fig1', None) and plt.fignum_exists(getattr(overlay.fig1, 'number', None))
                else None
            )
            fig2 = (
                overlay.fig2
                if getattr(overlay, 'fig2', None) and plt.fignum_exists(getattr(overlay.fig2, 'number', None))
                else None
            )

            # Force using overlay stacking mode and avoid clearing the overlay by default.
            stacked = overlay.stacked
            clear = False
        else:
            fig1, fig2 = None, None

        self.seq = seq
        self.stacked = stacked
        self._cursors = []
        self._vlines = None  # populated if show_guides enabled
        self._guide_cids = []  # mpl_connect IDs for motion events
        self._show_guides = show_guides

        # Respect plot_now even when matplotlib interactive mode is enabled:
        # if plot_now is False and interactive mode is currently on, temporarily turn it off
        prev_interactive = plt.isinteractive()
        turned_off_interactive = False
        if not plot_now and prev_interactive:
            plt.ioff()
            turned_off_interactive = True

        try:
            handles = _seq_plot(
                seq,
                label=label,
                save=save,
                show_blocks=show_blocks,
                time_range=time_range,
                time_disp=time_disp,
                grad_disp=grad_disp,
                clear=clear,
                fig1=fig1,
                fig2=fig2,
                stacked=stacked,
                show_rf_shim=show_rf_shim,
            )
        finally:
            # restore interactive state if we changed it
            if turned_off_interactive:
                plt.ion()

        if stacked:
            self.fig1, self.ax1, self.fig3, self.ax3, self.fig4, self.ax4 = handles
            self.fig2, self.ax2 = None, ()
        else:
            self.fig1, self.ax1, self.fig2, self.ax2, self.fig3, self.ax3, self.fig4, self.ax4 = handles

        # If overlay was provided and its figures existed, those figures are being reused:
        # ensure the canvas is refreshed (but do not force show here; respect plot_now).
        if overlay is not None:
            for fig in (self.fig1, self.fig2):
                if fig is not None:
                    with contextlib.suppress(Exception):
                        fig.canvas.draw_idle()

        if _MPLCURSORS_AVAILABLE:
            self._setup_cursor(self.fig1)
            if not stacked:  # Avoid double setup if same figure
                self._setup_cursor(self.fig2)

        # Setup dynamic guides if requested and not already provided by overlay
        if self._show_guides:
            # If overlay provided and overlay already has vlines, reuse them
            if overlay is not None and getattr(overlay, '_vlines', None):
                self._vlines = overlay._vlines
                self._guide_cids = overlay._guide_cids
            else:
                # create guides: one vertical line per unique axis in ax1 + ax2
                axes = tuple(self.ax1) + tuple(self.ax2)
                unique_axes = []
                for ax in axes:
                    if ax not in unique_axes:
                        unique_axes.append(ax)
                self._vlines = {}
                for ax in unique_axes:
                    ln = ax.axvline(0.0, color='r', linestyle='--', linewidth=1.0, visible=False, zorder=1000)
                    self._vlines[ax] = ln

                # Motion handler
                def _motion(event):
                    if event.inaxes in unique_axes and event.xdata is not None:
                        x = event.xdata
                        for ln in self._vlines.values():
                            ln.set_xdata([x])
                            ln.set_visible(True)
                        for fig in {self.fig1, self.fig2}:
                            if fig is not None:
                                with contextlib.suppress(Exception):
                                    fig.canvas.draw_idle()
                    else:
                        for ln in self._vlines.values():
                            if ln.get_visible():
                                ln.set_visible(False)
                        for fig in {self.fig1, self.fig2}:
                            if fig is not None:
                                with contextlib.suppress(Exception):
                                    fig.canvas.draw_idle()

                canvases = []
                if self.fig1 is not None:
                    canvases.append(self.fig1.canvas)
                if self.fig2 is not None and self.fig2 is not self.fig1:
                    canvases.append(self.fig2.canvas)
                for canvas in canvases:
                    cid = canvas.mpl_connect('motion_notify_event', _motion)
                    self._guide_cids.append((canvas, cid))

        # Only show now if requested. If plot_now is False, caller will manage plt.show()
        if plot_now:
            self.show()

    def show(self):
        plt.show()

    def _setup_cursor(self, fig):
        for ax in fig.axes:
            lines = ax.get_lines()
            for line in lines:
                with contextlib.suppress(Exception):
                    cursor = mplcursors.cursor(line, multiple=True)
                    cursor.connect('add', lambda sel: self._on_datatip(sel))
                    cursor.connect('remove', lambda sel: self._hide_datatip_guides(sel))  # new
                    self._cursors.append(cursor)

    def _on_datatip(self, sel):
        """
        Called when a datatip is created (user clicks via mplcursors).
        Populate annotation text and, if guides exist, move them to the selected x position.
        """
        artist = sel.artist
        ax = artist.axes
        x, y = sel.target
        ylabel = ax.get_ylabel().lower()

        if ylabel.startswith('adc') or (
            ylabel.startswith('rf/adc') and artist.get_linestyle() == 'none' and artist.get_marker() == '.'
        ):
            field = 'adc'
        else:
            field = ylabel[:2]

        # Convert the displayed x coordinate back to the sequence time units
        # _seq_plot stores a t_factor on each figure as _seq_t_factor
        fig = artist.axes.figure
        t_factor = getattr(fig, '_seq_t_factor', 1.0)
        seq_time = x / t_factor

        # Try finding corresponding block; if it fails (click outside sequence/time range) handle gracefully
        try:
            block_index = self.seq.find_block_by_time(seq_time)
            rb = self.seq.get_raw_block_content_IDs(block_index)
        except Exception:
            block_index = None
            rb = None

        lines_txt = [f't: {x:.3f}', f'Y: {y:.3f}']

        if rb is not None and block_index not in (None, 0):
            val = getattr(rb, field, None)
            try:
                display_blk = block_index + 1
            except Exception:
                display_blk = block_index
            if val is not None:
                try:
                    if field[0] == 'a':
                        name = self.seq.adc_id2name_map[val]
                    elif field[0] == 'r':
                        name = self.seq.rf_id2name_map[val]
                    else:
                        name = self.seq.grad_id2name_map[val]

                    lines_txt.append(f"blk: {display_blk} {field}_id: {val} '{name}'")
                except Exception:
                    lines_txt.append(f'blk: {display_blk} {field}_id: {val}')
            else:
                lines_txt.append(f'blk: {display_blk}')
        else:
            # Couldn't resolve a block for this x (outside plotted time_range or no block)
            lines_txt.append('blk: 1')

        sel.annotation.set_text('\n'.join(lines_txt))

        # If we have dynamic guides, move them to the datatip x position and show them
        if getattr(self, '_vlines', None):
            x_coord = x
            for ln in self._vlines.values():
                ln.set_xdata([x_coord])
                ln.set_visible(True)
            for fig in {self.fig1, self.fig2}:
                if fig is not None:
                    with contextlib.suppress(Exception):
                        fig.canvas.draw_idle()

        self._update_guides()

    def _hide_datatip_guides(self, sel):  # noqa
        # Hide guides when datatip removed (connected to mplcursors 'remove' event)
        if getattr(self, '_vlines', None):
            for ln in self._vlines.values():
                ln.set_visible(False)
            for fig in {self.fig1, self.fig2}:
                if fig is not None:
                    with contextlib.suppress(Exception):
                        fig.canvas.draw_idle()

    def _update_guides(self):
        # Update autoscale for all axes involved and redraw figures
        for ax in tuple(self.ax1) + tuple(self.ax2):  # Flatten tuples for iteration
            with contextlib.suppress(Exception):
                ax.relim()
                ax.autoscale_view()

        for fig in (self.fig1, self.fig2):
            if fig is not None:
                with contextlib.suppress(Exception):
                    fig.canvas.draw_idle()


def _seq_plot(
    seq,
    label,
    show_blocks,
    save,
    time_range,
    time_disp,
    grad_disp,
    clear,
    fig1,
    fig2,
    stacked,
    show_rf_shim: bool = False,
):
    mpl.rcParams['lines.linewidth'] = 0.75  # Set default Matplotlib linewidth

    valid_time_units = ['s', 'ms', 'us']
    valid_grad_units = ['kHz/m', 'mT/m']
    valid_labels = get_supported_labels()
    if not all(isinstance(x, (int, float)) for x in time_range) or len(time_range) != 2:
        raise ValueError('Invalid time range')
    if time_disp not in valid_time_units:
        raise ValueError('Unsupported time unit')

    if grad_disp not in valid_grad_units:
        raise ValueError('Unsupported gradient unit. Supported gradient units are: ' + str(valid_grad_units))

    # If figs were provided but closed (None), create new ones
    if stacked:
        # Reuse existing fig1 when provided (overlay) or create new figure.
        fig1 = plt.figure() if fig1 is None else fig1
        fig2 = fig1  # Use same figure

        # If clear requested clear, otherwise keep existing axes.
        if clear:
            with contextlib.suppress(Exception):
                fig1.clear()

        # Try to reuse existing axes when overlay/stacked is used (previous implementation always created new axes,
        # which caused overlay to fail when stacked=True).
        fig1_axes = fig1.get_axes()
        if not fig1_axes or clear:
            sp11 = fig1.add_subplot(611)
            sp12 = fig1.add_subplot(612, sharex=sp11)
            sp13 = fig1.add_subplot(613, sharex=sp11)
            sp21 = fig1.add_subplot(614, sharex=sp11)
            sp22 = fig1.add_subplot(615, sharex=sp11)
            sp23 = fig1.add_subplot(616, sharex=sp11)
        else:
            # Reuse first six axes if present; create any missing ones and preserve sharex with sp11.
            if len(fig1_axes) >= 6:
                sp11, sp12, sp13, sp21, sp22, sp23 = fig1_axes[:6]
            else:
                # At least one axis exists; use the first as sp11, create the rest
                if len(fig1_axes) >= 1:
                    sp11 = fig1_axes[0]
                else:
                    sp11 = fig1.add_subplot(611)
                # create remaining axes with sharex=sp11
                existing = len(fig1_axes)
                mapping_positions = [612, 613, 614, 615, 616]
                created_axes = []
                for pos in mapping_positions[existing - 1 if existing > 0 else 0 :]:
                    # pos is in the form 6xy; still safe to call add_subplot with position int
                    created_axes.append(fig1.add_subplot(pos, sharex=sp11))
                # now assemble sp12..sp23 from existing+created
                all_axes = fig1.get_axes()
                # ensure we pick the first six axes in document order
                sp_axes = all_axes[:6]
                # pad if somehow less than 6 (unlikely)
                while len(sp_axes) < 6:
                    sp_axes.append(fig1.add_subplot(616, sharex=sp11))
                sp11, sp12, sp13, sp21, sp22, sp23 = sp_axes[:6]
    else:
        # Two figures
        fig1 = plt.figure() if fig1 is None else fig1
        fig2 = plt.figure() if fig2 is None else fig2

        # Clear existing figures if clear=True
        if clear:
            with contextlib.suppress(Exception):
                fig1.clear()
            with contextlib.suppress(Exception):
                fig2.clear()

        # Create or reuse subplots of fig1
        fig1_axes = fig1.get_axes()
        if not fig1_axes or clear:
            sp11 = fig1.add_subplot(311)
            sp12 = fig1.add_subplot(312, sharex=sp11)
            sp13 = fig1.add_subplot(313, sharex=sp11)
        else:
            sp11, sp12, sp13 = fig1_axes[:3]

        # Create or reuse subplots of fig2
        fig2_axes = fig2.get_axes()
        if not fig2_axes or clear:
            sp21 = fig2.add_subplot(311, sharex=sp11)
            sp22 = fig2.add_subplot(312, sharex=sp11)
            sp23 = fig2.add_subplot(313, sharex=sp11)
        else:
            sp21, sp22, sp23 = fig2_axes[:3]

    t_factor_list = [1, 1e3, 1e6]
    t_factor = t_factor_list[valid_time_units.index(time_disp)]

    g_factor_list = [1e-3, 1e3 / seq.system.gamma]
    g_factor = g_factor_list[valid_grad_units.index(grad_disp)]

    t0 = 0
    label_defined = False
    label_idx_to_plot = []
    label_legend_to_plot = []
    label_store = {}
    for i in range(len(valid_labels)):
        label_store[valid_labels[i]] = 0
        if valid_labels[i] in label.upper():
            label_idx_to_plot.append(i)
            label_legend_to_plot.append(valid_labels[i])

    if len(label_idx_to_plot) != 0:
        p = parula.main(len(label_idx_to_plot) + 1)
        label_colors_to_plot = p(np.arange(len(label_idx_to_plot)))
        cycler = mpl.cycler(color=label_colors_to_plot)
        sp11.set_prop_cycle(cycler)

    # Block timings
    block_edges = np.cumsum([0] + [x[1] for x in sorted(seq.block_durations.items())])
    block_edges_in_range = block_edges[(block_edges >= time_range[0]) * (block_edges <= time_range[1])]
    if show_blocks:
        for sp in [sp11, sp12, sp13, sp21, sp22, sp23]:
            sp.set_xticks(t_factor * block_edges_in_range)
            sp.set_xticklabels(sp.get_xticklabels(), rotation=90)

    for block_counter in seq.block_events:
        block = seq.get_block(block_counter)
        is_valid = time_range[0] <= t0 + seq.block_durations[block_counter] and t0 <= time_range[1]
        if is_valid:
            if getattr(block, 'label', None) is not None:
                for i in range(len(block.label)):
                    if block.label[i].type == 'labelinc':
                        label_store[block.label[i].label] += block.label[i].value
                    else:
                        label_store[block.label[i].label] = block.label[i].value
                label_defined = True

            if getattr(block, 'trigger', None) is not None:  # Trigger
                for trigger in block.trigger.values():
                    if trigger.type == 'output':
                        t = trigger.delay
                        sp12.plot(t_factor * (t0 + t), 0, marker='D', color=(0, 0.5, 0), linestyle='None')
                        t += np.asarray([0, trigger.duration])
                        sp12.plot(t_factor * (t0 + t), [0, 0], linestyle='-', marker='.', color=(0, 0.5, 0))
                    if trigger.type == 'trigger':
                        t = trigger.delay
                        sp12.plot(t_factor * (t0 + t), 0, marker='>', color='b', linestyle='None')
                        sp12.plot(t_factor * (t0 + t), 0, marker='.', color='b', linestyle='None')

            if getattr(block, 'adc', None) is not None:  # ADC
                adc = block.adc
                # From Pulseq: According to the information from Klaus Scheffler and indirectly from Siemens this
                # is the present convention - the samples are shifted by 0.5 dwell
                t = adc.delay + (np.arange(int(adc.num_samples)) + 0.5) * adc.dwell
                sp11.plot(t_factor * (t0 + t), np.zeros(len(t)), 'rx')

                if adc.phase_modulation is None or len(adc.phase_modulation) == 0:
                    phase_modulation = 0.0
                else:
                    phase_modulation = adc.phase_modulation

                full_freq_offset = np.atleast_1d(adc.freq_offset + adc.freq_ppm * 1e-6 * seq.system.B0)
                full_phase_offset = np.atleast_1d(
                    adc.phase_offset + adc.phase_ppm * 1e-6 * seq.system.B0 + phase_modulation
                )

                sp13.plot(
                    t_factor * (t0 + t),
                    np.angle(np.exp(1j * full_phase_offset) * np.exp(1j * 2 * math.pi * t * full_freq_offset)),
                    'b.',
                    markersize=0.25,
                )

                if label_defined and len(label_idx_to_plot) != 0:
                    arr_label_store = list(label_store.values())
                    lbl_vals = np.take(arr_label_store, label_idx_to_plot)
                    t = t0 + adc.delay + (adc.num_samples - 1) / 2 * adc.dwell
                    _t = [t_factor * t] * len(lbl_vals)
                    # Plot each label individually to retrieve each corresponding Line2D object
                    p = itertools.chain.from_iterable(
                        [sp11.plot(__t, _lbl_vals, '.') for __t, _lbl_vals in zip(_t, lbl_vals, strict=True)]
                    )
                    if len(label_legend_to_plot) != 0:
                        sp11.legend(list(p), label_legend_to_plot, loc='upper left')
                        label_legend_to_plot = []

            if getattr(block, 'rf', None) is not None:  # RF
                rf = block.rf
                time_center, index_center = calc_rf_center(rf)
                time = rf.t
                signal = rf.signal

                if signal.shape[0] == 2 and rf.freq_offset != 0:
                    num_samples = min(int(abs(rf.freq_offset)), 256)
                    time = np.linspace(time[0], time[-1], num_samples)
                    signal = np.linspace(signal[0], signal[-1], num_samples)

                if abs(signal[0]) != 0:
                    signal = np.concatenate(([0], signal))
                    time = np.concatenate(([time[0]], time))
                    index_center += 1

                if abs(signal[-1]) != 0:
                    signal = np.concatenate((signal, [0]))
                    time = np.concatenate((time, [time[-1]]))

                signal_is_real = max(np.abs(np.imag(signal))) / max(np.abs(np.real(signal))) < 1e-6

                full_freq_offset = rf.freq_offset + rf.freq_ppm * 1e-6 * seq.system.B0
                full_phase_offset = rf.phase_offset + rf.phase_ppm * 1e-6 * seq.system.B0

                # If off-resonant and rectangular (2 samples), interpolate the pulse
                if len(signal) == 2 and full_freq_offset != 0:
                    num_interp = min(int(abs(full_freq_offset)), 256)
                    time = np.linspace(time[0], time[-1], num_interp)
                    signal = np.linspace(signal[0], signal[-1], num_interp)
                if abs(signal[0]) != 0:  # fix strangely looking phase / amplitude in the beginning
                    signal = np.concatenate([[0], signal])
                    time = np.concatenate([[time[0]], time])
                if abs(signal[-1]) != 0:  # fix strangely looking phase / amplitude at the end
                    signal = np.concatenate([signal, [0]])
                    time = np.concatenate([time, [time[-1]]])

                # Compute time vector with delay applied
                time_with_delay = t_factor * (t0 + time + rf.delay)
                time_center_with_delay = t_factor * (t0 + time_center + rf.delay)

                # Choose plot behavior based on realness of signal
                if signal_is_real:
                    # Plot real part of signal
                    sp12.plot(time_with_delay, np.real(signal))

                    # Include sign(real(signal)) factor like MATLAB
                    phase_corrected = (
                        signal
                        * np.sign(np.real(signal))
                        * np.exp(1j * full_phase_offset)
                        * np.exp(1j * 2 * math.pi * time * full_freq_offset)
                    )
                    sc_corrected = (
                        signal[index_center]
                        * np.exp(1j * full_phase_offset)
                        * np.exp(1j * 2 * math.pi * time[index_center] * full_freq_offset)
                    )

                    sp13.plot(
                        time_with_delay,
                        np.angle(phase_corrected),
                        time_center_with_delay,
                        np.angle(sc_corrected),
                        'xb',
                    )
                else:
                    # Plot magnitude of complex signal
                    sp12.plot(time_with_delay, np.abs(signal))

                    # Plot angle of complex signal
                    phase_corrected = (
                        signal * np.exp(1j * full_phase_offset) * np.exp(1j * 2 * math.pi * time * full_freq_offset)
                    )
                    sc_corrected = (
                        signal[index_center]
                        * np.exp(1j * full_phase_offset)
                        * np.exp(1j * 2 * math.pi * time[index_center] * full_freq_offset)
                    )

                    sp13.plot(
                        time_with_delay,
                        np.angle(phase_corrected),
                        time_center_with_delay,
                        np.angle(sc_corrected),
                        'xb',
                    )

            grad_channels = ['gx', 'gy', 'gz']
            for x in range(len(grad_channels)):  # Gradients
                if getattr(block, grad_channels[x], None) is not None:
                    grad = getattr(block, grad_channels[x])
                    if grad.type == 'grad':
                        # We extend the shape by adding the first and the last points in an effort of making the
                        # display a bit less confusing...
                        time = grad.delay + np.array([0, *grad.tt, grad.shape_dur])
                        waveform = g_factor * np.array((grad.first, *grad.waveform, grad.last))
                    else:
                        time = np.array(
                            cumsum(
                                0,
                                grad.delay,
                                grad.rise_time,
                                grad.flat_time,
                                grad.fall_time,
                            )
                        )
                        waveform = g_factor * grad.amplitude * np.array([0, 0, 1, 1, 0])
                    [sp21, sp22, sp23][x].plot(t_factor * (t0 + time), waveform)

            # Soft delays - plot as shaded regions with annotations
            if getattr(block, 'soft_delay', None) is not None:
                soft_delay = block.soft_delay
                block_duration = seq.block_durations[block_counter]
                t_mid = t0 + block_duration / 2  # Middle of the block

                # Add shaded region spanning the soft delay block duration on all subplots
                sp13.axvspan(t_factor * t0, t_factor * (t0 + block_duration), alpha=0.2, color='orange')
                sp12.axvspan(t_factor * t0, t_factor * (t0 + block_duration), alpha=0.2, color='orange')
                sp11.axvspan(t_factor * t0, t_factor * (t0 + block_duration), alpha=0.2, color='orange')
                for sp2x in [sp21, sp22, sp23]:
                    sp2x.axvspan(t_factor * t0, t_factor * (t0 + block_duration), alpha=0.2, color='orange')

                # Add text annotation with soft delay hint on the RF/ADC phase subplot
                y_lim = sp13.get_ylim()
                y_range = y_lim[1] - y_lim[0]
                y_pos = y_lim[0] + 0.1 * y_range
                y_text = y_lim[0] + 0.3 * y_range

                sp13.annotate(
                    f'{soft_delay.hint}',
                    xy=(t_factor * t_mid, y_pos),
                    xytext=(t_factor * t_mid, y_text),
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    bbox={'boxstyle': 'round,pad=0.3', 'facecolor': 'orange', 'alpha': 0.7},
                )

        t0 += seq.block_durations[block_counter]

    # Set axis labels
    sp11.set_ylabel('ADC')
    sp12.set_ylabel('RF mag (Hz)')
    sp13.set_ylabel('RF/ADC phase (rad)')
    sp13.set_xlabel(f't ({time_disp})')
    sp21.set_ylabel(f'Gx ({grad_disp})')
    sp22.set_ylabel(f'Gy ({grad_disp})')
    sp23.set_ylabel(f'Gz ({grad_disp})')
    if not stacked:
        sp23.set_xlabel(f't ({time_disp})')

    # Setting display limits
    disp_range = t_factor * np.array([time_range[0], min(t0, time_range[1])])
    for sp in [sp11, sp12, sp13, sp21, sp22, sp23]:
        sp.set_xlim(disp_range)

    # Enable grid on all subplots (explicitly set to True, don't toggle)
    for sp in [sp11, sp12, sp13, sp21, sp22, sp23]:
        sp.grid(True)

    # Store the t_factor on the figures so interactive callbacks can convert displayed x back to sequence time
    fig1._seq_t_factor = t_factor
    if fig2 is not None:
        fig2._seq_t_factor = t_factor

    fig1.tight_layout()
    if not stacked:
        fig2.tight_layout()
    if save:
        if stacked:
            fig1.savefig('seq_plot_stacked.jpg')
        else:
            fig1.savefig('seq_plot1.jpg')
            fig2.savefig('seq_plot2.jpg')

    # --- Optional pTx shim plots (Fig 3 & Fig 4): built inline when show_rf_shim=True ---
    fig3: 'plt.Figure | None' = None
    fig4: 'plt.Figure | None' = None
    ax3: tuple = ()
    ax4: tuple = ()

    if show_rf_shim:
        # Determine num_tx_ch from the first rf_shim extension found in the sequence.
        num_tx_ch = 0
        for block_counter in seq.block_events:
            block = seq.get_block(block_counter)
            rf_shim_ext = getattr(block, 'rf_shim', None)
            if rf_shim_ext is not None and rf_shim_ext.shim_vector.size > 0:
                num_tx_ch = rf_shim_ext.shim_vector.size
                break

        if num_tx_ch == 0:
            # No rf_shim extensions found in the sequence; skip pTx plots silently.
            print('No rf_shim extensions found in the sequence')
            show_rf_shim = False

    if show_rf_shim:
        fig3 = plt.figure()
        fig4 = plt.figure()
        fig3.suptitle('RF Magnitude per TX Channel')
        fig4.suptitle('RF Phase per TX Channel')

        ax3_list: list = []
        ax4_list: list = []
        for i in range(num_tx_ch):
            share3 = sp11 if i == 0 else ax3_list[0]
            share4 = sp11 if i == 0 else ax4_list[0]
            ax3_list.append(fig3.add_subplot(num_tx_ch, 1, i + 1, sharex=share3))
            ax4_list.append(fig4.add_subplot(num_tx_ch, 1, i + 1, sharex=share4))

        t0_ptx = 0.0
        for block_counter in seq.block_events:
            block = seq.get_block(block_counter)
            block_dur = seq.block_durations[block_counter]
            is_valid = time_range[0] <= t0_ptx + block_dur and t0_ptx <= time_range[1]

            if is_valid and getattr(block, 'rf', None) is not None:
                rf = block.rf

                # Resolve shim weights from the rf_shim extension (block.rf_shim.shim_vector),
                # which is a complex array of shape (num_tx_ch,) with magnitude and phase encoded.
                # Fall back to unweighted display when no rf_shim extension is present in the block.
                rf_shim_ext = getattr(block, 'rf_shim', None)
                shim_vector = rf_shim_ext.shim_vector if rf_shim_ext is not None else None
                has_shim = shim_vector is not None and shim_vector.size > 0
                if has_shim:
                    shim_mag = np.abs(shim_vector)  # per-channel magnitude weight
                    shim_pha = np.angle(shim_vector)  # per-channel constant phase offset (rad)

                # Zero-pad edges for clean fill_between rendering
                signal_ptx = rf.signal.copy()
                time_ptx = rf.t.copy()
                signal_ptx = np.concatenate(([0], signal_ptx, [0]))
                time_ptx = np.concatenate(([time_ptx[0]], time_ptx, [time_ptx[-1]]))


                signal_ptx_is_real = (
                        max(np.abs(np.imag(signal_ptx))) / max(np.abs(np.real(signal_ptx))) < 1e-6
                )
                signal_ptx_mag = np.real(signal_ptx) if signal_ptx_is_real else np.abs(signal_ptx)

                t_plot_ptx = t_factor * (t0_ptx + time_ptx + rf.delay)

                full_freq_offset_ptx = rf.freq_offset + rf.freq_ppm * 1e-6 * seq.system.B0
                full_phase_offset_ptx = rf.phase_offset + rf.phase_ppm * 1e-6 * seq.system.B0

                for i_ch in range(num_tx_ch):
                    if not has_shim:
                        # No shim data: display raw signal uniformly across all channel rows
                        mag = signal_ptx_mag
                        pha = np.angle(
                            signal_ptx
                            * np.exp(1j * full_phase_offset_ptx)
                            * np.exp(1j * 2 * math.pi * time_ptx * full_freq_offset_ptx)
                        )
                    else:
                        # Apply per-channel magnitude weight and constant phase offset
                        mag = np.where(np.abs(signal_ptx) > 0, shim_mag[i_ch], 0.0)
                        pha = np.angle(
                            signal_ptx
                            * np.sign(np.real(signal_ptx))
                            * np.exp(1j * shim_pha[i_ch])
                        )
                        pha = np.where(np.abs(signal_ptx) > 0, pha, 0.0)

                    ax3_list[i_ch].plot(t_plot_ptx, mag)
                    ax3_list[i_ch].fill_between(t_plot_ptx, 0, mag, alpha=0.4) # can be removed
                    ax4_list[i_ch].plot(t_plot_ptx, pha)
                    ax4_list[i_ch].fill_between(t_plot_ptx, 0, pha, alpha=0.4) # can be removed

            t0_ptx += block_dur

        # Axis decoration – applied once after the block loop
        disp_range_ptx = t_factor * np.array([time_range[0], min(t0_ptx, time_range[1])])
        for i_ch in range(num_tx_ch):
            is_last = i_ch == num_tx_ch - 1
            for ax in (ax3_list[i_ch], ax4_list[i_ch]):
                ax.set_ylabel(f'CH{i_ch}')
                ax.set_xlim(disp_range_ptx)
                ax.grid(True)
                ax.xaxis.set_visible(is_last)
                if is_last:
                    ax.set_xlabel(f't ({time_disp})')

        mag3_max = max(ax.get_ylim()[1] for ax in ax3_list)
        for ax in ax3_list:
            ax.set_ylim(0, mag3_max)

        for i_ch in range(num_tx_ch):
            ax4_list[i_ch].set_ylim(-np.pi, np.pi)

        ax3_list[0].set_title('Magnitude Factor (a.u.)', fontsize='small', loc='right')
        ax4_list[0].set_title('Phase Factor (rad)', fontsize='small', loc='right')

        fig3.tight_layout()
        fig4.tight_layout()

        if save:
            fig3.savefig('seq_ptx_rf_mag.jpg')
            fig4.savefig('seq_ptx_rf_phase.jpg')

        ax3 = tuple(ax3_list)
        ax4 = tuple(ax4_list)

    if stacked:
        return fig1, (sp11, sp12, sp13, sp21, sp22, sp23), fig3, ax3, fig4, ax4
    else:
        return fig1, (sp11, sp12, sp13), fig2, (sp21, sp22, sp23), fig3, ax3, fig4, ax4
