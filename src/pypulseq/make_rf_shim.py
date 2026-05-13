from types import SimpleNamespace

import math
import numpy as np


def make_rf_shim(shim_vec: np.ndarray) -> SimpleNamespace:
    """
    Create an RF shim event for use in a sequence block.

    The event is only valid when the block also contains an RF pulse.

    See also `pypulseq.Sequence.sequence.Sequence.add_block()`.

    Parameters
    ----------
    shim_vec : np.ndarray
        A 1D complex array of length num_tx_ch describing the relative
        amplitude and phase of each transmit channel.

    Returns
    -------
    SimpleNamespace
        RF shim event with fields ``type='rf_shim'`` and ``shim_vector``.

    Raises
    ------
    ValueError
        If ``shim_vec`` is not a non-empty 1D array.

    Examples
    --------
    ::

        shim = make_rf_shim(np.array([0.5+0j, 0.5*np.exp(1j*np.pi/2)]))
    """
    shim_vec = np.asarray(shim_vec, dtype=complex)
    if shim_vec.ndim != 1 or shim_vec.size == 0:
        raise ValueError("'shim_vec' must be a non-empty 1D array.")

    return SimpleNamespace(type='rf_shim', shim_vector=shim_vec)


def get_tx_mode(mode: str) -> np.ndarray:
    """
    Return the shim vector for a built-in preset as a numpy array.

    Parameters
    ----------
    mode : str
        Name of a built-in preset. Available presets:

        - ``Nova_Head_8Tx_CP``
        - ``Nova_Head_8Tx_EP``
        - ``Prisma_Body_2Tx_CP``
        - ``Prisma_Body_2Tx_EP``
        - ``CimaX_Body_2Tx_CP``
        - ``CimaX_Body_2Tx_EP``

    Returns
    -------
    np.ndarray
        A copy of the preset's 1D complex shim vector.

    Raises
    ------
    ValueError
        If ``mode`` is not a known preset.

    Examples
    --------
    ::

        shim = make_rf_shim(get_tx_mode("Nova_Head_8Tx_CP"))
    """
    if mode not in _RF_SHIM_PRESETS:
        available = ", ".join(sorted(_RF_SHIM_PRESETS))
        raise ValueError(
            f"Unknown RF shim preset '{mode}'. "
            f"Available presets: {available}"
        )
    return _RF_SHIM_PRESETS[mode].copy()


# ---------------------------------------------------------------------------
# Built-in RF shim presets.
# Each entry is a 1D complex array of length num_tx_ch.
# Amplitudes are normalized such that the total power sums to 1.
# ---------------------------------------------------------------------------
_RF_SHIM_PRESETS: dict[str, np.ndarray] = {
    # Nova Medical 8-channel head coil – circular polarization mode
    "Nova_Head_8Tx_CP": np.array([
        0.35 * np.exp(1j *    0 * math.pi / 180),
        0.35 * np.exp(1j *  -45 * math.pi / 180),
        0.35 * np.exp(1j *  -90 * math.pi / 180),
        0.35 * np.exp(1j * -135 * math.pi / 180),
        0.35 * np.exp(1j * -180 * math.pi / 180),
        0.35 * np.exp(1j *  135 * math.pi / 180),
        0.35 * np.exp(1j *   90 * math.pi / 180),
        0.35 * np.exp(1j *   45 * math.pi / 180),
    ]),
    # Nova Medical 8-channel head coil – elliptical polarization mode
    "Nova_Head_8Tx_EP": np.array([
        0.35 * np.exp(1j *    0 * math.pi / 180),
        0.35 * np.exp(1j *  -90 * math.pi / 180),
        0.35 * np.exp(1j * -180 * math.pi / 180),
        0.35 * np.exp(1j *   90 * math.pi / 180),
        0.35 * np.exp(1j *    0 * math.pi / 180),
        0.35 * np.exp(1j *  -90 * math.pi / 180),
        0.35 * np.exp(1j * -180 * math.pi / 180),
        0.35 * np.exp(1j *   90 * math.pi / 180),
    ]),
    # Siemens Prisma 2-channel body coil – circular polarization mode
    "Prisma_Body_2Tx_CP": np.array([
        0.71 * np.exp(1j *   0 * math.pi / 180),
        0.71 * np.exp(1j *  90 * math.pi / 180),
    ]),
    # Siemens Prisma 2-channel body coil – elliptical polarization mode
    "Prisma_Body_2Tx_EP": np.array([
        0.48 * np.exp(1j *   0 * math.pi / 180),
        0.88 * np.exp(1j * 130 * math.pi / 180),
    ]),
    # Siemens CimaX 2-channel body coil – circular polarization mode
    "CimaX_Body_2Tx_CP": np.array([
        0.71 * np.exp(1j *   0 * math.pi / 180),
        0.71 * np.exp(1j *  90 * math.pi / 180),
    ]),
    # Siemens CimaX 2-channel body coil – elliptical polarization mode
    "CimaX_Body_2Tx_EP": np.array([
        0.48 * np.exp(1j *   0 * math.pi / 180),
        0.88 * np.exp(1j * 130 * math.pi / 180),
    ]),
}