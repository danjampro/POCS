from contextlib import suppress
import numpy as np
from astropy import units as u

from pocs.utils import error
from pocs.utils.logger import get_root_logger


def focus_metric(data, merit_function='vollath_F4', **kwargs):
    """Compute the focus metric.

    Computes a focus metric on the given data using a supplied merit function.
    The merit function can be passed either as the name of the function (must be
    defined in this module) or as a callable object. Additional keyword arguments
    for the merit function can be passed as keyword arguments to this function.

    Args:
        data (numpy array) -- 2D array to calculate the focus metric for.
        merit_function (str/callable) -- Name of merit function (if in
            pocs.utils.images) or a callable object.

    Returns:
        scalar: result of calling merit function on data
    """
    if isinstance(merit_function, str):
        try:
            merit_function = globals()[merit_function]
        except KeyError:
            raise KeyError(
                "Focus merit function '{}' not found in pocs.utils.images!".format(merit_function))

    return merit_function(data, **kwargs)


def vollath_F4(data, axis=None):
    """Compute F4 focus metric

    Computes the F_4 focus metric as defined by Vollath (1998) for the given 2D
    numpy array. The metric can be computed in the y axis, x axis, or the mean of
    the two (default).

    Arguments:
        data (numpy array) -- 2D array to calculate F4 on.
        axis (str, optional, default None) -- Which axis to calculate F4 in. Can
            be 'Y'/'y', 'X'/'x' or None, which will calculate the F4 value for
            both axes and return the mean.

    Returns:
        float64: Calculated F4 value for y, x axis or both
    """
    if axis == 'Y' or axis == 'y':
        return _vollath_F4_y(data)
    elif axis == 'X' or axis == 'x':
        return _vollath_F4_x(data)
    elif not axis:
        return (_vollath_F4_y(data) + _vollath_F4_x(data)) / 2
    else:
        raise ValueError(
            "axis must be one of 'Y', 'y', 'X', 'x' or None, got {}!".format(axis))


def mask_saturated(data, saturation_level=None, threshold=0.9, bit_depth=None, dtype=None,
                   logger=None):
    """Convert data to a masked array with saturated values masked.
    Args:
        data (array_like): The numpy data array.
        saturation_level (scalar, optional): The saturation level. If not given then the
            saturation level will be set to threshold times the maximum pixel value.
        threshold (float, optional): The fraction of the maximum pixel value to use as
            the saturation level, default 0.9.
        bit_depth (astropy.units.Quantity or int, optional): The effective bit depth of the
            data. If given the maximum pixel value will be assumed to be 2**bit_depth,
            otherwise an attempt will be made to infer the maximum pixel value from the
            data type of the data. If data is not an integer type the maximum pixel value
            cannot be inferred and an IllegalValue exception will be raised.
        dtype (numpy.dtype, optional): The requested dtype for the masked array. If not given
            the dtype of the masked array will be same as data.
    Returns:
        numpy.ma.array: The masked numpy array.
    Raises:
        error.IllegalValue: Raised if bit_depth is an astropy.units.Quantity object but the
            units are not compatible with either bits or bits/pixel.
        error.IllegalValue: Raised if neither saturation level or bit_depth are given, and data
            has a non integer data type.
    """
    if logger is None:
        logger = get_root_logger()
    if not saturation_level:
        if bit_depth is not None:
            try:
                with suppress(AttributeError):
                    bit_depth = bit_depth.to_value(unit=u.bit)
            except u.UnitConversionError:
                try:
                    bit_depth = bit_depth.to_value(unit=u.bit / u.pixel)
                except u.UnitConversionError:
                    raise error.IllegalValue("bit_depth must have units of bits or bits/pixel, "
                                             f"got {bit_depth!r}")

            bit_depth = int(bit_depth)
            logger.debug(f"Using bit depth {bit_depth!r}")
            saturation_level = threshold * (2**bit_depth - 1)
        else:
            # No bit depth specified, try to guess.
            logger.debug(f"Inferring bit_depth from data type, {data.dtype!r}")
            try:
                # Try to use np.iinfo to compute machine limits. Will work for integer types.
                saturation_level = threshold * np.iinfo(data.dtype).max
            except ValueError:
                # ValueError from np.iinfo means not an integer type.
                raise error.IllegalValue("Neither saturation_level or bit_depth given, and data "
                                         "is not an integer type. Cannot determine correct "
                                         "saturation level.")
    logger.debug(f"Masking image using saturation level {saturation_level!r}")
    # Convert data to masked array of requested dtype, mask values above saturation level.
    return np.ma.array(data, mask=(data > saturation_level), dtype=dtype)


def _vollath_F4_y(data):
    A1 = (data[1:] * data[:-1]).mean()
    A2 = (data[2:] * data[:-2]).mean()
    return A1 - A2


def _vollath_F4_x(data):
    A1 = (data[:, 1:] * data[:, :-1]).mean()
    A2 = (data[:, 2:] * data[:, :-2]).mean()
    return A1 - A2
