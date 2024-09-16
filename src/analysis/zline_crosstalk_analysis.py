import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from qualang_tools.plot import Fit

def get_freq_axes( axes ):
    f_axes = []
    for a in axes:
        ax_len = a.shape[-1]
        d = np.abs(a[0]-a[1])
        print(f"length: {ax_len}, delta:{d}")

        f_axes.append( np.fft.fftshift(np.fft.fftfreq(ax_len, d=d)) )
    return f_axes

from scipy.interpolate import RegularGridInterpolator

def get_weighted_pos( data, axes ):
    max_index = np.argmax(data)
    indices = np.unravel_index(max_index, data.shape)
    
    edge_min_idx = [indices[0]-2, indices[1]-2]
    edge_max_idx = [indices[0]+2, indices[1]+2]

    partial_amp = data[edge_min_idx[0]:edge_max_idx[0]+1,edge_min_idx[1]:edge_max_idx[1]+1]
    f_z_t_range = axes[0][edge_min_idx[0]:edge_max_idx[0]+1]
    f_z_c_range = axes[1][edge_min_idx[1]:edge_max_idx[1]+1]
    f_z_t_range_new = np.linspace(f_z_t_range[0],f_z_t_range[-1],10)
    f_z_c_range_new = np.linspace(f_z_c_range[0],f_z_c_range[-1],10)
    # print(partial_amp.shape, f_z_t_range.shape, f_z_c_range.shape)
    interp = RegularGridInterpolator((f_z_t_range, f_z_c_range), partial_amp, method="cubic",
                                     fill_value=None, bounds_error=False)
    # print(f_z_t_range, f_z_t_range_new)
    tv, cv = np.meshgrid(f_z_t_range_new, f_z_c_range_new, indexing="ij")
    f_z_target_pos, f_z_crosstalk_pos = get_max_pos(interp((tv, cv)), [f_z_t_range_new, f_z_c_range_new])

    # tv, cv = np.meshgrid(f_z_t_range_new, f_z_c_range_new, indexing="ij")
    # print(f_z_t_range, f_z_c_range)
    # f_z_target_pos = np.average(tv, weights=partial_amp)
    # f_z_crosstalk_pos = np.average(cv, weights=partial_amp)

    # print(f_z_target_pos, f_z_crosstalk_pos)
    return f_z_target_pos, f_z_crosstalk_pos

def get_max_pos( data, axes ):

    max_index = np.argmax(data)
    max_indices = np.unravel_index(max_index, data.shape)
    f_z_target_pos = axes[0][max_indices[0]]
    f_z_crosstalk_pos = axes[1][max_indices[1]]
    print("Maximum value:", data[max_indices])

    return f_z_target_pos, f_z_crosstalk_pos

def get_extend( data, axes:list, extend_num = 50 ):
    extended_data = np.pad(data, extend_num, mode='constant', constant_values=0)
    ext_axes = []
    for a in axes:
        ax_len = a.shape[-1]
        d = a[1]-a[0]
        total_point = ax_len+2*extend_num
        ext_axes.append( np.linspace( a[0]-extend_num*d, a[-1]+extend_num*d, total_point) )
    print(extended_data,extended_data.shape)

    return extended_data, ext_axes


def get_interp( data, axes:list, extend_num = 50 ):
    from scipy.interpolate import interp2d
    # Create an interpolation function
    interp_func = interp2d(axes[1], axes[0], data, kind='linear')
    # Define new points for interpolation
    ext_axes = []
    for a in axes:
        ax_len = a.shape[-1]
        d = a[1]-a[0]
        total_point = ax_len+2*extend_num
        ext_axes.append( np.linspace( a[0], a[-1], total_point) )
    # Interpolate values at new points
    extended_data = interp_func(ext_axes[1], ext_axes[0] )
    print(extended_data,extended_data.shape)

    return extended_data, ext_axes

def get_fft_mag( data ):
    # Compute the 2D Fourier Transform
    fft_result = np.fft.fft2(data)
    # Shift zero frequency components to the center
    fft_result_shifted = np.fft.fftshift(fft_result)
    # Compute the magnitude spectrum (absolute values)
    magnitude_spectrum = np.abs(fft_result_shifted)
    return magnitude_spectrum

def analysis_crosstalk_value_fft(dataset):
    """
    z1 is shape (N,),  crosstalk voltage (other)\n
    z2 is shape (M,), compensation voltage (self)\n
    data with shape (N,M)
    """
    z1 = dataset.coords["crosstalk_z"].values
    z2 = dataset.coords["detector_z"].values
    data = dataset[0, :, :]
    offset = np.mean(data)
    data -= offset

    data, axes = get_extend(data, [z1, z2], 1000)
    # data, axes = get_interp(data, [d_z_target_amp, d_z_crosstalk_amp], 100)
    # print(axes[0].shape, axes[1].shape, data.shape)
    f_axes = get_freq_axes( axes )
    magnitude_spectrum = get_fft_mag(data)
    #  get_weighted_pos(magnitude_spectrum, f_axes )
    f_z_crosstalk_pos, f_z_target_pos = get_max_pos(magnitude_spectrum, f_axes)
    print(f"f_z_target: {f_z_crosstalk_pos}")
    print(f"f_z_crosstalk: {f_z_target_pos}")
    z_slope = -f_z_target_pos/f_z_crosstalk_pos
    crosstalk = -1/z_slope
    print(f"k space: {z_slope}")
    print(f"crosstalk: {crosstalk}")

    return crosstalk, f_axes, magnitude_spectrum

def linear_fit(x, m, b):
    return m * x + b

def analysis_crosstalk_value_fitting(dataset):
    """
    z1 is shape (N,),  crosstalk voltage (other)
    z2 is shape (M,), compensation voltage (self)
    data with shape (N,M)
    """
    z1 = dataset.coords["crosstalk_z"].values[:]
    z2 = dataset.coords["detector_z"].values[:]
    data = dataset[0, :, :].values.T
    offset = np.mean(data)
    data -= offset

    

    try:
        x_vals = []
        y_vals = []
        # 对每一条 crosstalk voltage 的数据进行处理
        for i in range(data.shape[1]):  # 迭代每一列 (对应不同的 crosstalk voltage)
            col_data = data[:, i]

            fit = Fit()
            res = fit.transmission_resonator_spectroscopy(z2 * 1e12, col_data, plot=False)
            y_vals.append(res["f"][0] * 1e-12)  # 取出对应的 detector voltage

            x_vals.append(z1[i])  # 取出当前的 crosstalk voltage

    except Exception as e1:
        print(f"Failed to fit by cutting crosstalk axis: {str(e1)}")
        print("Attempting to fit by cutting detector axis...")

        x_vals = []
        y_vals = []

        try:
            # 对每一条 detector voltage 的数据进行处理
            for i in range(data.shape[0]):  # 迭代每一行 (对应不同的 detector voltage)
                row_data = data[i, :]

                fit = Fit()
                res = fit.transmission_resonator_spectroscopy(z1 * 1e12, row_data, plot=False)
                x_vals.append(res["f"][0] * 1e-12)  # 取出对应的 crosstalk voltage

                y_vals.append(z2[i])  # 取出当前的 detector voltage

        except Exception as e2:
            print(f"Failed to fit by cutting detector axis: {str(e2)}")
            print("Both fitting attempts failed.")
            return None, None, None, None

    # 执行线性拟合
    try:
        popt, pcov = curve_fit(linear_fit, x_vals, y_vals)
        slope, intercept = popt

        print(f"Fitted Line: y = {slope} * x + {intercept}")

        # 返回拟合线的参数和数据点以便后续使用
        return slope, intercept, x_vals, y_vals

    except Exception as e3:
        print(f"Linear fitting failed: {str(e3)}")
        return None, None, None, None

def analysis_crosstalk_ellipse(dataset_q):
    from scipy.ndimage import gaussian_filter
    from skimage import feature, measure
    # Apply Gaussian filter
    heatmap = gaussian_filter(dataset_q, sigma=2)

    # Normalize the heatmap data
    heatmap_min = np.min(heatmap)
    heatmap_max = np.max(heatmap)
    heatmap_normalized = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)

    # Perform Canny edge detection
    edges = feature.canny(heatmap_normalized, sigma=2)

    # Get edge coordinates
    edge_y, edge_x = np.nonzero(edges)
    edge_coords = np.column_stack([edge_x, edge_y])

    # Fit an ellipse to the edge points
    ellipse = measure.EllipseModel()
    success = ellipse.estimate(edge_coords)

    if not success:
        print("Ellipse fitting failed")
        return None, None

    # Extract ellipse parameters
    xc, yc, a, b, theta = ellipse.params

    return edge_coords, (xc, yc, a, b, theta)
