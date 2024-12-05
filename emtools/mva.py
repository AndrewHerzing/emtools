# -*- coding: utf-8 -*-
#
# This file is part of EMTools

"""
MVA module for EMTools package.

@author: Andrew Herzing
"""

import numpy as np
import tqdm
from scipy import optimize
import hyperspy.api as hs
import exspy as ex
import matplotlib.pylab as plt
from sklearn.decomposition import PCA
from scipy.linalg import svd
from pymcr.mcr import McrAR
from pymcr.regressors import NNLS
from hyperspy.signal import _plot_x_results as plt_results
from emtools.color import merge_color_channels


def plot_rgb_overlay(
    mva, idx, element_list, figsize=(6, 2), y_text_step=15, y_text_start=3, x_offset=2
):
    """
    Plot an RGB overlay image with text annotations indicating the elements contributing
    to each color channel.

    Parameters:
    ----------
    mva : object
        An object containing MCR-ALS loadings. It is expected to have an attribute
        `loadingsMCR` that provides access to the spectral loadings for different indices.

    idx : list of int
        List of indices corresponding to the color channels to be merged into the RGB image.

    element_list : list of list of str
        A list where each entry is a list of strings describing the elements contributing to
        a given color channel.

    figsize : tuple of float, optional, default=(6, 2)
        The size of the figure (width, height) in inches.

    y_text_step : int, optional, default=15
        Vertical spacing between consecutive text annotations.

    y_text_start : int, optional, default=3
        Initial vertical position for the first text annotation.

    x_offset : int, optional, default=2
        Horizontal offset for the text annotations, determining their distance from the image.

    Returns:
    -------
    None
        Displays the RGB overlay plot with text annotations.

    """

    rgb = merge_color_channels([mva.loadingsMCR.inav[i].data for i in idx])
    elements = [element_list[i] for i in idx]
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgb)
    colors = ["r", "g", "b", "c", "y", "m", "gray"]
    x_text = rgb.shape[1] + x_offset  # Position text to the right of the image
    y_text = y_text_start
    for i, (region, color) in enumerate(zip(elements, colors)):
        ax.text(
            x_text,
            y_text,
            ", ".join(region),
            color=color,
            fontsize=12,
            ha="left",
            va="center",
        )
        y_text = y_text + y_text_step
    plt.tight_layout()
    return


class MVA:
    def __init__(self, si):
        self.axes = si.axes_manager.as_dictionary()
        self.si = si.deepcopy()
        self.si.unfold()
        self.si = self.si.swap_axes(0, 1)
        if self.si.data.min() == 0.0:
            self.si.data += 1e-6
            self.si_offset = 1e-6
        else:
            self.si_offset = 0.0
        self.nrows, self.ncols, self.nchannels = si.data.shape
        self.wt_im = None
        self.wt_spec = None

    def make_positive(self):
        """
        Ensure that the rotated factors and loadings have positive overall contributions.

        This method iterates through each component in the rotated factors (`self.factorsRotated`)
        and loadings (`self.loadingsRotated`). If the sum of a factor and its corresponding
        loading is negative (i.e., their contributions are primarily negative), then the sign of both the factor
        and loading are flipped to make their overall contributions positive.

        Parameters:
        ----------
        None

        Returns:
        -------
        None
            The method modifies `self.factorsRotated` and `self.loadingsRotated` in place.

        Notes:
        -----
        - The positivity check is based on the condition that the sum of elements in a factor
          and its corresponding loading must be greater than or equal to 1.
        """

        for i in range(0, self.factorsRotated.shape[1]):
            if (self.factorsRotated[:, i].sum() < 1) and (
                self.loadingsRotated[i, :].sum() < 1
            ):
                self.factorsRotated[:, i] *= -1
                self.loadingsRotated[i, :] *= -1
        return

    def rotate(self, domain="spatial", gamma=1.0, max_iters=20, tol=1e-6):
        """
        Perform an oblique rotation on PCA components to improve interpretability.

        The method applies a general rotation algorithm to PCA-derived factors and loadings,
        aiming to produce a rotated solution that balances simplicity and data fidelity.
        The rotation can be applied in either the spatial or spectral domain.

        Parameters:
        ----------
        domain : str, optional, default='spatial'
            Specifies the domain in which the rotation is applied:
            - 'spatial': Rotate the loadings while adjusting factors accordingly.
            - 'spectral': Rotate the factors while adjusting loadings accordingly.

        gamma : float, optional, default=1.0
            The weight controlling the simplicity constraint. Larger values favor simplicity
            in the rotated solution.  A value of 1.0 performs a varimax rotation.

        max_iters : int, optional, default=20
            Maximum number of iterations for the rotation algorithm.

        tol : float, optional, default=1e-6
            Convergence tolerance. The algorithm stops when the change in the
            rotation matrix is less than `tol`.

        Returns:
        -------
        None
            Updates the following instance attributes in place:
            - `self.rotation_matrix`: The computed rotation matrix.
            - `self.factorsRotated`: Rotated factors.
            - `self.loadingsRotated`: Rotated loadings.

        Raises:
        -------
        ValueError
            If `domain` is not `'spatial'` or `'spectral'`.

        Notes:
        -----
        - Uses singular value decomposition (SVD) to iteratively refine the rotation matrix.
        - The algorithm minimizes the loss function while enforcing a spatial or spectral simplicity constraint.
        """

        if domain == "spatial":
            Phi = self.loadingsPCA.T
        elif domain == "spectral":
            Phi = self.factorsPCA
        else:
            raise ValueError("Unknown domain.  Must be 'spatial' or 'spectral'")
        p, k = Phi.shape
        R = np.eye(k)
        for i in range(max_iters):
            Lambda = np.dot(Phi, R)
            u, s, vh = svd(
                np.dot(
                    Phi.T,
                    Lambda**3
                    - (gamma / p) * np.dot(Lambda, np.diag(np.sum(Lambda**2, axis=0))),
                )
            )
            R_new = np.dot(u, vh)
            if np.sum(np.abs(R_new - R)) < tol:
                break
            R = R_new
        self.rotation_matrix = R

        if domain == "spatial":
            self.loadingsRotated = np.dot(self.loadingsPCA.T, R).T
            self.factorsRotated = np.dot(self.factorsPCA, R)
        elif domain == "spectral":
            self.factorsRotated = np.dot(R, self.factorsPCA)
            self.loadingsRotated = np.dot(R, self.loadingsPCA.T).T
        return

    def weight_si(self):
        """
        Apply Poissonian weighting to the spectral and spatial components of the dataset.

        The weights are derived as the inverse square root of the mean values along the
        respective dimensions, with non-finite values (e.g., NaN or infinity) set to zero.

        Parameters:
        ----------
        None

        Returns:
        -------
        None
            Updates the following instance attributes in place:
            - `self.weighted_si`: The weighted dataset, with weights applied to both spectral
              and spatial components.
            - `self.wt_im`: The weights applied to the spatial (image) dimension.
            - `self.wt_spec`: The weights applied to the spectral dimension.

        Notes:
        -----
        - The method assumes the dataset is stored in `self.si.data`, which is expected to be
          a NumPy array with shape `(nchannels, npixels)`.
        """
        wt_spec = 1 / np.sqrt(self.si.data.mean(1))
        wt_spec[~np.isfinite(wt_spec)] = 0

        wt_im = 1 / np.sqrt(self.si.data.mean(0))
        wt_im[~np.isfinite(wt_im)] = 0

        self.weighted_si = self.si.deepcopy()
        self.weighted_si = self.si.data * wt_im[np.newaxis, :] * wt_spec[:, np.newaxis]
        self.wt_im = wt_im
        self.wt_spec = wt_spec
        return

    def svd(self, ncomps=32):
        """
        Perform Singular Value Decomposition (SVD) on the weighted dataset.

        This method decomposes the weighted dataset into singular vectors and singular values,
        capturing the dominant components up to the specified number of components (`ncomps`).
        The results include factors (left singular vectors, U), loadings (right singular vectors, V),
        and eigenvalues (singular values, S).

        Parameters:
        ----------
        ncomps : int, optional, default=32
            The number of components to retain from the SVD decomposition.

        Returns:
        -------
        None
            Updates the following instance attributes in place:
            - `self.factorsSVD`: The left singular vectors (factors) corresponding to the first
              `ncomps` components.
            - `self.loadingsSVD`: The right singular vectors (loadings) corresponding to the first
              `ncomps` components.
            - `self.eigenvalues`: The singular values corresponding to the first `ncomps` components.
            - `self.svd_comps`: Stores the number of components used in the decomposition.

        Notes:
        -----
        - The method operates on the weighted dataset, which is assumed to be stored in
          `self.weighted_si.data`.
        - The decomposition is performed using `np.linalg.svd` with `full_matrices=False` to
          ensure computational efficiency.
        - The singular values (eigenvalues) represent the variance explained by each component.
        """
        self.svd_comps = ncomps
        U, S, V = np.linalg.svd(self.weighted_si.data, full_matrices=False)
        self.factorsSVD = U[:, 0:ncomps]
        self.loadingsSVD = V[0:ncomps, :]
        self.eigenvalues = S[0:ncomps]
        return

    def plot_eigenvalues(self):
        """
        Plot the eigenvalues of an SVD decomposition.

        The eigenvalue plot is useful for determining the number of components to use for refining
        the decomposition.

        Parameters:
        ----------
        None

        Returns:
        -------
        None

        Notes:
        -----
        - The method assumes that an SVD decomposition has already been performed.
        """
        plt.figure()
        plt.semilogy(self.eigenvalues**2, "b.")
        return

    def pca(self, ncomps=None):
        """
        Perform Principal Component Analysis (PCA) on the weighted dataset.

        This method applies PCA to the weighted dataset to extract the specified number of
        components (`ncomps`). The resulting factors (spatial components) and loadings (spectral
        components).

        Parameters:
        ----------
        ncomps : int or None, optional, default=None
            The number of principal components to retain. If `None`, all components are retained.

        Returns:
        -------
        None
            Updates the following instance attributes in place:
            - `self.factorsPCA`: The spatial factors (PCA-transformed data).
            - `self.loadingsPCA`: The spectral loadings (PCA components).
            - `self.ncomps`: Stores the number of components used in the PCA decomposition.
            - `self.factorsSVD`: Truncated to the first `ncomps` components (if previously calculated).
            - `self.loadingsSVD`: Truncated to the first `ncomps` components (if previously calculated).

        Notes:
        -----
        - The method operates on the weighted dataset, assumed to be stored in
          `self.weighted_si.data`.
        - The PCA computation is performed using `sklearn.decomposition.PCA`.
        - If `ncomps` is not specified, the method retains all available components.
        - The existing SVD results (`self.factorsSVD` and `self.loadingsSVD`) are truncated
          to match the number of PCA components.
        """
        pca = PCA(n_components=ncomps)
        self.factorsPCA = pca.fit_transform(
            self.weighted_si.data
        )  # Initial spatial factors
        self.loadingsPCA = pca.components_  # Initial spectral loadings
        self.ncomps = ncomps

        self.factorsSVD = self.factorsSVD[:, 0:ncomps]
        self.loadingsSVD = self.loadingsSVD[0:ncomps, :]
        return

    def mcrals(self, domain="spatial"):
        """
        Perform Multivariate Curve Resolution with Alternating Least Squares (MCR-ALS) on the weighted dataset.

        This method applies the MCR-ALS algorithm to decompose the weighted dataset into
        factors and loadings, optimizing the solution based on the specified domain ('spatial' or 'spectral').
        The algorithm uses non-negative least squares (NNLS) regression to fit the data and enforce non-negativity constraints.

        Parameters:
        ----------
        domain : str, optional, default='spatial'
            Specifies the domain in which the MCR-ALS algorithm is applied:
            - `'spatial'`: Optimize the spatial factors (rows of the matrix).
            - `'spectral'`: Optimize the spectral loadings (columns of the matrix).

        Returns:
        -------
        None
            Updates the following instance attributes in place:
            - `self.factorsMCR`: The optimized spatial factors (MCR-ALS result).
            - `self.loadingsMCR`: The optimized spectral loadings (MCR-ALS result).

        Notes:
        -----
        - The method uses `McrAR` from a specified implementation of the MCR-ALS algorithm, with
          non-negative least squares (NNLS) as the regression model.
        - Depending on the selected `domain`, the method initializes either spatial factors or spectral loadings,
          and applies MCR-ALS to fit the weighted dataset.
        - The method assumes that the weighted dataset is stored in `self.weighted_si`, and that the
          rotated factors and loadings are available in `self.factorsRotated` and `self.loadingsRotated`.
        - The optimization process ensures that the resulting factors and loadings are non-negative.
        """
        mcrar = McrAR(
            c_regr=NNLS(),
            st_regr=NNLS(),
            c_constraints=[],
            st_constraints=[],
            tol_increase=0.5,
            tol_err_change=1e-4,
        )
        if domain == "spatial":
            ST_init = np.clip(self.factorsRotated.T, 0, None)
            mcrar.fit(self.weighted_si.T, ST=ST_init, verbose=True)
        elif domain == "spectral":
            C_init = np.clip(self.loadingsRotated.T, 0, None)
            mcrar.fit(self.weighted_si.T, C=C_init, verbose=True)
        self.factorsMCR = mcrar.ST_opt_.T
        self.loadingsMCR = mcrar.C_opt_.T
        return

    def unweight(self):
        """
        Undo the Poisson weighting applied to the factors and loadings.

        This method reverses the weighting applied to the spatial and spectral components of
        the dataset in preprocessing. It restores the original scale of the factors and loadings by dividing
        the factors by the spectral weights and the loadings by the image weights.

        Parameters:
        ----------
        None

        Returns:
        -------
        None
            Updates the following instance attributes in place:
            - `self.factorsMCR`: The unweighted spatial factors.
            - `self.loadingsMCR`: The unweighted spectral loadings.

        Notes:
        -----
        - The method assumes that the weighted factors and loadings are stored in
          `self.factorsMCR` and `self.loadingsMCR`, and that the weights for the spectral and
          spatial dimensions are stored in `self.wt_spec` and `self.wt_im`, respectively.
        - The unweighting process involves dividing each factor by the corresponding spectral weight
          and each loading by the corresponding spatial weight to restore the data to its original scale.
        """
        self.factorsMCR = (self.factorsMCR.T / self.wt_spec).T
        self.loadingsMCR = self.loadingsMCR / self.wt_im
        return

    def sum_to_one(self):
        """
        Normalize the factors and loadings so that the sum of the factors equals 1.

        This method normalizes the spectral factors by dividing them by their sum along each
        component. It also adjusts the spatial loadings by multiplying each loading by the
        corresponding factor sum to maintain consistency in the decomposition.

        Parameters:
        ----------
        None

        Returns:
        -------
        None
            Updates the following instance attributes in place:
            - `self.factors`: The normalized factors, where the sum of each factor is 1.
            - `self.loadings`: The adjusted loadings to maintain consistency with the factors.
            - `self.factor_sum`: The sum of the spatial factors before normalization.

        Notes:
        -----
        - The method operates on the `self.factors` and `self.loadings` attributes
        - The normalization ensures that the factors sum to 1 along each component, and adjusts the
          loadings to preserve the relationship between the factors and loadings.
        """
        factor_sum = self.factors.sum(0)
        self.factors = self.factors / factor_sum
        self.loadings = ((self.loadings).T * factor_sum).T
        self.factor_sum = factor_sum
        return

    def undo_sum_to_one(self):
        """
        Reverse the normalization applied by the `sum_to_one` method.

        This method restores the factors and loadings to their original
        scales by undoing the normalization.

        Parameters:
        ----------
        None

        Returns:
        -------
        None
            Updates the following instance attributes in place:
            - `self.factors`: The factors restored to their original scale.
            - `self.loadings`: The loadings restored to their original scale.

        Notes:
        -----
        - The method assumes that the factor sums from the normalization process are stored in
          `self.factor_sum`.
        - This method is the inverse operation of `sum_to_one`, and should be used to restore
          the factors and loadings to their prior state.
        """
        self.factors = self.factors * self.factor_sum
        self.loadings = ((self.loadings).T / self.factor_sum).T
        return

    def convert_to_hspy(self):
        """
        Convert the factors and loadings to Hyperspy Signal objects.

        This method converts the factors and loadings from SVD, PCA, rotation, and MCR decomposition
        into appropriate Hyperspy Signal objects. The resulting signals are assigned to the
        instance attributes `self.factorsSVD`, `self.loadingsSVD`, `self.factorsPCA`, `self.loadingsPCA`,
        `self.factorsRotated`, `self.loadingsRotated`, `self.factorsMCR`, and `self.loadingsMCR`.

        The factors are converted to `EDSTEMSpectrum` objects, while the loadings are converted
        to `Signal2D` objects, each with properly defined axes.

        Parameters:
        ----------
        None

        Returns:
        -------
        None
            Updates the following instance attributes in place:
            - `self.factorsSVD`: Converted SVD factors as an `EDSTEMSpectrum` signal.
            - `self.loadingsSVD`: Converted SVD loadings as a `Signal2D` signal.
            - `self.factorsPCA`: Converted PCA factors as an `EDSTEMSpectrum` signal.
            - `self.loadingsPCA`: Converted PCA loadings as a `Signal2D` signal.
            - `self.factorsRotated`: Converted rotated factors as an `EDSTEMSpectrum` signal.
            - `self.loadingsRotated`: Converted rotated loadings as a `Signal2D` signal.
            - `self.factorsMCR`: Converted MCR factors as an `EDSTEMSpectrum` signal.
            - `self.loadingsMCR`: Converted MCR loadings as a `Signal2D` signal.

        Notes:
        -----
        - The method assumes that the factors and loadings for SVD, PCA, rotated, and MCR are available
          in the corresponding attributes: `self.factorsSVD`, `self.loadingsSVD`, `self.factorsPCA`,
          `self.loadingsPCA`, `self.factorsRotated`, `self.loadingsRotated`, `self.factorsMCR`, and `self.loadingsMCR`.
        - The `comp_axis` defines the axis for the decomposition components and is shared across all signals.
        - The method uses the `ex.signals.EDSTEMSpectrum` from eXSpy and `hs.signals.Signal2D` from HyperSpy to convert the data.
        """
        comp_axis = {
            "_type": "UniformDataAxis",
            "name": "Decomposition component index",
            "units": None,
            "navigate": True,
            "is_binned": False,
            "size": self.ncomps,
            "scale": 1.0,
            "offset": 0.0,
        }
        self.factorsSVD = ex.signals.EDSTEMSpectrum(
            self.factorsSVD.T, axes=[comp_axis, self.axes["axis-2"]]
        )
        self.loadingsSVD = hs.signals.Signal2D(
            self.loadingsSVD.reshape([self.ncomps, self.nrows, self.ncols]),
            axes=[comp_axis, self.axes["axis-0"], self.axes["axis-1"]],
        )

        self.factorsPCA = ex.signals.EDSTEMSpectrum(
            self.factorsPCA.T, axes=[comp_axis, self.axes["axis-2"]]
        )
        self.loadingsPCA = hs.signals.Signal2D(
            self.loadingsPCA.reshape([self.ncomps, self.nrows, self.ncols]),
            axes=[comp_axis, self.axes["axis-0"], self.axes["axis-1"]],
        )

        self.factorsRotated = ex.signals.EDSTEMSpectrum(
            self.factorsRotated.T, axes=[comp_axis, self.axes["axis-2"]]
        )
        self.loadingsRotated = hs.signals.Signal2D(
            self.loadingsRotated.reshape([self.ncomps, self.nrows, self.ncols]),
            axes=[comp_axis, self.axes["axis-0"], self.axes["axis-1"]],
        )

        self.factorsMCR = ex.signals.EDSTEMSpectrum(
            self.factorsMCR.T, axes=[comp_axis, self.axes["axis-2"]]
        )
        self.loadingsMCR = hs.signals.Signal2D(
            self.loadingsMCR.reshape([self.ncomps, self.nrows, self.ncols]),
            axes=[comp_axis, self.axes["axis-0"], self.axes["axis-1"]],
        )
        return

    def plot_loadings(self, comp_idx=None, colormap="afmhot", figsize=(10, 4)):
        """
        Plot the MCR loadings for the specified components.

        This method generates images for the specified components of the MCR loadings and
        displays them using HyperSpy's `plot_images` function.

        Parameters:
        ----------
        comp_idx : list or array-like, optional
            A list or array of component indices to be plotted. If `None`, plots all components
            (from 0 to `self.ncomps - 1`).
        colormap : str, optional, default='amfhot'
            Colormap for displaying the images.
        figsize : tuple, optional, default=(10,4)
            Size of the resulting image.
        Returns:
        -------
        None
            Displays the plot(s) of the specified loadings.
        """
        if comp_idx is None:
            comp_idx = np.arange(0, self.ncomps)
        _ = hs.plot.plot_images(
            [self.loadingsMCR.inav[i] for i in comp_idx],
            cmap=colormap,
            tight_layout=True,
        )
        plt.gcf().set_size_inches(figsize)
        plt.tight_layout()
        return

    def plot_factors(self, comp_idx=None):
        """
        Plot the MCR factors for the specified components.

        This method generates spectra for the specified components of the MCR factors and
        displays them using HyperSpy's `plot_spectra` function.

        Parameters:
        ----------
        comp_idx : list or array-like, optional
            A list or array of component indices to be plotted. If `None`, plots all components
            (from 0 to `self.ncomps - 1`).

        Returns:
        -------
        None
            Displays the spectra of the specified factors.
        """
        if comp_idx is None:
            comp_idx = np.arange(0, self.ncomps)
        _ = hs.plot.plot_spectra([self.factorsMCR.inav[i] for i in comp_idx])
        return

    def plot_component(self, idx, elements, lines=None, offset=0.1, colormap="afmhot"):
        """
        Plot the MCR loading and factor for the specified component, with X-ray lines labeled.

        This method generates a two-panel plot where the first panel shows the 2D data of the MCR
        loading and the second panel shows the 1D data of the corresponding factor. It also annotates
        the x-ray lines on the factor plot.

        Parameters:
        ----------
        idx : int
            The index of the component to be plotted.
        elements : list of str
            A list of element names (strings) to be used for labeling the plots.
        lines : list of str, optional
            A list of x-ray lines to annotate on the factor plot. If `None`, only the default x-ray
            lines ('a') are annotated.
        offset : float, optional, default=0.1
            The offset for positioning the x-ray line labels on the factor plot.
        colormap : str, optional, default='amfhot'
            Colormap for displaying the images.

        Returns:
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plots.
        axes : array-like of matplotlib.axes.Axes
            The axes of the subplots (one for the loading plot and one for the factor plot).

        Notes:
        -----
        - The MCR loading and factor are accessed from `self.loadingsMCR.inav[idx]` and
          `self.factorsMCR.inav[idx]`, respectively.
        """
        loading = self.loadingsMCR.inav[idx]
        factor = self.factorsMCR.inav[idx]

        factor.set_elements([])
        factor.set_lines([])
        factor.set_elements(elements)
        if lines is None:
            factor.add_lines(only_one=False, only_lines=("a",))
        else:
            factor.add_lines(lines)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Plot the Signal2D data on the first axis
        _ = axes[0].imshow(loading.data, cmap=colormap)
        axes[0].set_title("Loading #" + str(idx) + " ; " + ", ".join(elements))
        axes[0].axis("off")  # Remove axis ticks for the image

        # Plot the Signal1D data on the second axis
        axes[1].plot(factor.axes_manager[0].axis, factor.data)
        axes[1].set_title("Factor #" + str(idx) + " ; " + ", ".join(elements))
        axes[1].set_xlabel(
            factor.axes_manager[0].name + f" ({factor.axes_manager[0].units})"
        )
        axes[1].set_ylabel("Intensity")

        # Annotate x-ray lines

        for line in factor.metadata.Sample.xray_lines:
            element, line_name = line.split("_")
            energy = (
                ex.material.elements[element]
                .Atomic_properties.Xray_lines[line_name]
                .energy_keV
            )
            axes[1].axvline(energy, color="r", linestyle="--", alpha=0.6)
            axes[1].text(
                offset + energy,
                max(factor.data) * 0.9,
                line,
                rotation=90,
                color="r",
            )

        # Adjust layout and display the plot
        plt.tight_layout()
        plt.show()
        return fig, axes

    def plot_results(self):
        """
        Plot the results of the MCR decomposition interactively.

        This method utilizes HyperSpy's internal `plt_results` function to generate an interactive
        plot that visualizes the MCR factors and loadings.

        Parameters:
        ----------
        None

        Returns:
        -------
        None
            The plot is displayed directly using the `plt_results` function. No value is returned.
        """
        plt_results(self.factorsMCR, self.loadingsMCR, "smart_auto", "smart_auto", 2, 2)


def upsample_factors(si, loadings, factors, scaling, **kwargs):
    """
    Upsample calculated factors to the original resolution

    Args
    ----------
    si : Hyperspy Signal1D
        Original hyperspectral dataset before decomposition
    loadings : NumPy array
        Loadings array of shape (npixels, nfactors)
    factors : fload
        Factor array of shape (nfactors, nchannels)
    scaling : list or NumPy array
        Factors by which the original SI was compressed in order to perform
        the decomposition.  The order must be:
        [compression_rows, compression_columns, compression_spectral]

    Returns
    -----------
    C_nnls : Numpy array
        Upsampled loadings array [nfactors, nrows, ncols]
    ST_nnls : Numpy array
        Upsampled spectral factors array [nchannels, nfactors]

    """
    tol = kwargs.get("tol", None)
    maxiter = kwargs.get("maxiter", None)
    ST = factors.T
    C = loadings
    nfactors = C.shape[1]
    nrows, ncols, nchannels = si.data.shape
    npixels = nrows * ncols
    si_binned = si.rebin(scale=(1, 1, scaling[2]))
    si_binned.unfold()
    si_binned = si_binned.data + 1e-6
    C_nnls = np.zeros([nfactors, si_binned.shape[0]])
    for i in tqdm.tqdm(range(npixels)):
        C_nnls[:, i], _ = optimize.nnls(ST, si_binned[i, :], maxiter=maxiter, atol=tol)

    si_binned = si.rebin(scale=(scaling[0], scaling[1], 1))
    si_binned.unfold()
    si_binned = si_binned.data + 1e-6
    ST_nnls = np.zeros([nfactors, si_binned.shape[1]])
    for i in tqdm.tqdm(range(nchannels)):
        ST_nnls[:, i], _ = optimize.nnls(C, si_binned[:, i], maxiter=maxiter, atol=tol)
    C_nnls = C_nnls.reshape([nfactors, nrows, ncols])
    return C_nnls, ST_nnls.T
