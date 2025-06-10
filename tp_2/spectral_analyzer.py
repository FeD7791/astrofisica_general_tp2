import pathlib

from astropy.io import fits
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from astropy.modeling import models, fitting

# Specutils
from specutils import Spectrum1D
from specutils.analysis import equivalent_width
from specutils.spectra import SpectralRegion
from specutils.manipulation import extract_region
import numpy as np
import uttr
import astropy.units as u



@uttr.s
class Spectrum:
    wave = uttr.ib(default=None, unit=u.AA, converter=lambda x: np.asarray(x, dtype=float))
    flux = uttr.ib(default=None, unit=(u.erg / (u.cm**2 * u.s * u.AA)), converter=lambda x: np.asarray(x, dtype=float))
    calibration_factor = uttr.ib(default=1e-19)
    _spectrum_1d = uttr.ib(init=False)

    @property
    def plot(self):
        fig, ax = plt.subplots(1,1, figsize=(7,7))
        ax.plot(self.arr_.wave, self.arr_.flux)
        ax.set_xlabel(r"Wavelength $(\AA)$")
        ax.set_ylabel(r"Intensity $\frac{\mathrm{erg}}{\mathrm{cm}^{2} \, \mathrm{s} \, \AA}$")
        ax.grid(True)
        return ax

    @classmethod
    def _create_new_spectrum(cls, params):
        return cls(
            wave = params['wave'],
            flux = params['flux'],
        )

    
    def __attrs_post_init__(self):
        """Clean spectra when generated"""
        index = np.isfinite(self.arr_.wave) & np.isfinite(self.arr_.flux)
        self.wave = self.wave[index]
        self.flux = self.flux[index]*self.calibration_factor

        # Build _spectrum_1d
        self._spectrum_1d = Spectrum1D(spectral_axis=self.wave, flux=self.flux)

    
    def transform_to_restframe(self, redshift):
        wave_shifted = self.wave / (1 + redshift)
        params = {'wave':wave_shifted, 'flux':self.flux}
        return self._create_new_spectrum(params=params)
    
    def get_equivalent_width(self, min_wave, max_wave):
        region = SpectralRegion(min_wave* self.wave.unit, max_wave * self.wave.unit)
        ew = equivalent_width(self._spectrum_1d, regions=region)
        return ew
    

    
    def cont_flux(self, center, window=10, min=5, max=20):
        linea_min_ = center - window
        linea_max_ = center + window

        continuo_izq_min_ = linea_min_ - max
        continuo_izq_max_ = linea_min_ - min
        continuo_der_min_ = linea_max_ + min
        continuo_der_max_ = linea_max_ + max

        mask_linea_ = (self.wave >= linea_min_) & (self.wave <= linea_max_)
        mask_cont_izq_ = (self.wave >= continuo_izq_min_) & (self.wave <= continuo_izq_max_)
        mask_cont_der_ = (self.wave >= continuo_der_min_) & (self.wave <= continuo_der_max_)

        cont_izq_ = np.mean(self.flux[mask_cont_izq_])
        cont_der_ = np.mean(self.flux[mask_cont_der_])
        cont_prom_ = (cont_izq_ + cont_der_) / 2

        flujo_linea_ = self.flux[mask_linea_] - cont_prom_
        flujo_integrado_ = np.trapezoid(flujo_linea_, self.wave[mask_linea_])
        return flujo_integrado_
    
    def plot_line(self, center, window=10):
        fig, ax = plt.subplots(1,1, figsize=(7,7))
        ax.plot(self.arr_.wave, self.arr_.flux)
        ax.set_xlim(center - 2*window, center + 2*window)
        ax.axvline(center, color='red', linestyle='--', label=f"{center} Å")
        ax.axvspan(center - window, center + window, color='gray', alpha=0.3, label=f'Región [{center-window},{center+window}]')
        ax.set_xlabel(r"Wavelength $(\AA)$")
        ax.set_ylabel(r"Intensity $\frac{\mathrm{erg}}{\mathrm{cm}^{2} \, \mathrm{s} \, \AA}$")
        ax.grid(True)
        return ax
    

    def medir_fwhm_gaussiana(self, centro, ventana=20):
        """
        Mide el FWHM de una línea espectral centrada en `centro` (Å)
        en un espectro dado por longitud (Å) y flujo (cualquier unidad).
        Ajusta una gaussiana local en ventana ±ventana Å alrededor del centro.
        Retorna FWHM en Å y km/s, y parámetros del ajuste.
        """
        c = 299792.458  # km/s velocidad de la luz

        # Seleccionar datos en la ventana
        idx = (self.wave > centro - ventana) & (self.wave < centro + ventana)
        x = self.wave[idx]
        y = self.flux[idx]

        if len(x) == 0:
            raise ValueError(f"No hay datos en la ventana para la línea en {centro} Å.")

        # Estimar y sustraer continuo local (mediana)
        continuo = np.median(y)
        y_corr = y - continuo

        # Filtrar valores no finitos (nan, inf)
        mask = np.isfinite(x) & np.isfinite(y_corr)
        x = x[mask]
        y_corr = y_corr[mask]

        if len(x) == 0:
            raise ValueError(f"No quedan datos finitos para ajustar la línea en {centro} Å.")

        # Ajuste gaussiano inicial
        g_init = models.Gaussian1D(amplitude=np.max(y_corr), mean=centro, stddev=2.0)
        fitter = fitting.LevMarLSQFitter()
        g_fit = fitter(g_init, x, y_corr)

        fwhm_aa = 2.3548 * g_fit.stddev.value  # FWHM en Angstroms
        fwhm_kms = (fwhm_aa / centro) * c      # FWHM en km/s

        return {
            "fwhm_aa": fwhm_aa,
            "fwhm_kms": fwhm_kms,
            "amplitud": g_fit.amplitude.value,
            "mean": g_fit.mean.value,
            "stddev": g_fit.stddev.value,
            "continuo": continuo
        }