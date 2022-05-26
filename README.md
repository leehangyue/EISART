# EISART
 Welcome to EISART! 
 
 EISART, or Electrochemical Impedance Spectra Analysis and Refining Tool, is a free software in Python for impedance spectrum analysis with GUI.
 
 Download '**contents**' for the setup & code. **After installation, run EISART.py**
 
 Download '**readme**' for instructions.
 
# Citation
 Please cite the following academic journal article if you use EISART in your work: 
 
 Li, Hangyue, Zewei Lyu, and Minfang Han. "Robust and Fast Estimation of Equivalent Circuit Model from Noisy Electrochemical Impedance Spectra." Electrochimica Acta (2022): 140474. https://doi.org/10.1016/j.electacta.2022.140474

# About
 EISART is a robust and fast impedance spectrum analysis tool originally developed for solid oxide fuel cells.
 
 It features one-click operations for:
 
 · Distribution of Relaxation Time (**DRT**) analysis
 
 · Auto or semi-auto Equivalent Circuit Model (**ECM**) fitting & ZView *.mdl model importing/exporting
 
 · Fast **batch** processing and easy-to-use saving format (*.csv, *.txt, *.png, etc.)
 
 · Direct import of Zahner Thales *.ism binary impedance files
 
 EISART is **robust to noise**, wiring induction, and deviated data points in EIS. It visualizes the original data and the fitting result as Nyquist and Bode plots, and shows the fitting residuals to the user in around one second.
 
 With respect to the developers of DRTtools (https://sites.google.com/site/drttools/home), the basic (no data screening and weighting) DRT evaluation algorithm yields nearly identical results as DRTtools (v4, Feb 8 2021, MATLAB code), while EISART offers better robustness against noise and auto/semi-auto ECM fitting with default settings.

 For more details of EISART, please refer to the user manual files (pdf) in the folder 'readme'.
