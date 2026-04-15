import camb

pars = camb.CAMBparams()

# 6-parameter LCDM: background part
pars.set_cosmology(
    H0=67.36,
    ombh2=0.02237,
    omch2=0.1200,
    tau=0.0544,
    mnu=0.06,   # if you want Planck baseline-like minimal neutrino mass
)

# primordial spectrum part
pars.InitPower.set_params(
    As=2.100e-9,
    ns=0.9649,
)

results = camb.get_results(pars)
cls = results.get_cmb_power_spectra(pars, CMB_unit='muK')
cltt = cls['total'][:, 0]