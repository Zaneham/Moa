# Nuclear Data Download

ENDF/B-VIII.0 evaluated nuclear data files from the IAEA mirror.
Plain text, 80-column fixed-width format. No conversion needed.

## Source

Individual isotope zips from:
https://www-nds.iaea.org/public/download-endf/ENDF-B-VIII.0/n/

## Required Files

### Godiva + Flattop-25 (uranium benchmarks)

```bash
cd data/
curl -LO https://www-nds.iaea.org/public/download-endf/ENDF-B-VIII.0/n/n_9225_92-U-234.zip
curl -LO https://www-nds.iaea.org/public/download-endf/ENDF-B-VIII.0/n/n_9228_92-U-235.zip
curl -LO https://www-nds.iaea.org/public/download-endf/ENDF-B-VIII.0/n/n_9237_92-U-238.zip

unzip n_9225_92-U-234.zip
unzip n_9228_92-U-235.zip
unzip n_9237_92-U-238.zip
```

### Jezebel + Jezebel-240 (plutonium benchmarks)

```bash
curl -LO https://www-nds.iaea.org/public/download-endf/ENDF-B-VIII.0/n/n_9437_94-Pu-239.zip
curl -LO https://www-nds.iaea.org/public/download-endf/ENDF-B-VIII.0/n/n_9440_94-Pu-240.zip
curl -LO https://www-nds.iaea.org/public/download-endf/ENDF-B-VIII.0/n/n_9443_94-Pu-241.zip
curl -LO https://www-nds.iaea.org/public/download-endf/ENDF-B-VIII.0/n/n_9446_94-Pu-242.zip
curl -LO https://www-nds.iaea.org/public/download-endf/ENDF-B-VIII.0/n/n_3125_31-Ga-69.zip

unzip n_9437_94-Pu-239.zip
unzip n_9440_94-Pu-240.zip
unzip n_9443_94-Pu-241.zip
unzip n_9446_94-Pu-242.zip
unzip n_3125_31-Ga-69.zip
```

## After Download

Rename (or symlink) the extracted `.endf` files to match what the
benchmark inputs expect. The zip files contain files named like
`n-092_U_235.endf` — the benchmark `.inp` files reference them
by the zip naming convention.

## File Sizes

| Isotope | ZIP    | ENDF   |
|---------|--------|--------|
| U-234   | 511 KB | ~2 MB  |
| U-235   | 13 MB  | ~40 MB |
| U-238   | 4.4 MB | ~15 MB |
| Pu-239  | 5.1 MB | ~17 MB |
| Pu-242  | 680 KB | ~2 MB  |

## Alternative: NNDC Sigma

Browse and download individual isotopes interactively:
https://www.nndc.bnl.gov/sigma/

## Library Version

ENDF/B-VIII.0 (2018) or VIII.1 (2024) both work. The CIELO
evaluations (U-235, U-238, Pu-239) are the gold standard for
fast criticality benchmarks.
