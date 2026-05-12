# Daily Predictions of F10.7 and F30 Solar Indices with Deep Learning

## Author
Zhenduo Wang, Yasser Abduallah, Jason T. L. Wang

**Affiliation:** Institute for Space Weather Sciences, New Jersey
Institute of Technology

## Abstract
The F10.7 and F30 solar indices are the solar radio fluxes measured at wavelengths of 10.7 cm and 30 cm, respectively, which are key indicators of solar activity. F10.7 is valuable for explaining the impact of solar ultraviolet (UV) radiation on the upper atmosphere of Earth, while F30 is more sensitive and could improve the reaction of thermospheric density to solar stimulation. In this study, we present a new deep learning model, named the Solar Index Network, or SINet for short, to predict daily values of the F10.7 and F30 solar indices. The SINet model is designed to make medium-term predictions of the index values (1-60 days in advance). The observed data used for SINet training were taken from the National Oceanic and Atmospheric Administration (NOAA) as well as Toyokawa and Nobeyama facilities. Our experimental results show that SINet performs better than five closely related statistical and deep learning methods for the prediction of F10.7. Furthermore, to our knowledge, this is the first time deep learning has been used to predict the F30 solar index.

## Project Structure

- `data/`: Directory containing test datasets.
- `layers/`: Neural network layers and helper modules.
- `model/`: Directory containing pretrained models.
- `utils.py`: Model architecture and utility functions.
- `F107_test.py`: Testing script for F10.7 prediction.
- `F30_test.py`: Testing script for F30 prediction.

## Requirements

Dependencies:

```txt
python==3.9.21
numpy>=1.21
pandas>=1.3
matplotlib>=3.4
scikit-learn>=0.24
torch>=1.12
```

## Testing

### F10.7 Prediction

Run testing:

```bash
python F107_test.py
```

### F30 Prediction

Run testing:

```bash
python F30_test.py
```

## Pretrained Models

Pretrained models are stored under:

```text
model/
```

The code for loading pretrained models is included in the testing scripts.


## Reference

- Wang, Z., Abdallah, Y., Wang, J. T. L., Wang, H., Xu, Y., Yurchyshyn, V., Oria, V., Alobaid, K. A., & Bai, X. (2026). *Daily Predictions of F10.7 and F30 Solar Indices With Deep Learning*. Journal of Geophysical Research: Space Physics.https://doi.org/10.1029/2025JA034868