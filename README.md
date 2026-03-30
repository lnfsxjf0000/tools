# tools

## get_mask.py

Convert LabelMe JSON annotations into grayscale mask images.

### Install

```powershell
pip install numpy opencv-python
```

### Usage

Run in the dataset root directory:

```powershell
python .\get_mask.py
```

Specify a custom root directory:

```powershell
python .\get_mask.py --root .\your_dataset
```

Stop immediately if any file fails:

```powershell
python .\get_mask.py --strict
```

### Label Map

Use an external label map JSON file:

```powershell
python .\get_mask.py --label-map-file .\label_map.example.json
```

Example file:

```json
{
  "impurity|notSure": 244,
  "graphite-edgeCrinkle": 180
}
```

### Output Rule

If a JSON file is inside a directory containing `label`, that path segment will be replaced with `mask`.

Example:

```text
dataset/label/a/b/sample.json
-> dataset/mask/a/b/sample_mask.png
```

If the path does not contain `label`, the script writes output into a `mask` subdirectory next to the JSON file.
