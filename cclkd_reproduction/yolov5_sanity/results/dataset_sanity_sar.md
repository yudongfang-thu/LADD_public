# Dataset sanity: sar

- yaml: `shared/configs/datasets_public/ogsod1_sar_detect.yaml`
- root: `/path/to/OGSOD-1.0/sar`
- status: **warning**
- nc: 3
- names: bridge, harbor, storage_tank
- label files: 0

## Split Counts

| split | images | entries |
|---|---:|---|
| train | 0 | `['images/train']` |
| val | 0 | `['images/test']` |
| test | 0 | `['images/test']` |

## Instance Counts

| class id/name | instances |
|---|---:|

## Paper Table Class Mapping

- Oil Tank -> `storage_tank`
- Bridge -> `bridge`
- Harbor -> `harbor`

## Warnings

- Class order differs from the CCLKD/shared table order Oil Tank / Bridge / Harbor. YAML order is: ['bridge', 'harbor', 'storage_tank']. Use mapping Oil Tank -> storage_tank, Bridge -> bridge, Harbor -> harbor when reporting per-class AP.
- Image split path does not exist: /path/to/OGSOD-1.0/sar/images/train
- Image split path does not exist: /path/to/OGSOD-1.0/sar/images/test
- Image split path does not exist: /path/to/OGSOD-1.0/sar/images/test
- Split count differs from CoLD/CCLKD paper: train 0, test 0; expected 14665/3666.
- No label files found; dataset root may be unavailable on this machine.
- No images available for dimension sampling.
