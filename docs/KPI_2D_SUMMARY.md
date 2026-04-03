# KPI 2D Summary

Source: `results/unet2d_kpi.json`

## KPI2 - Debit patients/seconde

Measured with `workers=0` in the latest quick run:

| Strategy | Patients/s |
| --- | ---: |
| DirectFiles | 24.016 |
| MongoBinary | 208.894 |
| MongoPolygons | 1.644 |

## KPI3 - Occupation disque pour 100 patients

| Strategy | Image (Go) | Mask (Go) | Meta (Go) |
| --- | ---: | ---: | ---: |
| DirectFiles | 2.965 | 0.015 | 0.000 |
| MongoBinary | 0.881 | 0.270 | 0.024 |
| MongoPolygons | 0.000 | 0.005 | 0.003 |

## KPI4 - Temps de preparation ETL

| Strategy | Seconds |
| --- | ---: |
| DirectFiles | 0.00 |
| MongoBinary | 0.00 |
| MongoPolygons | 0.00 |

## Interpretation

- DirectFiles is easy to use but heavier on disk.
- MongoBinary is the fastest ingestion path in the current quick benchmark.
- MongoPolygons is the most compact for masks but slower than binary ingestion.
- The quick run did not include a non-zero one-shot ETL measurement, so KPI4 remains 0.00 s in the current JSON output.