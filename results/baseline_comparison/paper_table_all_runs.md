| Molecule | n_qubits | n_pauli | Backend   | Device            | Wall (s) | s/iter | Iters | E (Ha)   | |Δ FCI| (mHa) | Speedup vs hpchybrid |
|----------|----------|---------|-----------|-------------------|----------|--------|-------|----------|---------------|----------------------|
| H2       | 4        | 15      | hpchybrid | gpu-custatevec    | 0.56     | 0.0056 | 100   | -1.2015  | 64.21         | 1.00x                |
| H2       | 4        | 15      | lightning | gpu-lightning.gpu | 0.67     | 0.0067 | 100   | -0.5838  | 553.46        | 0.83x                |
| H2       | 4        | 15      | aer-mpi   | gpu-aer-blocking  | 0.62     | 0.0062 | 100   | -0.9745  | 162.80        | 0.89x                |
| LiH      | 12       | 631     | hpchybrid | gpu-custatevec    | 7.33     | 0.0733 | 100   | -5.4418  | 2440.67       | 1.00x                |
| LiH      | 12       | 631     | lightning | gpu-lightning.gpu | 6.53     | 0.0653 | 100   | -7.1430  | 739.50        | 1.12x                |
| LiH      | 12       | 631     | aer-mpi   | gpu-aer-blocking  | 6.93     | 0.0693 | 100   | -5.2676  | 2614.90       | 1.06x                |
| BeH2     | 14       | 666     | hpchybrid | gpu-custatevec    | 12.13    | 0.1213 | 100   | -14.5939 | 1001.19       | 1.00x                |
| BeH2     | 14       | 666     | lightning | gpu-lightning.gpu | 7.52     | 0.0752 | 100   | -15.4205 | 174.66        | 1.61x                |
| BeH2     | 14       | 666     | aer-mpi   | gpu-aer-blocking  | 11.85    | 0.1185 | 100   | -14.0684 | 1526.71       | 1.02x                |
| H2O      | 14       | 1086    | hpchybrid | gpu-custatevec    | 18.69    | 0.1869 | 100   | -80.3738 | 5361.35       | 1.00x                |
| H2O      | 14       | 1086    | lightning | gpu-lightning.gpu | 10.44    | 0.1044 | 100   | -75.8364 | 824.01        | 1.79x                |
| H2O      | 14       | 1086    | aer-mpi   | gpu-aer-blocking  | 18.54    | 0.1854 | 100   | -73.0126 | 1999.87       | 1.01x                |

[aggregate_baseline] 12 runs loaded, 1 device class(es): {'gpu'}
