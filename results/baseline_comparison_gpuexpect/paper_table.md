| Molecule | n_qubits | n_pauli | Backend   | Device            | Wall (s) | s/iter | Iters | E (Ha)    | |Δ FCI| (mHa) | Speedup vs hpchybrid |
|----------|----------|---------|-----------|-------------------|----------|--------|-------|-----------|---------------|----------------------|
| H2       | 4        | 15      | hpchybrid | gpu-custatevec    | 0.60     | 0.0060 | 100   | -1.2015   | 64.21         | 1.00x                |
| H2       | 4        | 15      | lightning | gpu-lightning.gpu | 0.81     | 0.0081 | 100   | -0.5838   | 553.46        | 0.74x                |
| H2       | 4        | 15      | aer-mpi   | gpu-aer-blocking  | 0.58     | 0.0058 | 100   | -0.9745   | 162.80        | 1.05x                |
| LiH      | 12       | 631     | hpchybrid | gpu-custatevec    | 9.76     | 0.0976 | 100   | -5.4418   | 2440.67       | 1.00x                |
| LiH      | 12       | 631     | lightning | gpu-lightning.gpu | 8.19     | 0.0819 | 100   | -7.1430   | 739.50        | 1.19x                |
| LiH      | 12       | 631     | aer-mpi   | gpu-aer-blocking  | 10.15    | 0.1015 | 100   | -5.2676   | 2614.90       | 0.96x                |
| BeH2     | 14       | 666     | hpchybrid | gpu-custatevec    | 11.31    | 0.1131 | 100   | -14.5939  | 1001.19       | 1.00x                |
| BeH2     | 14       | 666     | lightning | gpu-lightning.gpu | 9.30     | 0.0930 | 100   | -15.4205  | 174.66        | 1.22x                |
| BeH2     | 14       | 666     | aer-mpi   | gpu-aer-blocking  | 17.32    | 0.1732 | 100   | -14.0684  | 1526.71       | 0.65x                |
| H2O      | 14       | 1086    | hpchybrid | gpu-custatevec    | 15.24    | 0.1524 | 100   | -80.3738  | 5361.35       | 1.00x                |
| H2O      | 14       | 1086    | lightning | gpu-lightning.gpu | 11.38    | 0.1138 | 100   | -75.8364  | 824.01        | 1.34x                |
| H2O      | 14       | 1086    | aer-mpi   | gpu-aer-blocking  | 20.92    | 0.2092 | 100   | -73.0126  | 1999.87       | 0.73x                |
| NH3      | 16       | 3057    | hpchybrid | gpu-custatevec    | 21.56    | 0.4312 | 50    | -59.5961  | 4077.83       | 1.00x                |
| NH3      | 16       | 3057    | lightning | gpu-lightning.gpu | 23.37    | 0.4673 | 50    | -57.5310  | 2012.72       | 0.92x                |
| NH3      | 16       | 3057    | aer-mpi   | gpu-aer-blocking  | 81.08    | 1.6217 | 50    | -54.4544  | 1063.89       | 0.27x                |
| N2       | 20       | 2951    | hpchybrid | gpu-custatevec    | 25.92    | 0.5185 | 50    | -110.5037 | 2855.54       | 1.00x                |
| N2       | 20       | 2951    | lightning | gpu-lightning.gpu | 25.61    | 0.5123 | 50    | -110.6632 | 3015.05       | 1.01x                |
| N2       | 20       | 2951    | aer-mpi   | gpu-aer-blocking  | 222.73   | 4.4547 | 50    | -109.7075 | 2059.31       | 0.12x                |

[aggregate_baseline] 18 runs loaded, 1 device class(es): {'gpu'}
