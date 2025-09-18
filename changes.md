# Changes

## v5.0.0

### Changed

* ENV: Updated conda environment.yml to Python 3.11 and latest versions of libraries July 2024/
* ENV: Updated JupyterLab is 4.4.7 (sept 25)

### Fixed

* Patched `geopandas` > 1.0.0 breaking changes:
    * `read_file()` now returns a `DataFrame` if the file contains no geometry. Code updated to avoid runtime error.
    * `read_file()` does not require `crs` option. Code updated to remove runtime warning.
    * LAB 1: Modified introduction text for and exercise 4. Completed for Solutions + Student copy of notebook.

* Patched Lab 2:
    * numpy >= 2.0 deprecates np.Inf. Replaced with np.inf
    * Removed unnecessary sort of result array from Genetic algorithm code.

* Patched `metapy`. Version 0.2.0. np.Inf -> np.inf for `HillClimber`


## v4.0.0
* Updated conda environment.yml to Python 3.10 and latest versions of libraries Dec 2023.
* Updated Lab1
    * removed stamin base maps as these are no longer supported
    * Update inset map code to use new version of `matplotlib` API


## v3.0.0
* Tested for Academic Year 2022/23
* Updated conda environment.yml to latest versions of libraries
