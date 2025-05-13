# SLA Segmentation Tool (SST)

This tool can analyze plant scans and determine the specific leaf area (SLA) via segmentation. For this, it uses a flood-fill approach to segment the pixels of the plants in the image, and then calculate the leaf area using the number of pixels segmented.

The tool uses python and runs in a Jupyter notebook.

## Run

To run a notebook, create a python environment with python 3 (recommended 3.8 or higher) and install the requirements:

```
pip install -r requirements.txt
```

Then run the tool using `batch launch_st.bat` on Windows or `bash launch_st.sh`on Linux. This will launch the Jupyter server and open the tool in your web browser.

## Parameters

The tool requires different user-defined parameters to be used. Some parameters affect the behaviour of the algorithm and others are paths for saving the results. Each of the relevant parameters is provided in the Jupyter notebook with a sensible default value, explained in detail and can be adapted to the user's requirements.
