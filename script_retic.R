#this script requires the Anaconda IDE for Python
#https://www.anaconda.com/products/individual

#it also requires the package reticulate
#install.packages("reticulate")
library(reticulate)

#initiate a python session, then type "quit" into the console
repl_python()

#py$ is used to specify an object in the Python environment
#you can switch between viewing the global R and Python environments
#by clicking on the "R v" button to the left of "Global Environment"

#define the path where the scans are located
py$dirname = "C:/Users/Nutzer/allscans" 

#define the resolution the leaves were scanned at
py$dpi = 300 
  
#define by how much the picture should be cropped before calculation
py$border_crop_relative = 0.02  

#define a csv file to write your results to
py$save_path = "C:/Users/Nutzer/la_allscans.csv" 
  
#this runs the python script from the specified location
#this step might take a while...
#a preview of the calculated area will be shown in the Plots window
py_run_file("C:/Users/Nutzer/Segmentation Tool.py")

#we can also take the resulting dataframe py$df from the python environment
#and transform it into a regular dataframe to work with right here
results <- py$df
