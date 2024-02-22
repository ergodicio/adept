FAQ
-----

**Q. Why use MLflow?**

A. MLFlow handles all the incredibly rote metadata management that computational scientists have historically either just 
completely ignored, written in an excel file, used a lab notebook, etc (You may always be an exception!).

You can store parameters (inputs to the simulation, box size, driver parameters, etc.), metrics (run time, total electrostatic energy, temperature at t=200 ps etc.) 
and artifacts (the fields, distribution functions, plots, post processed quantities, configuration etc.) in a single place. 

This place can either be your local machine, or better yet, a remote server that is backed by a database and an object store.

---------------------

**Q. Why use xarray?**

A. Xarray is a great way to handle gridded data. It is performant, has a stable API, has high level plotting features. It is fairly portable, maybe not as much as HDF5, but it is a netCDF4 file so 
it can't be that bad! 

---------------------

**Q. Why use diffrax?**

A. Diffrax provides the ODE integrator capabilities. However, you can, and we often do, side-step the actual time-integrator but only use diffrax for yet again, a stable API that enables us to 
save data in a consistent way. Diffrax lets us pass functions to the integrator, which is a great way to store custom post-processed (e.g. interpolated) quantities. You can also handle the result 
in the same consistent way. Yes, we could have just designed an API. But diffrax DOES also provide the time integrator.

Another thing diffrax does is it has a great loop handling system that compiles much faster than anything I have written. I don't know why that is, but it is. 
