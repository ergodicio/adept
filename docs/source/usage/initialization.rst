The Tanh profile
-----------------------

This profile is used for spatial and temporal envelopes everywhere in this code. 

This is different per simulation type but there are some consistent concepts. These are that the density can be defined
as a function of space, the driver (antenna or ponderomotive) can be a function of time and space, and the collision frequency 
can be a function of time and space.

That is, you can specify,

.. math:: 
    n(x), E(t, x), \nu(t, x)

Each of these profiles is provided using a tanh function that is parameterized using

``p_{wL}``, ``p_{wR}`` - the rise and fall of the flat-top

``p_w`` - width of the flat-top

``p_c`` - center of the flat-top

``p_L = p_c - p_w / 2`` - left edge of the flat-top

``p_R = p_c + p_w / 2`` - right edge of the flat-top

where p can be the time or space coordinate depending on the context

Then, the overall shape is given by

.. math:: 
    f(p) = 0.5 * \tanh((ax - p_L) / p_{wL}) - \tanh((ax - p_R) / p_{wR})

where ``ax`` is the time or space axis

If you plot this, it looks like a flat top centered at ``p_c`` and has a width ``p_w``, and a rise and fall of ``p_{wL}`` and ``p_{wR}`` respectively.





