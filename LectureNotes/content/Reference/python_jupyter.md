# Python and Jupyter notebooks

This is a rapid introduction to the basics of Python and the use of Jupyter notebooks.

Good online cheatsheets:



What you should know from [](/notebooks/Reference/Jupyter_Python_intro_01.ipynb):

1. Know how to switch between `Code` and `Markdown` cells and how to run them (`shift-Return` or `Run` button)

1. Basics of Python evaluation and printing
    * string concatenation
    * exponentiation with `**`
    * use of fstrings

1. Importing numpy and basic functions (sqrt, exp, sin, $\ldots$)
    * numpy arrays using `arange(min, max, step)`

1. Getting help via Google (or alternative): StackOverflow and manuals

1. Defining functions in Python
    * `def my_function(x):` $\longleftarrow$ semicolon and then indented lines
    * position vs. keyword argument and defaults
    * `shift + tab + tab` to reveal definitions

1. Plotting with Matplotlib
    * Standard sequence for plot: data $\longrightarrow$ make figure $\longrightarrow$ add subplots $\longrightarrow$ make plot
    * dressing up and saving a plot
    * names for the figure and axis objects are our choice

    <!---
    * `%matplotlib inline` to generate inline plots in Jupyter notebooks
    -->

1. Numpy linear algebra
    * defining a matrix and finding its shape or particular elements
    * matrix (or vector) operations: multiplying them with @, transpose, trace, inverse, eigensolution    