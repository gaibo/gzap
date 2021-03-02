## GZAP - flexible Python data analysis package

**This is a work in progress.**

---

Current organization of the folders:

* *data* - raw data files; *bbg_automated_pull.csv* contains useful test data 
pulled via script from Bloomberg

* *misc* - development-related scratch files

* *model* - data structures

* *output* - example outputs

* *utility* - useful functions for cleaning data, visualization, etc.

* */* - example scripts

---

Usage notes:

1. You'll want to set your Windows environment variables $MPLCONFIGDIR and
   $MATPLOTLIBRC to "C:\path\to\where\you\installed\gzap\.matplotlib\" so you
   can access custom matplotlib graphical defaults and styles. You'll need to
   restart PyCharm (or other IDE) for the new variables to kick in.
