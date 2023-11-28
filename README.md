<br/>
<p align="center">
  <a href="https://github.com/NotFrancee/mpl_bsic">
    <img src="images/logo.png" alt="Logo" height="80">
  </a>

  <h3 align="center">BSIC FX Systematic Strategy Article</h3>

  <p align="center">
    Code for the article on Systematic FX
    <br/>
    <br/>
    <a href="https://bsic.it/what-works-today-might-not-work-tomorrow-overview-of-a-systematic-fx-strategy/"><strong>Read the article »</strong></a>
  </p>
</p>

## About the Code

The code is grouped in the following directories: 

* `backtester`: code for the backtesting classes, which include the strategy as described by the paper, our improved version of the signal, and a signal which uses a rolling z-score instead of a simple Rolling MA
* `notebooks`: code for creating the plots for the article, and the notebook where we performed the factor analysis
* `utils`: code to load the data for currencies, swaps, cpi, and EM
* `other`: the code used to run the strategies using scripts and to run robustness checks