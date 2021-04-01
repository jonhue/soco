---
layout: primer_without_heading
title: Publicly share LaTeX sources
---

# Publicly share LaTeX sources

1. Clone this template
2. [Setup GitHub Pages deployments](https://github.com/peaceiris/actions-gh-pages#%EF%B8%8F-create-ssh-deploy-key)
3. List all `.tex` files you want to compile in `.github/workflows/publish.yml`
4. Enable GitHub Pages at the root directory in your `master` branch

From now on, the `master` branch will host your deployed files while you commit your changes to `sources`.

You may do additional setup work in `scripts/setup.sh`.

## Documents

* [example](https://jonhue.github.io/latex/example.pdf)
