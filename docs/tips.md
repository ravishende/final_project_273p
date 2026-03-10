# Tips

- after pip installing the libraries in requirements.txt (commands in the readme), use nbdime for viewing git diffs of jupyter notebooks
  - `nbdime config-git --enable --global` will set it so if you run `git diff ____.ipynb`, you will see a pretty git diff rather under-the-hood json diffs
