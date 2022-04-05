# GitHub Interaction using Terminal
In order to interact with GitHub, we will install git first onto our debian system
```console
sudo apt update
sudo apt install git
```
To evaluate whether the installation was successful, type:
```console
git --version
```

These are the commonly used command to interact with GitHub.
```console
git clone <link> # To clone a repository to your local machine.
git fetch # To fetch any updates in your current repository
git pull # To be used after fetching, pull the changes into your local machine
git add <file> # Add any changes you've made for commit
git commit -m "commit message" # Commit the added changes with comments
git push # Push the commits into the repository
git checkout <branch_name> # Switch to another branch in this repository
```
