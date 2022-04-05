# Process to install anaconda, Intellij, and GitHub Desktop on Chromebook (Debian 10)

## Anaconda
Go to https://www.anaconda.com/products/individual to download the Anaconda installer for Linux.

Then Copy the downloaded files into Linux file, navigate to the target directory (usually it just start in Linux file directory already), then type ./Anaconda..., this will then install Anaconda/Jupyter/Python 3.8 automatically

## Intellij

### Install Java 17 JDK
Reference: https://computingforgeeks.com/install-oracle-java-openjdk-on-debian-linux/
Firstly, install wget
```console
sudo apt update
sudo apt -y install wget curl
```
Then using wget, download the .deb file assocaited with Java 17 JDK, then install the .deb file
```console
wget https://download.oracle.com/java/17/latest/jdk-17_linux-x64_bin.deb
sudo apt install ./jdk-17_linux-x64_bin.deb
```
Add jdk-17 to path
```console
cat <<EOF | sudo tee /etc/profile.d/jdk.sh
export JAVA_HOME=/usr/lib/jvm/jdk-17/
export PATH=\$PATH:\$JAVA_HOME/bin
EOF

source /etc/profile.d/jdk.sh
```
To Test whether the installation has been successful, type:
```console
java -version
```

### Install Intellij
Install some dependencies and snapd which contain intellij 
```console
sudo apt update 

sudo apt install libsquashfuse0 squashfuse fuse 
sudo apt install snapd
```
Install Intellij using snapd
```console
sudo snap install intellij-idea-community --classic
```
To launch Intellij, navigate to /snap/intellij-idea-community/323/bin/ using "cd" and "ls" and do 
```console
./idea.sh
```

## GitHub Desktop (Unstable, follow other tutorial to directly interact with GitHub using Terminal)
```console
wget -qO - https://packagecloud.io/shiftkey/desktop/gpgkey | sudo apt-key add -

sudo sh -c 'echo "deb [arch=amd64] https://packagecloud.io/shiftkey/desktop/any/ any main" > /etc/apt/sources.list.d/packagecloud-shiftky-desktop.list'

sudo apt-get update

sudo apt install github-desktop
```
After succesful install, launch it by typing
```console
github-desktop.
```

## R (Newest Version)
https://cran.r-project.org/bin/linux/debian/#supported-branches

Add url of the newest version of R to /etc/apt/sources.list by typing
```console
echo "deb http://cloud.r-project.org/bin/linux/debian buster-cran40/" | sudo tee -a /etc/apt/sources.list
```
Then update apt and install r-base using the updated apt
```console
apt update
apt install -t buster-cran40 r-base
```

Caveteat: If it says that some key (missing_key) are missing, add the key by using:
```console
sudo apt-key adv --keyserver keys.gnupg.net --recv-key (missing_key)
```
then run apt update again

