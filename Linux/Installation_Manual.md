# Process to install anaconda, Intellij, and GitHub Desktop on Chromebook (Debian 10)

## Anaconda
Go to https://www.anaconda.com/products/individual to download the Anaconda installer for Linux.

Then Copy the downloaded files into Linux file, navigate to the target directory (usually it just start in Linux file directory already), then type ./Anaconda..., this will then install Anaconda/Jupyter/Python 3.8 automatically

## Intellij

### Install Java 17 JDK
Reference: https://computingforgeeks.com/install-oracle-java-openjdk-on-debian-linux/

sudo apt update
sudo apt -y install wget curl

wget https://download.oracle.com/java/17/latest/jdk-17_linux-x64_bin.deb

sudo apt install ./jdk-17_linux-x64_bin.deb

cat <<EOF | sudo tee /etc/profile.d/jdk.sh
export JAVA_HOME=/usr/lib/jvm/jdk-17/
export PATH=\$PATH:\$JAVA_HOME/bin
EOF

source /etc/profile.d/jdk.sh
java -version

If this returns without error, then Java 17 JDK has been installed succesfully.

### Install Intellij
sudo apt update 

sudo apt install libsquashfuse0 squashfuse fuse 
sudo apt install snapd

sudo snap install intellij-idea-community --classic

launch it by navigating to cd /snap/intellij-idea-community/323/bin/ and do ./idea.sh


## GitHub Desktop
wget -qO - https://packagecloud.io/shiftkey/desktop/gpgkey | sudo apt-key add -

sudo sh -c 'echo "deb [arch=amd64] https://packagecloud.io/shiftkey/desktop/any/ any main" > /etc/apt/sources.list.d/packagecloud-shiftky-desktop.list'

sudo apt-get update

sudo apt install github-desktop

After succesful install, launch it by typing github-desktop.


## R (Newest Version)
https://cran.r-project.org/bin/linux/debian/#supported-branches

Add
deb http://cloud.r-project.org/bin/linux/debian buster-cran40/
to /etc/apt/sources.list by typing
echo "new line of text" | sudo tee -a /etc/apt/sources.list

apt update

If it says that some key are missing, add the key by using:

sudo apt-key adv --keyserver keys.gnupg.net --recv-key (missing_key)

then run apt update again

apt install -t buster-cran40 r-base