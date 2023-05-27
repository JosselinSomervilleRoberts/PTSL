# Save current path
pwd=`pwd`

# Install Mujoco
cd /home/ubuntu
mkdir .mujoco
cd .mujoco
wget https://www.roboti.us/file/mjkey.txt # Key
wget https://www.roboti.us/download/mujoco200_linux.zip # Mujoco 200
yes y | sudo apt-get install unzip
unzip mujoco200_linux.zip
mv mujoco200_linux mujoco200
rm -r -f mujoco200_linux.zip
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz # Mujoco 210 (Not needed)
tar -xvf mujoco210-linux-x86_64.tar.gz
rm -r -f mujoco210-linux-x86_64.tar.gz
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/.mujoco/mujoco200/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia' >> ~/.bashrc
source ~/.bashrc
cd $pwd

# Create conda env and activate it
conda env create -f environment.yml
conda init bash
source ~/.bashrc
conda activate mtrl

# Finish mujoco installs
yes y | sudo apt-get install patchelf
yes y | sudo apt-get install libglew-dev
yes y | pip install mujoco
yes y | pip install scipy
yes y | sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3

# Additional installs (metaworld, mtenv)
pip install git+https://github.com/Farama-Foundation/Metaworld.git@af8417bfc82a3e249b4b02156518d775f29eb28 -y
pip install "mtenv[metaworld]" -y
pip install git+https://github.com/JosselinSomervilleRoberts/JossPythonToolbox.git -y
pip install wandb -y

