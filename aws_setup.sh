# Save current path
pwd=`pwd`

# Install Mujoco
echo "Installing mujoco..."
unzip .mujoco.zip
mv .mujoco ~/.mujoco

# OLD INSTALL: Mujoco is broken right now
# cd /home/ubuntu
# mkdir .mujoco
# cd .mujoco
# wget https://www.roboti.us/file/mjkey.txt # Key
# wget https://www.roboti.us/download/mujoco200_linux.zip # Mujoco 200
# yes y | sudo apt-get install unzip
# unzip mujoco200_linux.zip
# mv mujoco200_linux mujoco200
# rm -r -f mujoco200_linux.zip
# wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz # Mujoco 210 (Not needed)
# tar -xvf mujoco210-linux-x86_64.tar.gz
# rm -r -f mujoco210-linux-x86_64.tar.gz
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/.mujoco/mujoco200/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia' >> ~/.bashrc
source ~/.bashrc
cd $pwd
echo "Mujoco installed.\n"

# Create conda env and activate it
echo "Creating conda env..."
conda env create -f environment.yml
conda init bash
source ~/.bashrc
conda activate mtrl
echo "Conda env created.\n"

# Install Mujoco dependencies
echo "Installing mujoco dependencies..."
# pip install "cython<3"
# yes y | sudo add-apt-repository ppa:jonathonf/gcc
# yes y | sudo apt-get update
# yes y | sudo apt install gcc-7
# yes y | sudo apt-get install patchelf
# yes y | sudo apt-get install libglu1-mesa-dev mesa-common-dev
yes y | sudo apt update
yes y | sudo apt-get install patchelf
yes y | sudo apt-get install libglew-dev
yes y | sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3

# Finish mujoco installs
conda install -c conda-forge libstdcxx-ng=12
sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt-get dist-upgrade -y
sudo apt-get install build-essential software-properties-common -y
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo apt-get update -y
sudo apt-get install gcc-9 g++-9 -y
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9
sudo update-alternatives --config gcc
echo "Mujoco dependencies installed.\n"

# Install gym and mujoco-py
echo "Installing gym and mujoco-py..."
yes y | pip install gym==0.21.0
yes y | pip install mujoco-py==2.0.2.5
yes y | pip install scipy==1.9.1
echo "Gym and mujoco-py installed.\n"

# Additional installs (metaworld, mtenv)
echo "Installing additional dependencies..."
yes y | pip install git+https://github.com/Farama-Foundation/Metaworld.git@af8417bfc82a3e249b4b02156518d775f29eb28
yes y | pip install "mtenv[metaworld]"
yes y | pip install git+https://github.com/JosselinSomervilleRoberts/JossPythonToolbox.git
yes y | pip install wandb
yes y | pip install protobuf==3.20.0
echo "Additional dependencies installed.\n\n\n"

echo "Setup complete.\n"
echo "Now run:"
echo "sudo nano opt/conda/envs/mtrl/lib/python3.8/site-packages/metaworld/envs/mujoco/mujoco_env.py"
echo "to comment ou the maximum path.\n"
echo "Then run:"
echo "PYTHONPATH=. python3 -u main.py setup=metaworld agent=state_sac env=metaworld-mt10 agent.multitask.num_envs=10 agent.multitask.should_use_disentangled_alpha=True"

