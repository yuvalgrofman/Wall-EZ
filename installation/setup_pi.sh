#!/bin/bash

# ==========================================================
# Raspberry Pi 4 Setup Script - Projects (MAGDAD/Robot)
# ==========================================================

echo "--- Starting Raspberry Pi Setup ---"

# 1. Update and Upgrade System
echo "Updating system packages..."
sudo apt update
sudo apt upgrade -y

# 2. Install Core Dependencies and Tools
echo "Installing core tools (Git, Pip, Venv, I2C Tools, Build tools)..."
# Added wget, unzip, and build-essential to ensure we can build from source if needed
sudo apt install -y git python3-pip python3-venv i2c-tools libgpiod-dev wget unzip build-essential

# 3. Install pigpio (Robust Method with Source Fallback)
echo "Installing pigpio..."
if sudo apt install -y pigpio python3-pigpio; then
    echo "pigpio successfully installed via apt."
else
    echo "Apt installation failed. Building pigpio from source..."
    # Download and extract
    cd /tmp
    wget https://github.com/joan2937/pigpio/archive/master.zip
    unzip -o master.zip
    cd pigpio-master
    
    # Compile and install
    make
    sudo make install
    
    # CRITICAL FIX: Refresh system shared libraries so 'pigpiod' command is recognized
    sudo ldconfig
    
    # Clean up
    cd /tmp
    sudo rm -rf pigpio-master master.zip
    cd ~
fi

# Enable and start the daemon so it runs on boot
echo "Enabling and starting pigpiod service..."
sudo systemctl enable pigpiod
sudo systemctl start pigpiod

# 4. Install OpenCV Dependencies
echo "Installing OpenCV and GUI dependencies..."
sudo apt install -y python3-opencv libcanberra-gtk-module libcanberra-gtk3-module

# 5. Create Virtual Environment (Optional but recommended)
echo "Setting up Python Virtual Environment (named 'env')..."
python3 -m venv my_env
source my_env/bin/activate

# 6. Install Python Packages inside the Venv
echo "Installing Python libraries (Matplotlib, NumPy, etc.)..."
pip install --upgrade pip
pip install numpy matplotlib spidev

# 7. Final Configuration Notes
echo "----------------------------------------------------------"
echo "Setup complete!"
echo ""
echo "REMINDER: Don't forget to run 'sudo raspi-config' to:"
echo "1. Enable I2C"
echo "2. Enable SPI"
echo "3. Enable Camera (if needed)"
echo "4. Set Wi-Fi Country"
echo ""
echo "To test if pigpio is working, run: pigs t"
echo "To activate your python environment, run: source env/bin/activate"
echo "----------------------------------------------------------"