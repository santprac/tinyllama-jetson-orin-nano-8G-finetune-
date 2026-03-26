  
sudo nvpmodel -m 0
sudo jetson_clocks

echo 1020000000 | sudo tee /sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/min_freq
echo 1020000000 | sudo tee /sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/max_freq
