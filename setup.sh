pip install matplotlib==2.0.2
pip install six==1.10.0
pip install python-tk


# ##############如果遇到此問題##################
# 1. ModuleNotFoundError: No module named 'tkinter.filedialog'
#>> sudo apt-get install python3-tk
#如果輸入上面指令有錯
# 2. You might want to run 'apt-get -f install' to correct these:
# The following packages have unmet dependencies:
#  libpython3.6-stdlib : Depends: libpython3.6-minimal (= 3.6.5-5~16.04.york0) but 3.6.5-5~16.04.york1 is to be installed
#  python3.6 : Depends: libpython3.6-stdlib (= 3.6.5-5~16.04.york1) but 3.6.5-5~16.04.york0 is to be installed
# E: Unmet dependencies. Try 'apt-get -f install' with no packages (or specify a solution).
#>>sudo apt-get -f install
#如果還有下列問題
# 3. Errors were encountered while processing:
#  /var/cache/apt/archives/libpython3.6-stdlib_3.6.5-5~16.04.york1_amd64.deb
# E: Sub-process /usr/bin/dpkg returned an error code (1)
#>>sudo dpkg --install --force all /var/cache/apt/archives/libpython3.6-stdlib_3.6.5-5~16.04.york1_amd64.deb
###########################################
