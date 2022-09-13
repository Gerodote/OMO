# OMO_lab1
a laboratory work for basics of computational methods. ( for my uni )


## How to run this?
1.  Install python3 , desirable versions are 3.10.x or 3.9.x . Depending on your system:
      
      1.  Windows, find [installer](https://www.python.org/downloads/windows/) or install [`scoop`](https://scoop.sh/), then `scoop install python` .
      2.  Linux, check is it exist in repositories of your distributive
      3.  MacOs, idk google it. Maybe in [brew](https://formulae.brew.sh/formula/python@3.9) ?
      4.  Android? [Termux](https://github.com/termux/termux-app/releases) is your best friend.
2.  `git clone https://github.com/Gerodote/OMO_lab1.git` . If you haven't, install git. Again, in which way to do so depends on what's your system is 
3.  `python -mpip install --upgrade pip` . Depending on your system, the beginning of the command can be `python3`
3.  `pip install -r requirements.txt` or try `pip install sympy pytest`. If your CPU has ARM architecture, strongly recommend type `pip install wheel` before this command. Especially, if you run in [termux](https://github.com/termux/termux-app/releases)
4.  `pytest` in directory with project to check are tests working.
