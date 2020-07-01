# Virtual self driving car :car:

<img style="display: block; margin-left: auto; margin-right: auto;" src="./output.gif" />


## About project :information_source:
Virtual self driving car is a reinforcement learning educational mini-project developed using two awesome frameworks of <a href="https://python.org/">Python</a>:
 1. <a href="https://kivy.org/">Kivy</a> framework :kiwi_fruit: (used for GUI as environment) and <BR />
 2. <a href="https://pytorch.org/">PyTorch</a> framework :fire: (used for reinforcement learning). <BR />

* _Each source code file contains comments to explain meaning of that code along with some reference links._ <BR />
* _Setup instructions below also provided in **Instructions.txt** file of project root, so you can follow there if required._ 

## Project dependencies setup instructions :hammer_and_wrench::chains:
It is highly recommended that to create a virtual environment using 'conda' because, there are lot of dependencies which will be installed automatically. <BR />
Project requires two packages mainly along with their dependencies (which will installed automatically with them), also read '**NOTE**' sections for each packages:
 1. <a href="https://kivy.org/">Kivy</a> framework :kiwi_fruit: (_Nightly build_): <BR />
    * _**Install command**_ -> <BR />
        &emsp; `pip install --upgrade kivy[base] kivy_examples --pre --extra-index-url https://kivy.org/downloads/simple/`
    * _**NOTE**_ : <BR />
        After you installed the 'Kivy' framework using above command, check installed version using command in your virtual environment: <BR />
        &emsp; `python -c "import kivy; kivy.__version__; exit(0);"` <BR />
        There will be some command output such as, <BR />
        &emsp; `[INFO   ] [Kivy        ] v2.0.0rc3, git-20c14b2, 20200615` <BR />
        Where, **v2.0.0rc3** means installed version is **2.0.0**, similarly check your installed version & update accordingly in first line of "**_car.kv_**" file only if program while running gives error related to _Kivy_. <BR />

 2. <a href="https://kivy.org/">PyTorch</a> framework :fire: (_Using 'pip' command_): <BR />
    * _**NOTE**_ : <BR />
    &emsp; Make sure to check official <a href="https://pytorch.org/">PyTorch</a> site & install latest version instead of **1.5.1**or **0.6.1** as they're latest at the time of writing. <BR />
    * _**Install CPU only version (Recommended & used in this project)**_ -> <BR />
    &emsp; `pip install torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html` <BR />
    * _**Install GPU version (No GPU used in this project)**_ -> <BR />
    &emsp; `pip install torch===1.5.1 torchvision===0.6.1 -f https://download.pytorch.org/whl/torch_stable.html` <BR />

## Finally execute :racing_car:
 1. Activate virtual environment containing above package installation in project root.
 2. Enter the command: `python map.py` <BR /> <BR />
