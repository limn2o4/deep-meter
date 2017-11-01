# deep-meter
A computer vision project

Well I use some CNN work to recognize digital words

and normal way to calculate the angle

dev dir:
2017.10.25:
   I try to find another way to feed data but It's not work.BUt I still fix a little bugs in input_data.py
2017.10.26:
    I lower the learning rate with random init and it's work fantastic in single number maybe it's related to init but
    if I change the number it's not work again......
    with 10000+ times train it's work better with same data
    but I still don't know why change number make big different
    I found : fix data by num -> low lr(0.0001) 2000pro
2017.11.1
    I find my network have special for number 5......I should fix that problem first
    mabey it's time to change the init way and stay the network as it is