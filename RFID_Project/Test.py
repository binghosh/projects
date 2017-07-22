import Tkinter as tk
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import io
import sys
import pandas as pd

root = tk.Tk()



fields = 'Antenna-Width', 'Antenna-Breadth'

def callback():
   print ("hello from the otherside")

def fetch(entries):
   dict1 = []
   for entry in entries:
      field = entry[0]
      text  = entry[1].get()
      print('%s: "%s"' % (field, text))
      dict1.append(field)
   print dict1
   df = pd.DataFrame.from_dict(dict1)


   print df
   

def makeform(root, fields):
   entries = []
   for field in fields:
      row = tk.Frame(root)
      lab = tk.Label(row, width=15, text=field, anchor='w')
      ent = tk.Entry(row)
      row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
      lab.pack(side=tk.LEFT)
      ent.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
      entries.append((field, ent))
   return entries

if __name__ == '__main__':
   
   ents = makeform(root, fields)
   #root.bind('<Return>', (lambda event, e=ents:))   
   b1 = tk.Button(root, text='Show',
          command=(lambda e=ents: fetch(e)))
   b1.pack(side=tk.LEFT, padx=5, pady=5)
   b2 = tk.Button(root, text='Quit', command=root.quit)
   b2.pack(side=tk.LEFT, padx=5, pady=5)

c1 =tk.Canvas(root, width=500, height=15, borderwidth=0, background='white')
c1.pack()
c1.create_text(250,10,text="Please click on the cordinates which you want to measure and move hight wise")
c = tk.Canvas(root, width=500, height=500, borderwidth=5, background='white')
c.pack()

c.bind("<Button-1>", callback)


root.mainloop()
