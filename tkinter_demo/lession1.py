import tkinter as tk

window = tk.Tk()
window.title("我的第一个窗口")
window.geometry('1000x500')
var = tk.StringVar()
on_hit = False


def hit_me():
    global on_hit
    if on_hit:
        on_hit = False
        var.set('')
    else:
        on_hit = True
        var.set('你点击了我')
    pass


et = tk.Entry(window, show='*')
et.pack()

txt = tk.Text(window, height=2)
txt.pack()


def get_edit():
    var = et.get()
    txt.insert('end',var)


btn = tk.Button(window,
                text='获取edit',
                comman=get_edit,
                width=15, height=2
                )
btn.pack()
lb = tk.Label(window,
              textvariable=var,
              bg='green',
              font=('Arial', 12),
              width=15, height=2
              )
lb.pack()
btn = tk.Button(window,
                text='点击我',
                comman=hit_me,
                width=15, height=2
                )
btn.pack()

window.mainloop()
