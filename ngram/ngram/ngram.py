import reflex as rx
from rxconfig import config

from ngram.model import *

class K:
    def __init__(self,n=3):
        self.n = n
    
    def suggest_text(to_list=0, n_words=4, n_texts=10):
        return [['F', 'a', '5', 'ggg9']]*10

ngram = 3
ts = create('C:/Users/artem/Desktop/NLP/HW1/emails.csv', n=ngram)
#ts = K()

def render(begin):
    def render_pairs(pair):
        return rx.text(
            rx.text.strong(begin),
            pair,
            as_="p",
        )
    return render_pairs

class State(rx.State):
    """The app state."""
    user_input: str = ""
    out: list = []
    
    def get(self, it):
        self.user_input = it
        to_list = inp_string(self.user_input, ngram)
        print(to_list)
        predict = ts.suggest_text(to_list, n_words=4, n_texts=10)
        print(predict)
        self.out = out_string(to_list, predict)
        #self.out = list_pred[0] if len(self.user_input) !=0 else ''
        #self.out = ['F', 'a', '5', 'ggg']
    
def index() -> rx.Component:
    return rx.container(
        rx.color_mode.button(position="top-right"),
        rx.vstack(
            rx.heading("Сontinuation", size="8"),
            rx.input(
                placeholder="Type something...",
                value=State.user_input,
                on_change=State.get,\
            ),
            rx.foreach(State.out, render(State.user_input)),
            pacing="5",
            justify="center",
            min_height="85vh",
        ),
    )

app = rx.App()
app.add_page(index)